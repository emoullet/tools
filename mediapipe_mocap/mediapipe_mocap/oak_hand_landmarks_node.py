# Copyright 2026 Etienne Moullet
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the Etienne Moullet nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import datetime
import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge  # noqa: F401
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Bool, Header

from mediapipe_mocap.hand_landmarks_common import (  # noqa: I100
    ensure_3_tuple,
    prepare_runtime_imports,
)
from mediapipe_mocap.landmark_processing import (
    LandmarkFilterBank,
    OneEuroFilterConfig,
    relative_points,
)
from mediapipe_mocap.mediapipe_runtime import (
    create_hand_landmarker_with_delegate,
    HandLandmarkerConfig,
    HandLandmarkerRuntime,
    parse_delegate_mode,
    PeriodicPerformanceTracker,
    timestamp_sec_from_header,
)
from mediapipe_mocap.oak_depth import (
    CameraIntrinsics,
    DepthProcessingConfig,
    OakDepthProjector,
    parse_missing_depth_strategy,
    prepare_metric_hand_for_output,
)
from mediapipe_mocap.reference import (
    ReferenceState,
    ResetRequestResult,
)
from mediapipe_mocap.viewer import (
    ControlOverlayConfig,
    HandLandmarksViewer,
    parse_overlay_normalization_mode,
)


prepare_runtime_imports()

import cv2  # noqa: E402,I100
import depthai as dai  # noqa: E402,I100
import mediapipe as mp  # noqa: E402,I100
from mediapipe.tasks.python import BaseOptions  # noqa: E402,I100
from mediapipe.tasks.python.vision import (  # noqa: E402,I100
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np  # noqa: E402,I100


class OakHandLandmarksNode(Node):
    """
    Capture OAK-D S2 RGBD frames and publish 3D hand landmarks.

    Output semantics for the first detected hand:
      - 21 points in MediaPipe order
      - point.x/y/z are reference-relative RGB-camera coordinates in meters
    """

    def __init__(self):
        """Initialize parameters, DepthAI capture, MediaPipe, and ROS interfaces."""
        super().__init__('oak_hand_landmarks_node')

        package_share_dir = get_package_share_directory('mediapipe_mocap')
        default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

        # Log active virtual environment for pip-installed MediaPipe/DepthAI debugging.
        virtual_env = os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')
        self.get_logger().info(f'Using venv: {virtual_env}')

        self.declare_parameters(
            namespace='',
            parameters=[
                ('landmarks_topic', '/hand_landmarks'),
                ('model_path', default_model_path),
                ('num_hands', 1),
                ('min_hand_detection_confidence', 0.5),
                ('min_hand_presence_confidence', 0.5),
                ('min_tracking_confidence', 0.5),
                ('delegate', 'AUTO'),
                ('camera_frame_id', 'oak_rgb_camera_optical_frame'),
                ('rgb_width', 640),
                ('rgb_height', 400),
                ('fps', 50.0),
                ('rgb_socket', 'CAM_A'),
                ('left_socket', 'CAM_B'),
                ('right_socket', 'CAM_C'),
                ('stereo_preset', 'FAST_DENSITY'),
                ('stereo_left_right_check', True),
                ('stereo_subpixel', False),
                ('stereo_extended_disparity', False),
                ('stereo_rectify_edge_fill_color', 0),
                ('sync_threshold_ms', 15.0),
                ('sync_attempts', -1),
                ('sync_run_on_host', True),
                ('depth_sample_radius_px', 2),
                ('min_depth_m', 0.12),
                ('max_depth_m', 3.0),
                ('depth_percentile', 50.0),
                ('missing_depth_strategy', 'reuse_last'),
                ('max_missing_depth_landmarks', 8),
                ('show_control_overlay', False),
                ('overlay_dead_zone', 0.05),
                ('overlay_saturation_zone', 0.3),
                ('overlay_normalization_mode', 'vector'),
                ('tracked_landmark_index', 0),
                ('initial_reference', [0.0, 0.0, 0.6]),
                ('auto_reference_on_first_detection', True),
                ('reset_reference_topic', '/reset_reference'),
                ('reset_reference_cooldown_sec', 0.25),
                ('enable_one_euro_filter', True),
                ('one_euro_mincutoff', 1.0),
                ('one_euro_beta', 0.1),
                ('one_euro_derivative_cutoff', 1.0),
                ('visualize', False),
                ('window_name', '3D Hand Landmarks OAK'),
            ],
        )

        self.landmarks_topic = self._get_str('landmarks_topic')
        self.model_path = self._get_str('model_path') or default_model_path
        self.num_hands = max(1, self._get_int('num_hands'))
        self.camera_frame_id = self._get_str('camera_frame_id')
        self.rgb_resolution = (
            max(1, self._get_int('rgb_width')),
            max(1, self._get_int('rgb_height')),
        )
        self.fps = max(1e-3, self._get_float('fps'))
        min_depth_m = max(0.0, self._get_float('min_depth_m'))
        self.depth_config = DepthProcessingConfig(
            sample_radius_px=max(
                0,
                self._get_int('depth_sample_radius_px'),
            ),
            min_depth_m=min_depth_m,
            max_depth_m=max(
                min_depth_m + 1e-6,
                self._get_float('max_depth_m'),
            ),
            percentile=min(
                100.0,
                max(0.0, self._get_float('depth_percentile')),
            ),
            missing_strategy=parse_missing_depth_strategy(
                self._get_str('missing_depth_strategy'),
                self.get_logger().warning,
            ),
            max_missing_landmarks=max(
                0,
                self._get_int('max_missing_depth_landmarks'),
            ),
        )
        self.show_control_overlay = self._get_bool('show_control_overlay')
        self.control_overlay = None
        if self.show_control_overlay:
            self.control_overlay = ControlOverlayConfig(
                dead_zone=max(0.0, self._get_float('overlay_dead_zone')),
                saturation_zone=max(
                    1e-6,
                    self._get_float('overlay_saturation_zone'),
                ),
                normalization_mode=parse_overlay_normalization_mode(
                    self._get_str('overlay_normalization_mode'),
                    self.get_logger().warning,
                ),
            )
        self.tracked_landmark_index = self._get_int('tracked_landmark_index')
        self.auto_reference_on_first_detection = self._get_bool(
            'auto_reference_on_first_detection'
        )
        self.reset_reference_topic = self._get_str('reset_reference_topic')
        self.reset_reference_cooldown_sec = max(
            0.0,
            self._get_float('reset_reference_cooldown_sec'),
        )
        self.visualize = self._get_bool('visualize')
        self.window_name = self._get_str('window_name')
        self.viewer = (
            HandLandmarksViewer(self.window_name)
            if self.visualize
            else None
        )
        self.enable_one_euro_filter = self._get_bool('enable_one_euro_filter')

        initial_reference_values = (
            self.get_parameter('initial_reference').get_parameter_value().double_array_value
        )
        initial_reference = ensure_3_tuple(
            initial_reference_values,
            [0.0, 0.0, 0.6],
            logger=self.get_logger(),
            parameter_name='initial_reference',
        )
        self.reference_state = ReferenceState(
            initial_position=initial_reference,
            cooldown_sec=self.reset_reference_cooldown_sec,
            initially_initialized=not self.auto_reference_on_first_detection,
        )

        one_euro_mincutoff = max(self._get_float('one_euro_mincutoff'), 1e-6)
        one_euro_beta = max(self._get_float('one_euro_beta'), 0.0)
        one_euro_derivative_cutoff = max(
            self._get_float('one_euro_derivative_cutoff'),
            1e-6,
        )
        self.landmark_filters = None
        if self.enable_one_euro_filter:
            self.landmark_filters = LandmarkFilterBank(
                self.num_hands,
                OneEuroFilterConfig(
                    frequency=self.fps,
                    min_cutoff=one_euro_mincutoff,
                    beta=one_euro_beta,
                    derivative_cutoff=one_euro_derivative_cutoff,
                ),
            )

        delegate_mode = parse_delegate_mode(
            self._get_str('delegate'),
            self.get_logger().warning,
        )
        landmarker = create_hand_landmarker_with_delegate(
            HandLandmarkerConfig(
                model_asset_path=self.model_path,
                num_hands=self.num_hands,
                min_hand_detection_confidence=self._get_float(
                    'min_hand_detection_confidence'
                ),
                min_hand_presence_confidence=self._get_float(
                    'min_hand_presence_confidence'
                ),
                min_tracking_confidence=self._get_float(
                    'min_tracking_confidence'
                ),
            ),
            requested_delegate=delegate_mode,
            base_options_type=BaseOptions,
            delegate_type=BaseOptions.Delegate,
            landmarker_options_type=HandLandmarkerOptions,
            landmarker_type=HandLandmarker,
            running_mode_type=RunningMode,
            info=self.get_logger().info,
            warn=self.get_logger().warning,
        )
        self._runtime = HandLandmarkerRuntime(landmarker)

        self.landmarks_pub = self.create_publisher(PointCloud, self.landmarks_topic, 10)
        self.reset_reference_sub = self.create_subscription(
            Bool,
            self.reset_reference_topic,
            self.reset_reference_callback,
            10,
        )

        self.depth_projector = None
        self.pipeline = None
        self.sync_queue = None
        self.capture_thread = None
        self.running = False
        self._debug_performance = PeriodicPerformanceTracker()

        self._build_and_start_pipeline()
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()

        self.get_logger().info(
            f'OakHandLandmarksNode started.\n'
            f'  landmarks_topic  = {self.landmarks_topic}\n'
            f'  model_path       = {self.model_path}\n'
            f'  running_mode     = VIDEO\n'
            f'  delegate         = {delegate_mode.value}\n'
            f'  rgb/fps          = {self.rgb_resolution[0]}x'
            f'{self.rgb_resolution[1]} @ {self.fps:.1f}\n'
            f'  output           = reference-relative metric camera coordinates\n'
            f'  depth_range      = [{self.depth_config.min_depth_m:.2f}, '
            f'{self.depth_config.max_depth_m:.2f}] m '
            f'radius={self.depth_config.sample_radius_px}px\n'
            f'  reset_reference  = {self.reset_reference_topic} '
            f'(auto_first={self.auto_reference_on_first_detection})\n'
            f'  control_overlay  = {self.show_control_overlay}\n'
            f'  one_euro_filter  = {self.enable_one_euro_filter}\n'
            f'  visualize        = {self.visualize}'
        )
        if self.control_overlay is not None:
            self.get_logger().info(
                'Control overlay enabled with '
                f'mode={self.control_overlay.normalization_mode.value}, '
                f'dead_zone={self.control_overlay.dead_zone:.3f}m, '
                f'saturation_zone='
                f'{self.control_overlay.effective_saturation_zone:.3f}m, '
                f'landmark_index={self.tracked_landmark_index}'
            )
        if self.enable_one_euro_filter:
            self.get_logger().info(
                'One Euro filter enabled with '
                f'frequency={self.fps:.3f} (camera fps), '
                f'mincutoff={one_euro_mincutoff:.3f}, '
                f'beta={one_euro_beta:.3f}, '
                f'derivative_cutoff={one_euro_derivative_cutoff:.3f}'
            )

    def _get_str(self, name):
        """
        Read a string ROS parameter.

        Parameters
        ----------
        name : str
            Name of a declared ROS parameter whose value is expected to be stored
            in the string field.

        Returns
        -------
        str
            Parameter value as a string.

        """
        return self.get_parameter(name).get_parameter_value().string_value

    def _get_bool(self, name):
        """
        Read a boolean ROS parameter.

        Parameters
        ----------
        name : str
            Name of a declared ROS parameter whose value is expected to be stored
            in the boolean field.

        Returns
        -------
        bool
            Parameter value as a bool.

        """
        return self.get_parameter(name).get_parameter_value().bool_value

    def _get_int(self, name):
        """
        Read an integer ROS parameter.

        Parameters
        ----------
        name : str
            Name of a declared ROS parameter whose value is expected to be stored
            in the integer field.

        Returns
        -------
        int
            Parameter value as an int.

        """
        return int(self.get_parameter(name).get_parameter_value().integer_value)

    def _get_float(self, name):
        """
        Read a floating-point ROS parameter.

        Parameters
        ----------
        name : str
            Name of a declared ROS parameter whose value is expected to be stored
            in the floating-point field.

        Returns
        -------
        float
            Parameter value as a float.

        """
        return float(self.get_parameter(name).get_parameter_value().double_value)

    def _camera_socket(self, parameter_name):
        """
        Resolve a DepthAI camera socket parameter with AUTO fallback.

        Parameters
        ----------
        parameter_name : str
            Name of the ROS parameter containing a DepthAI
            ``CameraBoardSocket`` member name, such as ``CAM_A``.

        Returns
        -------
        object
            DepthAI ``CameraBoardSocket`` enum value, or ``AUTO`` when the
            configured name is invalid.

        """
        socket_name = self._get_str(parameter_name).upper()
        if not hasattr(dai.CameraBoardSocket, socket_name):
            self.get_logger().warning(
                f"Invalid {parameter_name} '{socket_name}', falling back to AUTO."
            )
            return dai.CameraBoardSocket.AUTO
        return getattr(dai.CameraBoardSocket, socket_name)

    def _stereo_preset(self):
        """
        Resolve the configured DepthAI stereo-depth preset.

        Returns
        -------
        object
            DepthAI StereoDepth preset enum value.

        """
        preset_name = self._get_str('stereo_preset').upper()
        presets = dai.node.StereoDepth.PresetMode
        if not hasattr(presets, preset_name):
            self.get_logger().warning(
                f"Invalid stereo_preset '{preset_name}', falling back to FAST_DENSITY."
            )
            preset_name = 'FAST_DENSITY'
        return getattr(presets, preset_name)

    def _build_and_start_pipeline(self):
        """Create, configure, and start the OAK RGBD pipeline."""
        width, height = self.rgb_resolution
        self.pipeline = dai.Pipeline()

        rgb_socket = self._camera_socket('rgb_socket')
        left_socket = self._camera_socket('left_socket')
        right_socket = self._camera_socket('right_socket')

        color = self.pipeline.create(dai.node.Camera).build(rgb_socket, sensorFps=self.fps)
        left = self.pipeline.create(dai.node.Camera).build(left_socket, sensorFps=self.fps)
        right = self.pipeline.create(dai.node.Camera).build(right_socket, sensorFps=self.fps)
        stereo = self.pipeline.create(dai.node.StereoDepth)
        sync = self.pipeline.create(dai.node.Sync)

        stereo.setDefaultProfilePreset(self._stereo_preset())
        stereo.setDepthAlign(rgb_socket)
        stereo.setOutputSize(width, height)
        stereo.setLeftRightCheck(self._get_bool('stereo_left_right_check'))
        stereo.setSubpixel(self._get_bool('stereo_subpixel'))
        stereo.setExtendedDisparity(self._get_bool('stereo_extended_disparity'))
        stereo.setRectifyEdgeFillColor(self._get_int('stereo_rectify_edge_fill_color'))

        color.requestOutput(
            self.rgb_resolution,
            dai.ImgFrame.Type.BGR888i,
            fps=self.fps,
        ).link(sync.inputs['rgb'])
        left.requestOutput(self.rgb_resolution, fps=self.fps).link(stereo.left)
        right.requestOutput(self.rgb_resolution, fps=self.fps).link(stereo.right)
        stereo.depth.link(sync.inputs['depth'])

        sync.setRunOnHost(self._get_bool('sync_run_on_host'))
        sync.setSyncThreshold(
            datetime.timedelta(milliseconds=max(0.0, self._get_float('sync_threshold_ms')))
        )
        sync.setSyncAttempts(self._get_int('sync_attempts'))

        self.sync_queue = sync.out.createOutputQueue(maxSize=1, blocking=False)

        self.pipeline.start()
        calibration = self.pipeline.getDefaultDevice().readCalibration()
        intrinsics_matrix = calibration.getCameraIntrinsics(
            rgb_socket,
            self.rgb_resolution[0],
            self.rgb_resolution[1],
        )
        intrinsics = CameraIntrinsics.from_matrix(intrinsics_matrix)
        self.depth_projector = OakDepthProjector(
            intrinsics,
            self.depth_config,
            self.num_hands,
        )
        self.get_logger().info(
            'OAK calibration loaded: '
            f'fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}, '
            f'cx={intrinsics.cx:.2f}, cy={intrinsics.cy:.2f}'
        )

    def _capture_loop(self):
        """Read synchronized OAK RGBD frames until shutdown."""
        while self.running and rclpy.ok():
            try:
                message_group = self.sync_queue.get()
            except Exception as exc:
                if self.running:
                    self.get_logger().error(f'DepthAI queue error: {exc}')
                break

            if message_group is None:
                continue

            try:
                rgb_msg = message_group['rgb']
                depth_msg = message_group['depth']
                self._process_rgbd_frame(rgb_msg, depth_msg)
            except Exception as exc:
                self.get_logger().error(f'Error processing OAK RGBD frame: {exc}')

    def _process_rgbd_frame(self, rgb_msg, depth_msg):
        """
        Convert one OAK RGBD message group and run hand detection.

        Parameters
        ----------
        rgb_msg : depthai.ImgFrame
            DepthAI RGB frame message from the synchronized output group. It must
            provide ``getCvFrame`` and a device timestamp.
        depth_msg : depthai.ImgFrame
            DepthAI depth frame message aligned to the RGB camera. Its
            ``getFrame`` result is expected to contain depth in millimeters.

        """
        cv_bgr = rgb_msg.getCvFrame()
        if cv_bgr is None:
            return
        cv_bgr = np.ascontiguousarray(cv_bgr)
        if cv_bgr.ndim != 3 or cv_bgr.shape[2] != 3:
            self.get_logger().warning('Received non-RGB color frame; skipping.')
            return

        depth_mm = depth_msg.getFrame()
        if depth_mm is None:
            return
        if depth_mm.shape[:2] != cv_bgr.shape[:2]:
            depth_mm = cv2.resize(
                depth_mm,
                (cv_bgr.shape[1], cv_bgr.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = self.camera_frame_id
        ts_sec = timestamp_sec_from_header(header)
        candidate_ts_ms = self._device_timestamp_ms(rgb_msg)

        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_rgb)

        try:
            detection = self._runtime.detect(
                mp_image,
                candidate_ts_ms,
            )
        except Exception as exc:
            self.get_logger().error(
                f'Error in HandLandmarker.detect_for_video: {exc}'
            )
            return

        self._handle_result(
            result=detection.result,
            header=header,
            ts_sec=ts_sec,
            depth_mm=depth_mm,
            cv_rgb_for_visualization=cv_bgr if self.visualize else None,
            t_mediapipe=detection.duration_sec,
        )

    def _device_timestamp_ms(self, rgb_msg) -> int:
        """
        Return the candidate MediaPipe timestamp for an OAK frame.

        Parameters
        ----------
        rgb_msg : depthai.ImgFrame
            DepthAI RGB frame message with a device timestamp. If reading the
            device timestamp fails, wall-clock time is used as a fallback.

        Returns
        -------
        int
            Device timestamp in milliseconds, or wall-clock milliseconds when
            the device timestamp is unavailable. The shared runtime enforces
            strict monotonicity before submitting the frame.

        """
        try:
            timestamp = rgb_msg.getTimestamp()
            ts_ms = int(timestamp.total_seconds() * 1000.0)
        except Exception:
            ts_ms = int(time.time() * 1000.0)

        return ts_ms

    def reset_reference_callback(self, msg: Bool):
        """
        Request 3D reference recentering on a rising boolean signal.

        Parameters
        ----------
        msg : std_msgs.msg.Bool
            Reset command message. A false-to-true transition queues recentering
            on the next valid 3D hand, subject to
            ``reset_reference_cooldown_sec``.

        """
        now_sec = self.get_clock().now().nanoseconds * 1e-9
        reset_result = self.reference_state.request_reset(bool(msg.data), now_sec)
        if reset_result is ResetRequestResult.INACTIVE:
            return

        if reset_result is ResetRequestResult.COOLDOWN:
            self.get_logger().debug(
                f'Reset ignored due to cooldown ({self.reset_reference_cooldown_sec:.3f} s)'
            )
            return

        self.get_logger().info('Reference reset requested; waiting for next valid 3D hand')

    def _handle_result(
        self,
        result,
        header,
        ts_sec: float,
        depth_mm,
        cv_rgb_for_visualization,
        t_mediapipe,
    ):
        """
        Build, publish, and optionally visualize a detection result.

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.HandLandmarkerResult
            Detection result returned by MediaPipe for the current RGB frame.
        header : std_msgs.msg.Header
            Header used for all ``PointCloud`` messages published from this frame.
        ts_sec : float
            Frame timestamp in seconds. The One Euro filters use this value as
            their sample time when filtering is enabled.
        depth_mm : numpy.ndarray
            Depth image aligned to the RGB frame, with values in millimeters.
        cv_rgb_for_visualization : numpy.ndarray | None
            Optional OpenCV color frame used for overlay drawing. ``None`` skips
            visualization while still allowing publishing.
        t_mediapipe : float
            MediaPipe processing duration in seconds for display and debug output.

        """
        processed_metric_hands = []
        processed_image_hands = []
        missing_depth_count = 0

        if result.hand_landmarks:
            hands_to_process = (
                result.hand_landmarks
                if self.visualize
                else result.hand_landmarks[:1]
            )
            for hand_idx, hand_landmarks in enumerate(hands_to_process):
                metric_hand, image_hand, missing_count = self._build_3d_hand_landmarks(
                    hand_landmarks,
                    depth_mm,
                    ts_sec,
                    hand_idx,
                )
                missing_depth_count += missing_count
                if metric_hand is None:
                    continue
                processed_metric_hands.append(metric_hand)
                processed_image_hands.append(image_hand)

        if processed_metric_hands:
            self._update_reference_if_needed(
                processed_metric_hands[0],
                processed_image_hands[0],
            )
            reference_snapshot = self.reference_state.snapshot()
            reference = reference_snapshot.position
            reference_initialized = reference_snapshot.initialized

            if reference_initialized:
                relative_landmarks = relative_points(
                    processed_metric_hands[0],
                    reference,
                    (1.0, 1.0, 1.0),
                )

                cloud = PointCloud()
                cloud.header = header
                cloud.points = relative_landmarks
                self.landmarks_pub.publish(cloud)
        else:
            if self.landmark_filters is not None:
                self.landmark_filters.reset()
            self.get_logger().debug('No valid 3D hand detected in current frame.')

        if (
            self.visualize
            and cv_rgb_for_visualization is not None
        ):
            self._visualize(
                cv_rgb_for_visualization,
                processed_image_hands,
                processed_metric_hands[0] if processed_metric_hands else None,
                missing_depth_count,
                t_mediapipe,
            )

        if self.get_logger().is_enabled_for(rclpy.logging.LoggingSeverity.DEBUG):
            performance = self._debug_performance.tick()
            if performance is not None:
                self.get_logger().debug(
                    f'OAK 3D hand FPS = {performance.rate_hz:.2f}'
                )

    def _build_3d_hand_landmarks(self, hand_landmarks, depth_mm, ts_sec, hand_idx):
        """
        Back-project MediaPipe image landmarks into metric camera points.

        Parameters
        ----------
        hand_landmarks : Sequence[object]
            MediaPipe normalized landmarks for one hand. Each landmark must provide
            normalized ``x`` and ``y`` image coordinates.
        depth_mm : numpy.ndarray
            Depth image aligned to the RGB frame, with values in millimeters.
        ts_sec : float
            Frame timestamp in seconds used by smoothing filters.
        hand_idx : int
            Hand slot index used to select per-hand filters and last-known depth
            values for missing-depth recovery.

        Returns
        -------
        tuple[list[Point32] | None, list[Point32] | None, int]
            Metric camera-frame points in meters, normalized image-space points for
            visualization/reference display, and the number of landmarks without a
            direct depth sample. Point lists are ``None`` when the hand should be
            skipped.

        """
        if self.depth_projector is None:
            raise RuntimeError('OAK depth projector is not initialized')

        projection = self.depth_projector.project_hand(
            hand_landmarks,
            depth_mm,
            hand_idx,
        )
        if not projection.valid:
            return None, None, projection.missing_depth_count

        if self.landmark_filters is not None:
            def filter_metric_point(index, point):
                return self.landmark_filters.filter_point(
                    hand_idx,
                    index,
                    point,
                    ts_sec,
                )
            transform = filter_metric_point
        else:
            transform = None

        prepared_hand = prepare_metric_hand_for_output(
            projection.metric_points,
            self.depth_projector.intrinsics,
            self.rgb_resolution[0],
            self.rgb_resolution[1],
            transform,
        )
        if prepared_hand is None:
            return None, None, projection.missing_depth_count
        metric_hand, image_hand = prepared_hand

        return metric_hand, image_hand, projection.missing_depth_count

    def _update_reference_if_needed(self, metric_hand, image_hand):
        """
        Initialize or recenter the 3D reference from the tracked landmark.

        Parameters
        ----------
        metric_hand : Sequence[geometry_msgs.msg.Point32]
            Current hand landmarks in metric RGB camera coordinates, in meters.
        image_hand : Sequence[geometry_msgs.msg.Point32]
            Current hand landmarks in normalized image coordinates. These are saved
            only to display the reference marker at the matching image position.

        """
        if not (0 <= self.tracked_landmark_index < len(metric_hand)):
            return

        tracked_metric = metric_hand[self.tracked_landmark_index]
        tracked_image = image_hand[self.tracked_landmark_index]
        updated = self.reference_state.update_reference(
            (tracked_metric.x, tracked_metric.y, tracked_metric.z),
            image_position=(tracked_image.x, tracked_image.y, tracked_image.z),
            initialize_if_needed=self.auto_reference_on_first_detection,
        )
        if updated is None:
            return

        self.get_logger().info(
            '3D reference recentered from landmark '
            f'{self.tracked_landmark_index}: '
            f'({updated.position[0]:.3f}, '
            f'{updated.position[1]:.3f}, '
            f'{updated.position[2]:.3f}) m'
        )

    def _visualize(
        self,
        cv_rgb,
        image_hands,
        primary_metric_hand,
        missing_depth_count,
        t_mediapipe,
    ):
        """
        Render the OAK image overlay and process window-close keys.

        Parameters
        ----------
        cv_rgb : numpy.ndarray
            OpenCV color image to annotate. The caller passes the frame in display
            color order expected by OpenCV.
        image_hands : Sequence[Sequence[geometry_msgs.msg.Point32]]
            Detected hands in normalized image coordinates, used to draw landmark
            connections.
        primary_metric_hand : Sequence[geometry_msgs.msg.Point32] | None
            First valid hand in metric camera coordinates. ``None`` means no metric
            hand was available for reference feedback.
        missing_depth_count : int
            Number of landmarks in the current result that did not have direct
            valid depth samples.
        t_mediapipe : float
            MediaPipe processing duration in seconds.

        """
        reference_snapshot = self.reference_state.snapshot()

        exit_requested = self.viewer.show_3d(
            cv_rgb,
            image_hands,
            primary_metric_hand=primary_metric_hand,
            missing_depth_count=missing_depth_count,
            mediapipe_time_sec=t_mediapipe,
            reference_metric=reference_snapshot.position,
            reference_image=reference_snapshot.image_position,
            reference_initialized=reference_snapshot.initialized,
            tracked_landmark_index=self.tracked_landmark_index,
            control_overlay=self.control_overlay,
            camera_intrinsics=(
                self.depth_projector.intrinsics.fx,
                self.depth_projector.intrinsics.fy,
                self.depth_projector.intrinsics.cx,
                self.depth_projector.intrinsics.cy,
            ),
        )
        if exit_requested:
            self.get_logger().info('Visualization window closed by user.')
            rclpy.shutdown()

    def destroy_node(self):
        """Stop DepthAI, MediaPipe, and OpenCV resources before shutdown."""
        self.running = False
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
        except Exception:
            pass
        if self.capture_thread is not None and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        try:
            self._runtime.close()
        except Exception:
            pass
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    """
    Run the OAK 3D hand landmarks ROS node.

    Parameters
    ----------
    args : list[str] | None
        Optional ROS command-line arguments passed through to ``rclpy.init``.

    """
    rclpy.init(args=args)
    node = OakHandLandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
