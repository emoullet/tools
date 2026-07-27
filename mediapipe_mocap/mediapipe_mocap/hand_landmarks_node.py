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

import os
import threading
import time

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
from geometry_msgs.msg import Point32

from mediapipe_mocap.hand_landmarks_common import (
    ensure_3_tuple,
    get_best_mediapipe_delegate,
    OneEuroFilter,
    prepare_runtime_imports,
    relative_points,
    reset_filter_bank,
    timestamp_sec_from_header,
)
from mediapipe_mocap.reference import (
    ReferenceState,
    ResetRequestResult,
    ResetTriggerMode,
)
from mediapipe_mocap.ros_qos import latest_reliable_qos
from mediapipe_mocap.viewer import HandLandmarksViewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud
from std_msgs.msg import Bool


prepare_runtime_imports()

import cv2  # noqa: E402,I100
import mediapipe as mp  # noqa: E402,I100
from mediapipe.tasks.python import BaseOptions  # noqa: E402,I100
from mediapipe.tasks.python.vision import (  # noqa: E402,I100
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
import numpy as np  # noqa: E402,I100


class HandLandmarksNode(Node):
    """
    Subscribe to RGB images and publish 2D hand-control inputs.

    Output semantics for the first detected hand:
      - 21 points in fixed order (MediaPipe index 0..20)
      - point.x/y are reference-relative offsets scaled by frame dimensions
      - point.z is 0.0; the 2D node does not publish MediaPipe's relative z

    header.frame_id and header.stamp are copied from the input Image.
    """

    def __init__(self):
        """Initialize parameters, MediaPipe landmarker, and ROS interfaces."""
        super().__init__('hand_landmarks_node')

        # Get package share directory for default model path
        package_share_dir = get_package_share_directory('mediapipe_mocap')
        default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

        # Log active virtual environment for pip-installed MediaPipe debugging.
        self.get_logger().info(
            f"Using venv: {os.environ.get('VIRTUAL_ENV', 'Not in a virtual environment')}"
        )

        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_topic', '/camera/color/image_raw'),
                ('landmarks_topic', '/hand_landmarks'),
                ('model_path', default_model_path),
                ('num_hands', 1),
                ('min_hand_detection_confidence', 0.5),
                ('min_hand_presence_confidence', 0.5),
                ('min_tracking_confidence', 0.5),
                ('running_mode', 'VIDEO'),
                ('selfie_mode', False),
                ('enable_one_euro_filter', False),
                ('one_euro_frequency', 30.0),
                ('one_euro_mincutoff', 1.0),
                ('one_euro_beta', 0.1),
                ('visualize', True),
                ('window_name', 'Hand Landmarks (Node)'),
                ('reset_reference_topic', '/reset_reference'),
                ('reset_reference_cooldown_sec', 0.25),
                ('initial_reference', [0.5, 0.5, 0.5]),
                ('show_control_zones', True),
                ('dead_zone', 0.05),
                ('saturation_zone', 0.3),
                ('tracked_landmark_index', 0),
            ]
        )

        # Retrieve parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        # Use default path if YAML provides empty string
        if not model_path:
            model_path = default_model_path
        num_hands = int(self.get_parameter('num_hands').get_parameter_value().integer_value)
        min_det_conf = float(
            self.get_parameter('min_hand_detection_confidence').get_parameter_value().double_value
        )
        min_presence_conf = float(
            self.get_parameter('min_hand_presence_confidence').get_parameter_value().double_value
        )
        min_track_conf = float(
            self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        )
        running_mode_param = (
            self.get_parameter('running_mode').get_parameter_value().string_value.upper()
        )
        if running_mode_param not in ('VIDEO', 'LIVE_STREAM'):
            self.get_logger().warning(
                f"Invalid running_mode '{running_mode_param}', falling back to VIDEO. "
                "Expected 'VIDEO' or 'LIVE_STREAM'."
            )
            running_mode_param = 'VIDEO'
        self.running_mode = (
            RunningMode.LIVE_STREAM
            if running_mode_param == 'LIVE_STREAM'
            else RunningMode.VIDEO
        )
        self.selfie_mode = (
            self.get_parameter('selfie_mode').get_parameter_value().bool_value
        )
        self.enable_one_euro_filter = (
            self.get_parameter('enable_one_euro_filter').get_parameter_value().bool_value
        )
        one_euro_frequency = float(
            self.get_parameter('one_euro_frequency').get_parameter_value().double_value
        )
        one_euro_mincutoff = float(
            self.get_parameter('one_euro_mincutoff').get_parameter_value().double_value
        )
        one_euro_beta = float(
            self.get_parameter('one_euro_beta').get_parameter_value().double_value
        )
        self.visualize = self.get_parameter('visualize').get_parameter_value().bool_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value
        self.viewer = (
            HandLandmarksViewer(self.window_name)
            if self.visualize
            else None
        )
        self.reset_reference_topic = (
            self.get_parameter('reset_reference_topic').get_parameter_value().string_value
        )
        self.reset_reference_cooldown_sec = max(
            0.0,
            float(
                self.get_parameter('reset_reference_cooldown_sec')
                .get_parameter_value()
                .double_value
            ),
        )
        initial_reference_values = (
            self.get_parameter('initial_reference').get_parameter_value().double_array_value
        )
        initial_reference = ensure_3_tuple(
            initial_reference_values,
            [0.5, 0.5, 0.5],
            logger=self.get_logger(),
            parameter_name='initial_reference',
        )
        self.reference_state = ReferenceState(
            initial_position=initial_reference,
            cooldown_sec=self.reset_reference_cooldown_sec,
            trigger_mode=ResetTriggerMode.RISING_EDGE,
        )
        self.show_control_zones = (
            self.get_parameter('show_control_zones').get_parameter_value().bool_value
        )
        self.dead_zone = max(
            0.0,
            float(self.get_parameter('dead_zone').get_parameter_value().double_value),
        )
        self.saturation_zone = max(
            1e-6,
            float(self.get_parameter('saturation_zone').get_parameter_value().double_value),
        )
        self.tracked_landmark_index = int(
            self.get_parameter('tracked_landmark_index').get_parameter_value().integer_value
        )

        # Keep filter parameters in valid ranges to avoid unstable behavior.
        one_euro_frequency = max(one_euro_frequency, 1e-3)
        one_euro_mincutoff = max(one_euro_mincutoff, 1e-6)
        one_euro_beta = max(one_euro_beta, 0.0)

        self.one_euro_filters = []
        if self.enable_one_euro_filter:
            filter_hand_slots = max(num_hands, 1)
            self.one_euro_filters = [
                [
                    OneEuroFilter(one_euro_frequency, one_euro_mincutoff, one_euro_beta)
                    for _ in range(21 * 3)
                ]
                for _ in range(filter_hand_slots)
            ]

        self.bridge = CvBridge()

        # ---------------- Publisher / Subscriber ----------------
        self.landmarks_pub = self.create_publisher(PointCloud, landmarks_topic, 10)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            latest_reliable_qos(),
        )
        self.reset_reference_sub = self.create_subscription(
            Bool,
            self.reset_reference_topic,
            self.reset_reference_callback,
            10,
        )

        self.frame_size = None  # (width, height) of the first image received
        # (width / min(width, height), height / min(width, height)).
        self.frame_normalization_factor = None

        # Detect delegate: use GPU on native Linux, CPU on WSL or other systems
        delegate = get_best_mediapipe_delegate(mp, self.get_logger())

        self._ts_lock = threading.Lock()
        self._last_ts_ms = -1
        self._header_by_ts_ms = {}
        self._detect_start_by_ts_ms = {}
        self._max_pending_timestamps = 120
        options_kwargs = {
            'base_options': BaseOptions(
                model_asset_path=model_path,
                delegate=delegate
            ),
            'running_mode': self.running_mode,
            'num_hands': num_hands,
            'min_hand_detection_confidence': min_det_conf,
            'min_hand_presence_confidence': min_presence_conf,
            'min_tracking_confidence': min_track_conf,
        }
        if self.running_mode == RunningMode.LIVE_STREAM:
            options_kwargs['result_callback'] = self._on_live_stream_result

        options = HandLandmarkerOptions(**options_kwargs)

        self.landmarker = HandLandmarker.create_from_options(options)

        initial_reference = self.reference_state.snapshot().position
        self.get_logger().info(
            f'HandLandmarksNode started.\n'
            f'  image_topic      = {image_topic}\n'
            f'  landmarks_topic  = {landmarks_topic}\n'
            f'  model_path       = {model_path}\n'
            f'  running_mode     = {running_mode_param}\n'
            f'  selfie_mode      = {self.selfie_mode}\n'
            f'  prime_offload    = '
            f"{os.environ.get('__NV_PRIME_RENDER_OFFLOAD', '<unset>')} "
            f"glx_vendor={os.environ.get('__GLX_VENDOR_LIBRARY_NAME', '<unset>')}\n"
            f'  reset_reference  = {self.reset_reference_topic} '
            f'(cooldown={self.reset_reference_cooldown_sec:.3f}s)\n'
            f'  initial_ref      = ({initial_reference[0]:.3f}, '
            f'{initial_reference[1]:.3f}, {initial_reference[2]:.3f})\n'
            f'  control_zones    = {self.show_control_zones} '
            f'(dead={self.dead_zone:.3f}, sat={self.saturation_zone:.3f}, '
            f'lm_idx={self.tracked_landmark_index})\n'
            f'  one_euro_filter  = {self.enable_one_euro_filter}\n'
            f'  visualize        = {self.visualize}'
        )

        if self.enable_one_euro_filter:
            self.get_logger().info(
                'One Euro filter enabled with '
                f'frequency={one_euro_frequency:.3f}, '
                f'mincutoff={one_euro_mincutoff:.3f}, '
                f'beta={one_euro_beta:.3f}'
            )

        self.last_debug_time = time.time()
        self.frame_count = 0

    # -------------------------------------------------------------
    # Image callback: convert ROS image → MediaPipe Image → detect
    # -------------------------------------------------------------
    def image_callback(self, msg: Image):
        """
        Convert an RGB image message and submit it to MediaPipe.

        Parameters
        ----------
        msg : sensor_msgs.msg.Image
            RGB image message from the configured camera topic. Its header stamp is
            converted to a MediaPipe timestamp and its full header is preserved on
            the published landmark cloud.

        """
        # Convert directly to RGB for MediaPipe. Only convert to BGR later when
        # the optional OpenCV visualization needs it.
        try:
            cv_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
            return

        if self.selfie_mode:
            cv_rgb = cv2.flip(cv_rgb, 1)

        if self.frame_size is None:
            self.frame_size = (cv_rgb.shape[1], cv_rgb.shape[0])
            self.frame_normalization_factor = (
                self.frame_size[0] / min(self.frame_size),
                self.frame_size[1] / min(self.frame_size),
            )
            self.get_logger().info(f'First image received: size={self.frame_size}')
            self.get_logger().info(f'Normalization factor: {self.frame_normalization_factor}')

        # Wrap as MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_rgb)

        # Use ROS time and enforce strictly increasing timestamps for MediaPipe.
        ts_ms = self._next_timestamp_ms(msg)
        ts_sec = timestamp_sec_from_header(msg.header)
        if self.running_mode == RunningMode.LIVE_STREAM:
            with self._ts_lock:
                self._header_by_ts_ms[ts_ms] = msg.header
                self._detect_start_by_ts_ms[ts_ms] = time.time()
                self._trim_pending_timestamp_maps()
            try:
                self.landmarker.detect_async(mp_image, ts_ms)
            except Exception as e:
                with self._ts_lock:
                    self._header_by_ts_ms.pop(ts_ms, None)
                    self._detect_start_by_ts_ms.pop(ts_ms, None)
                self.get_logger().error(f'Error in HandLandmarker.detect_async: {e}')
            return

        now = time.time()
        try:
            result = self.landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:
            self.get_logger().error(f'Error in HandLandmarker.detect_for_video: {e}')
            return
        t_mediapipe = time.time() - now

        self._handle_result(
            result=result,
            header=msg.header,
            ts_sec=ts_sec,
            cv_bgr_for_visualization=(
                cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR) if self.visualize else None
            ),
            t_mediapipe=t_mediapipe,
        )

    def _on_live_stream_result(self, result, output_image, timestamp_ms: int):
        """
        Handle asynchronous MediaPipe results and restore ROS context.

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.HandLandmarkerResult
            MediaPipe detection result for the image associated with
            ``timestamp_ms``.
        output_image : mediapipe.Image | None
            Optional image returned by MediaPipe in live-stream mode. When
            visualization is enabled, it is copied and converted for OpenCV
            drawing.
        timestamp_ms : int
            MediaPipe timestamp key used to recover the stored ROS header and
            detection start time.

        """
        with self._ts_lock:
            header = self._header_by_ts_ms.pop(timestamp_ms, None)
            detect_start = self._detect_start_by_ts_ms.pop(timestamp_ms, None)

        if header is None:
            return

        ts_sec = timestamp_sec_from_header(header)
        t_mediapipe = None
        if detect_start is not None:
            t_mediapipe = max(time.time() - detect_start, 0.0)

        cv_bgr = None
        if self.visualize and output_image is not None:
            try:
                cv_rgb = np.array(output_image.numpy_view(), copy=True)
                cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                cv_bgr = None

        self._handle_result(
            result=result,
            header=header,
            ts_sec=ts_sec,
            cv_bgr_for_visualization=cv_bgr,
            t_mediapipe=t_mediapipe,
        )

    def _next_timestamp_ms(self, msg: Image) -> int:
        """
        Return a strictly increasing MediaPipe timestamp in milliseconds.

        Parameters
        ----------
        msg : sensor_msgs.msg.Image
            Input image whose header stamp provides the source time. If that stamp
            does not increase monotonically, the returned timestamp is advanced by
            one millisecond.

        Returns
        -------
        int
            Strictly increasing timestamp accepted by MediaPipe's video and
            live-stream APIs.

        """
        ts_ms = int(msg.header.stamp.sec) * 1000 + int(msg.header.stamp.nanosec) // 1_000_000
        with self._ts_lock:
            if ts_ms <= self._last_ts_ms:
                ts_ms = self._last_ts_ms + 1
            self._last_ts_ms = ts_ms
        return ts_ms

    def reset_reference_callback(self, msg: Bool):
        """
        Request reference recentering on a rising boolean signal.

        Parameters
        ----------
        msg : std_msgs.msg.Bool
            Reset command message. A false-to-true transition queues recentering on
            the next detected hand, subject to ``reset_reference_cooldown_sec``.

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

        self.get_logger().info(
            'Reference reset requested; waiting for next landmark frame to recenter'
        )

    def _trim_pending_timestamp_maps(self):
        """Drop old async MediaPipe timestamp contexts beyond the limit."""
        while len(self._header_by_ts_ms) > self._max_pending_timestamps:
            # Dict preserves insertion order; pop the oldest pending timestamp first.
            oldest_key = next(iter(self._header_by_ts_ms))
            self._header_by_ts_ms.pop(oldest_key, None)
            self._detect_start_by_ts_ms.pop(oldest_key, None)

    def _update_reference_if_needed(self, processed_hand_landmarks):
        """
        Recenter the reference on the tracked landmark when requested.

        Parameters
        ----------
        processed_hand_landmarks : list[list[geometry_msgs.msg.Point32]]
            Detected hands after optional filtering. Coordinates are normalized
            image-space x/y values with planar ``z = 0``.

        """
        first_hand = processed_hand_landmarks[0]
        if not (0 <= self.tracked_landmark_index < len(first_hand)):
            return

        tracked_lm = first_hand[self.tracked_landmark_index]
        updated = self.reference_state.update_reference(
            (tracked_lm.x, tracked_lm.y, tracked_lm.z)
        )
        if updated is None:
            return

        self.get_logger().info(
            'Reference position recentered from current landmark '
            f'{self.tracked_landmark_index}: '
            f'({updated.position[0]:.3f}, '
            f'{updated.position[1]:.3f}, '
            f'{updated.position[2]:.3f})'
        )

    def _handle_result(self, result, header, ts_sec: float, cv_bgr_for_visualization, t_mediapipe):
        """
        Publish processed hand landmarks and update optional visualization.

        Parameters
        ----------
        result : mediapipe.tasks.python.vision.HandLandmarkerResult
            Detection result returned by MediaPipe for the current image.
        header : std_msgs.msg.Header
            Header copied to the published ``PointCloud`` so consumers can trace
            landmarks back to the source image.
        ts_sec : float
            Source timestamp in seconds. The One Euro filters use this value as
            their sample time when filtering is enabled.
        cv_bgr_for_visualization : numpy.ndarray | None
            Optional BGR image used for the OpenCV overlay. ``None`` skips
            visualization even when the node is otherwise running normally.
        t_mediapipe : float | None
            MediaPipe processing duration in seconds. ``None`` means no timing was
            available for this result path.

        """
        processed_hand_landmarks = []
        if result.hand_landmarks:
            hands_to_process = (
                result.hand_landmarks
                if self.visualize
                else result.hand_landmarks[:1]
            )
            for hand_idx, hand_landmarks in enumerate(hands_to_process):
                processed_hand = []
                for i, lm in enumerate(hand_landmarks):
                    x = float(lm.x)
                    y = float(lm.y)
                    # Keep the 2D node planar; the OAK path handles metric depth.
                    z = 0.0
                    if self.enable_one_euro_filter and hand_idx < len(self.one_euro_filters):
                        hand_filters = self.one_euro_filters[hand_idx]
                        base_idx = i * 3
                        x = hand_filters[base_idx].filter(x, ts_sec)
                        y = hand_filters[base_idx + 1].filter(y, ts_sec)
                        z = hand_filters[base_idx + 2].filter(z, ts_sec)
                    processed_hand.append(Point32(x=x, y=y, z=z))
                processed_hand_landmarks.append(processed_hand)

        if processed_hand_landmarks:
            self._update_reference_if_needed(processed_hand_landmarks)

            reference = self.reference_state.snapshot().position
            norm_x, norm_y = self.frame_normalization_factor
            ref_x, ref_y = reference[0], reference[1]

            relative_landmarks = relative_points(
                processed_hand_landmarks[0],
                (ref_x, ref_y, reference[2]),
                (norm_x, norm_y, 0.0),
            )

            cloud = PointCloud()
            cloud.header = header
            cloud.points = relative_landmarks
            self.landmarks_pub.publish(cloud)
        else:
            if self.enable_one_euro_filter:
                reset_filter_bank(self.one_euro_filters)
            self.get_logger().debug('No hand detected in current frame.')

        if self.visualize:
            if cv_bgr_for_visualization is None:
                return
            reference = self.reference_state.snapshot().position
            exit_requested = self.viewer.show_2d(
                cv_bgr_for_visualization,
                processed_hand_landmarks,
                mediapipe_time_sec=t_mediapipe,
                reference_xyz=reference,
                tracked_landmark_index=self.tracked_landmark_index,
                show_control_zones=self.show_control_zones,
                dead_zone=self.dead_zone,
                saturation_zone=self.saturation_zone,
            )
            if exit_requested:
                self.get_logger().info('Visualization window closed by user.')
                rclpy.shutdown()
        # self.get_logger().debug(f'Published {len(cloud.points)} landmarks.')

        # --- FPS MEASUREMENT (debug mode only) ---
        if self.get_logger().is_enabled_for(rclpy.logging.LoggingSeverity.DEBUG):
            self.frame_count += 1
            now = time.time()
            elapsed = now - self.last_debug_time
            if elapsed >= 1.0:  # every 1 second
                fps = self.frame_count / elapsed
                self.get_logger().debug(f'Mediapipe FPS = {fps:.2f}')
                self.last_debug_time = now
                self.frame_count = 0

    def destroy_node(self):
        """Close MediaPipe and OpenCV resources before node shutdown."""
        # Cleanly close MediaPipe resources
        try:
            self.landmarker.close()
        except Exception:
            pass
        if self.viewer is not None:
            self.viewer.close()
        super().destroy_node()


def main(args=None):
    """
    Run the 2D hand landmarks ROS node.

    Parameters
    ----------
    args : list[str] | None
        Optional ROS command-line arguments passed through to ``rclpy.init``.

    """
    rclpy.init(args=args)
    node = HandLandmarksNode()
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
