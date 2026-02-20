import rclpy
from rclpy.node import Node
from rclpy.logging import LoggingSeverity

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from sensor_msgs_py import point_cloud2
from cv_bridge import CvBridge

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


import time
import platform
import os
from ament_index_python.packages import get_package_share_directory


class HandLandmarksNode(Node):
    """
    Subscribe to an RGB image, run MediaPipe Tasks HandLandmarker,
    optionally fuse with depth, then publish landmarks as PointCloud2.

        Output semantics (for the FIRST detected hand):
            - 21 points in fixed order (MediaPipe index 0..20)
            - if use_depth=false:
                    point.x = normalized x in [0,1]
                    point.y = normalized y in [0,1]
                    point.z = 0.0
            - if use_depth=true and camera intrinsics are available:
                    point.x/point.y/point.z = metric XYZ (meters) in camera frame

    header.frame_id and header.stamp are copied from the input Image.
    """

    def __init__(self):
        super().__init__('hand_landmarks_node')

        # Get package share directory for default model path
        package_share_dir = get_package_share_directory('mediapipe_mocap')
        default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

        # Declare parameters
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('depth_topic', '/camera/aligned_depth_to_color/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('landmarks_topic', '/hand_landmarks')
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('num_hands', 1)
        self.declare_parameter('use_depth', False)
        self.declare_parameter('depth_time_tolerance_ms', 10.0)
        self.declare_parameter('depth_min_m', 0.05)
        self.declare_parameter('depth_max_m', 2.0)
        self.declare_parameter('min_hand_detection_confidence', 0.5)
        self.declare_parameter('min_hand_presence_confidence', 0.5)
        self.declare_parameter('min_tracking_confidence', 0.5)


        # Retrieve parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
        camera_info_topic = self.get_parameter('camera_info_topic').get_parameter_value().string_value
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        # Use default path if YAML provides empty string
        if not model_path:
            model_path = default_model_path
        num_hands = int(self.get_parameter('num_hands').get_parameter_value().integer_value)
        self.use_depth = bool(self.get_parameter('use_depth').get_parameter_value().bool_value)
        self.depth_time_tolerance_ms = float(
            self.get_parameter('depth_time_tolerance_ms').get_parameter_value().double_value
        )
        self.depth_min_m = float(
            self.get_parameter('depth_min_m').get_parameter_value().double_value
        )
        self.depth_max_m = float(
            self.get_parameter('depth_max_m').get_parameter_value().double_value
        )
        min_det_conf = float(
            self.get_parameter('min_hand_detection_confidence').get_parameter_value().double_value
        )
        min_presence_conf = float(
            self.get_parameter('min_hand_presence_confidence').get_parameter_value().double_value
        )
        min_track_conf = float(
            self.get_parameter('min_tracking_confidence').get_parameter_value().double_value
        )

        self.bridge = CvBridge()
        self.last_depth_image = None
        self.last_depth_stamp_ns = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.warned_missing_depth = False
        self.warned_missing_intrinsics = False

        # ---------------- Publisher / Subscriber ----------------
        self.landmarks_pub = self.create_publisher(PointCloud2, landmarks_topic, 50)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            20
        )

        if self.use_depth:
            self.depth_sub = self.create_subscription(
                Image,
                depth_topic,
                self.depth_callback,
                20
            )
            self.camera_info_sub = self.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.camera_info_callback,
                20
            )

        # Detect delegate: use GPU on native Linux, CPU on WSL or other systems
        delegate = self._get_best_delegate()
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(
                model_asset_path=model_path,
                delegate=delegate
            ),
            running_mode=RunningMode.VIDEO,   # stream of frames with timestamps
            num_hands=num_hands,
            min_hand_detection_confidence=min_det_conf,
            min_hand_presence_confidence=min_presence_conf,
            min_tracking_confidence=min_track_conf,
        )

        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.get_logger().info(
            f'HandLandmarksNode started.\n'
            f'  image_topic      = {image_topic}\n'
            f'  use_depth        = {self.use_depth}\n'
            f'  depth_topic      = {depth_topic if self.use_depth else "(disabled)"}\n'
            f'  camera_info_topic= {camera_info_topic if self.use_depth else "(disabled)"}\n'
            f'  landmarks_topic  = {landmarks_topic}\n'
            f'  model_path       = {model_path}'
        )

        self.last_time = time.time()
        self.frame_count = 0

    # -------------------------------------------------------------
    # Image callback: convert ROS image → MediaPipe Image → detect
    # -------------------------------------------------------------
    def image_callback(self, msg: Image):
        # Convert ROS Image to OpenCV BGR
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting RGB image: {e}')
            return

        # BGR → RGB (MediaPipe expects SRGB)
        cv_rgb = cv2.cvtColor(cv_bgr, cv2.COLOR_BGR2RGB)

        # Wrap as MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv_rgb)

        # Use ROS time as monotonically increasing timestamp (ms)
        ts_ms = (
            msg.header.stamp.sec * 1000
            + msg.header.stamp.nanosec // 1_000_000
        )

        try:
            result = self.landmarker.detect_for_video(mp_image, ts_ms)
        except Exception as e:
            self.get_logger().error(f'Error in HandLandmarker.detect_for_video: {e}')
            return

        if not result.hand_landmarks:
            # No hands → nothing to publish
            self.get_logger().debug('No hand detected in current frame.')
            return
        
        if self.use_depth:
            points = self._build_depth_fused_points(msg, cv_rgb.shape, result.hand_landmarks[0])
        else:
            points = [[float(lm.x), float(lm.y), 0.0] for lm in result.hand_landmarks[0]]

        cloud = point_cloud2.create_cloud_xyz32(msg.header, points)
        self.landmarks_pub.publish(cloud)

        self.get_logger().debug(f'Published {len(points)} landmarks.')

        # --- FPS MEASUREMENT (debug mode only) ---
        if self.get_logger().is_enabled_for(LoggingSeverity.DEBUG):
            self.frame_count += 1
            now = time.time()
            elapsed = now - self.last_time
            if elapsed >= 1.0:  # every 1 second
                fps = self.frame_count / elapsed
                self.get_logger().debug(f"Mediapipe FPS = {fps:.2f}")
                self.last_time = now
                self.frame_count = 0

    def depth_callback(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error converting depth image: {e}')
            return

        self.last_depth_image = depth
        self.last_depth_stamp_ns = self._stamp_to_ns(msg.header.stamp)

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def _build_depth_fused_points(self, rgb_msg: Image, rgb_shape, hand_landmarks):
        if self.last_depth_image is None or self.last_depth_stamp_ns is None:
            if not self.warned_missing_depth:
                self.get_logger().warning('Depth enabled but no depth image has been received yet.')
                self.warned_missing_depth = True
            return [[float(lm.x), float(lm.y), 0.0] for lm in hand_landmarks]

        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            if not self.warned_missing_intrinsics:
                self.get_logger().warning('Depth enabled but no camera intrinsics received yet.')
                self.warned_missing_intrinsics = True
            return [[float(lm.x), float(lm.y), 0.0] for lm in hand_landmarks]

        rgb_stamp_ns = self._stamp_to_ns(rgb_msg.header.stamp)
        dt_ms = abs(rgb_stamp_ns - self.last_depth_stamp_ns) / 1_000_000.0
        if dt_ms > self.depth_time_tolerance_ms:
            self.get_logger().debug(
                f'Depth/RGB timestamp mismatch ({dt_ms:.2f} ms > {self.depth_time_tolerance_ms:.2f} ms).'
            )
            return [[float(lm.x), float(lm.y), 0.0] for lm in hand_landmarks]

        h, w = rgb_shape[0], rgb_shape[1]
        depth_h, depth_w = self.last_depth_image.shape[:2]
        points = []
        for lm in hand_landmarks:
            u = int(np.clip(round(lm.x * (w - 1)), 0, w - 1))
            v = int(np.clip(round(lm.y * (h - 1)), 0, h - 1))
            ud = int(np.clip(round(u * depth_w / w), 0, depth_w - 1))
            vd = int(np.clip(round(v * depth_h / h), 0, depth_h - 1))

            z = self._extract_depth_m(self.last_depth_image, ud, vd)
            if not np.isfinite(z) or z <= 0.0:
                points.append([float(lm.x), float(lm.y), 0.0])
                continue

            x = (u - self.cx) * z / self.fx
            y = (v - self.cy) * z / self.fy
            points.append([float(x), float(y), float(z)])

        return points

    def _extract_depth_m(self, depth_image: np.ndarray, u: int, v: int) -> float:
        depth_value = float(depth_image[v, u])

        if depth_image.dtype == np.uint16:
            depth_m = depth_value * 0.001
        else:
            depth_m = depth_value

        if not np.isfinite(depth_m):
            return 0.0
        if depth_m < self.depth_min_m or depth_m > self.depth_max_m:
            return 0.0
        return depth_m

    def _stamp_to_ns(self, stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    

    def destroy_node(self):
        # Cleanly close MediaPipe resources
        try:
            self.landmarker.close()
        except Exception:
            pass
        super().destroy_node()

    def _is_wsl(self) -> bool:
        """Check if running on Windows Subsystem for Linux."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower() or 'wsl' in f.read().lower()
        except Exception:
            return False

    def _get_best_delegate(self) -> str:
        """Get the best delegate available on the current platform."""
        system = platform.system()
        delegate = mp.tasks.BaseOptions.Delegate.CPU
        delegate_name = 'CPU'
        if system == 'Linux':
            if not self._is_wsl():
                system = 'Linux (native)'
                delegate = mp.tasks.BaseOptions.Delegate.GPU
                delegate_name = 'GPU'
            else:
                system = 'WSL (Windows Subsystem for Linux)'
        elif system == 'Darwin':
            system = 'macOS'
        self.get_logger().info(f'Platform: {system}. Using {delegate_name} delegate.')
        return delegate


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarksNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
