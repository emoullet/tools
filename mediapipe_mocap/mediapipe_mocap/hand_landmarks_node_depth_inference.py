import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
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


class HandLandmarksDepthInferenceNode(Node):
    """
    Subscribe to an RGB image, run MediaPipe Tasks HandLandmarker,
    compute hand depth relative to a reference position.

    Uses:
      - 2D hand landmarks (normalized coordinates)
      - 3D world landmarks (metric coordinates in hand-centric frame)
      - Camera intrinsics (approximated from image shape)
      - Reference position (registered via service call)

    Publishes:
      - Hand landmarks as PointCloud (2D normalized + inferred depth)
      - Depth value relative to reference position
    """

    def __init__(self):
        super().__init__('hand_landmarks_depth_inference_node')

        # Get package share directory for default model path
        package_share_dir = get_package_share_directory('mediapipe_mocap')
        default_model_path = os.path.join(package_share_dir, 'models', 'hand_landmarker.task')

        # Declare parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('image_topic', '/camera/color/image_raw'),
                ('landmarks_topic', '/hand_landmarks_depth'),
                ('depth_topic', '/hand_depth'),
                ('model_path', default_model_path),
                ('num_hands', 1),
                ('min_hand_detection_confidence', 0.5),
                ('min_hand_presence_confidence', 0.5),
                ('min_tracking_confidence', 0.5),
                ('focal_length_factor', 1.0),  # Adjust if needed for camera calibration
            ]
        )


        # Retrieve parameters
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter('depth_topic').get_parameter_value().string_value
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
        self.focal_length_factor = float(
            self.get_parameter('focal_length_factor').get_parameter_value().double_value
        )

        self.bridge = CvBridge()

        # Camera intrinsics (will be estimated from first image)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.img_width = None
        self.img_height = None

        # Reference position (3D world landmark position to be registered)
        self.reference_position = None  # Will store wrist position as reference
        self.reference_depth = None     # Estimated depth at reference

        # ---------------- Publisher / Subscriber ----------------
        self.landmarks_pub = self.create_publisher(PointCloud2, landmarks_topic, 50)
        self.depth_pub = self.create_publisher(Float32, depth_topic, 50)

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            20
        )

        # Service to register reference position
        self.register_ref_service = self.create_service(
            Trigger,
            '~/register_reference',
            self.register_reference_callback
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
        
        # Store latest detection for reference registration
        self.latest_world_landmarks = None
        self.latest_landmarks_2d = None
        self.latest_depth_estimate = None
        
        self.get_logger().info(
            f'HandLandmarksDepthInferenceNode started.\n'
            f'  image_topic      = {image_topic}\n'
            f'  landmarks_topic  = {landmarks_topic}\n'
            f'  depth_topic      = {depth_topic}\n'
            f'  model_path       = {model_path}\n'
            f'Call "ros2 service call ~/register_reference std_srvs/srv/Trigger" to set reference position.'
        )

        self.last_time = time.time()
        self.frame_count = 0

    # -------------------------------------------------------------
    # Camera intrinsics estimation from image shape
    # -------------------------------------------------------------
    def _estimate_camera_intrinsics(self, width: int, height: int):
        """
        Estimate camera intrinsics from image dimensions.
        Assumes a typical webcam with ~60 degree horizontal FOV.
        """
        if self.fx is not None:
            return  # Already initialized
        
        self.img_width = width
        self.img_height = height
        
        # Approximate focal length based on typical FOV
        # For FOV ~60 degrees: f ≈ width / (2 * tan(FOV/2))
        # FOV 60 deg → f ≈ width * 0.866
        self.fx = width * 0.866 * self.focal_length_factor
        self.fy = width * 0.866 * self.focal_length_factor  # Assume square pixels
        
        # Principal point at image center
        self.cx = width / 2.0
        self.cy = height / 2.0
        
        self.get_logger().info(
            f'Camera intrinsics estimated:\n'
            f'  fx = {self.fx:.2f}, fy = {self.fy:.2f}\n'
            f'  cx = {self.cx:.2f}, cy = {self.cy:.2f}\n'
            f'  image: {width}x{height}'
        )

    # -------------------------------------------------------------
    # Depth estimation from 2D and 3D landmarks
    # -------------------------------------------------------------
    def _estimate_depth_from_landmarks(self, landmarks_2d, world_landmarks_3d):
        """
        Estimate depth using perspective projection.
        
        Args:
            landmarks_2d: List of 2D normalized landmarks (x, y in [0,1])
            world_landmarks_3d: List of 3D world landmarks in meters
        
        Returns:
            Estimated depth (z-coordinate) in camera frame
        """
        # Use wrist landmark (index 0) as reference point
        wrist_2d = landmarks_2d[0]
        wrist_3d = world_landmarks_3d[0]
        
        # Convert normalized coordinates to pixel coordinates
        u = wrist_2d.x * self.img_width
        v = wrist_2d.y * self.img_height
        
        # 3D world landmark gives us the hand size in metric units
        # We can use the scale of the hand to infer depth
        # Compute hand span (distance from wrist to middle finger tip)
        middle_finger_tip_3d = world_landmarks_3d[12]  # Middle finger tip
        hand_size_3d = np.sqrt(
            (wrist_3d.x - middle_finger_tip_3d.x)**2 +
            (wrist_3d.y - middle_finger_tip_3d.y)**2 +
            (wrist_3d.z - middle_finger_tip_3d.z)**2
        )
        
        # Compute same distance in 2D image space
        middle_finger_tip_2d = landmarks_2d[12]
        hand_size_2d_px = np.sqrt(
            ((wrist_2d.x - middle_finger_tip_2d.x) * self.img_width)**2 +
            ((wrist_2d.y - middle_finger_tip_2d.y) * self.img_height)**2
        )
        
        # Depth from perspective projection: Z = (f * real_size) / pixel_size
        if hand_size_2d_px > 1e-6:  # Avoid division by zero
            depth = (self.fx * hand_size_3d) / hand_size_2d_px
        else:
            depth = None
            
        return depth

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

        # Initialize camera intrinsics from first image
        height, width = cv_bgr.shape[:2]
        self._estimate_camera_intrinsics(width, height)

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

        if not result.hand_landmarks or not result.hand_world_landmarks:
            # No hands → nothing to publish
            self.get_logger().debug('No hand detected in current frame.')
            self.latest_world_landmarks = None
            self.latest_landmarks_2d = None
            self.latest_depth_estimate = None
            return
        
        # Extract first hand's landmarks
        landmarks_2d = result.hand_landmarks[0]
        world_landmarks_3d = result.hand_world_landmarks[0]
        
        # Store for reference registration
        self.latest_landmarks_2d = landmarks_2d
        self.latest_world_landmarks = world_landmarks_3d
        
        # Estimate depth
        depth = self._estimate_depth_from_landmarks(landmarks_2d, world_landmarks_3d)
        
        if depth is None:
            self.get_logger().debug('Could not estimate depth.')
            return
        
        self.latest_depth_estimate = depth
        
        # Compute relative depth if reference is registered
        relative_depth = None
        if self.reference_depth is not None:
            relative_depth = depth - self.reference_depth
            
            # Publish relative depth
            depth_msg = Float32()
            depth_msg.data = float(relative_depth)
            self.depth_pub.publish(depth_msg)
            
            self.get_logger().debug(
                f'Depth: {depth:.3f}m, Reference: {self.reference_depth:.3f}m, '
                f'Relative: {relative_depth:.3f}m'
            )
        else:
            self.get_logger().debug(f'Depth: {depth:.3f}m (no reference set)')
        
        # Publish landmarks with inferred depth
        # Create list of points (x=normalized_x, y=normalized_y, z=depth)
        points = []
        for lm_2d in landmarks_2d:
            points.append([float(lm_2d.x), float(lm_2d.y), float(depth)])
        
        # Create PointCloud2 message using sensor_msgs_py helper
        msg.header.frame_id = "hand_landmarks"
        cloud = point_cloud2.create_cloud_xyz32(msg.header, points)
        
        self.landmarks_pub.publish(cloud)
        self.get_logger().debug(f'Published {len(points)} landmarks with depth.')

        # --- FPS MEASUREMENT (debug mode only) ---
        if self.get_logger().is_enabled_for(rclpy.logging.LoggingSeverity.DEBUG):
            self.frame_count += 1
            now = time.time()
            elapsed = now - self.last_time
            if elapsed >= 1.0:  # every 1 second
                fps = self.frame_count / elapsed
                self.get_logger().debug(f"Mediapipe FPS = {fps:.2f}")
                self.last_time = now
                self.frame_count = 0

    # -------------------------------------------------------------
    # Service callback: register current position as reference
    # -------------------------------------------------------------
    def register_reference_callback(self, request, response):
        """
        Register the current hand position as the reference position.
        """
        if self.latest_depth_estimate is None or self.latest_world_landmarks is None:
            response.success = False
            response.message = 'No hand detected. Cannot register reference position.'
            self.get_logger().warn('Reference registration failed: No hand detected.')
            return response
        
        # Store reference position (wrist in world coordinates)
        self.reference_position = self.latest_world_landmarks[0]  # Wrist
        self.reference_depth = self.latest_depth_estimate
        
        response.success = True
        response.message = (
            f'Reference position registered at depth {self.reference_depth:.3f}m'
        )
        self.get_logger().info(
            f'Reference registered: depth = {self.reference_depth:.3f}m'
        )
        
        return response

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
        return 'CPU'


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarksDepthInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
