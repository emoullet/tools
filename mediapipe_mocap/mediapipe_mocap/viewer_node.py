import os
import sys
import threading

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud


def _add_venv_site_packages_to_path():
    candidate_roots = []

    venv_root = os.environ.get('VIRTUAL_ENV')
    if venv_root:
        candidate_roots.append(venv_root)

    current_dir = os.getcwd()
    for _ in range(4):
        candidate = os.path.join(current_dir, '.venv_mediapipe')
        if os.path.isdir(candidate):
            candidate_roots.append(candidate)
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    for root in candidate_roots:
        site_packages = os.path.join(
            root,
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
        )
        if os.path.isdir(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)
            return


_add_venv_site_packages_to_path()

import mediapipe as mp  # noqa: E402,I100
from mediapipe.framework.formats import landmark_pb2  # noqa: E402,I100

# MediaPipe drawing utilities
solutions = mp.solutions


class HandLandmarksViewer(Node):
    """
    Display images with hand landmarks overlaid using OpenCV.

    Subscribe to:
      - /camera/color/image_raw (sensor_msgs/Image)
      - /hand_landmarks (sensor_msgs/PointCloud, 21 normalized points)
    """

    def __init__(self):
        super().__init__('hand_landmarks_viewer')

        # Parameters (topics can be changed if needed)
        self.declare_parameter('image_topic', '/camera/color/image_raw')
        self.declare_parameter('landmarks_topic', '/hand_landmarks')
        self.declare_parameter('window_name', 'Hand Landmarks Viewer')

        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value
        landmarks_topic = self.get_parameter('landmarks_topic').get_parameter_value().string_value
        self.window_name = self.get_parameter('window_name').get_parameter_value().string_value

        self.bridge = CvBridge()

        # Shared state
        self.latest_image = None          # OpenCV BGR image
        self.latest_landmarks = None      # list[Point32]
        self.lock = threading.Lock()

        # Subscribers
        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            10
        )

        self.landmarks_sub = self.create_subscription(
            PointCloud,
            landmarks_topic,
            self.landmarks_callback,
            10
        )

        # Timer to periodically display the image (e.g., 30 Hz)
        self.timer = self.create_timer(1.0 / 30.0, self.timer_callback)

        self.get_logger().info(
            f'HandLandmarksViewer started.\n'
            f'  image_topic      = {image_topic}\n'
            f'  landmarks_topic  = {landmarks_topic}\n'
            f'  window_name      = {self.window_name}'
        )

    # ---------------- Callbacks ----------------

    def image_callback(self, msg: Image):
        try:
            cv_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error converting image: {e}')
            return

        with self.lock:
            self.latest_image = cv_bgr

    def landmarks_callback(self, msg: PointCloud):
        with self.lock:
            # Copy points into a simple list
            self.latest_landmarks = list(msg.points)

    # ---------------- Display Timer ----------------

    def timer_callback(self):
        with self.lock:
            if self.latest_image is None:
                return
            img = self.latest_image.copy()
            landmarks = self.latest_landmarks

        # Draw landmarks with MediaPipe styling if available
        if landmarks is not None and len(landmarks) > 0:
            img = self.draw_landmarks_on_image(img, landmarks)

        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            # ESC or 'q' → stop the node cleanly
            self.get_logger().info('Viewer: exit requested by user.')
            rclpy.shutdown()

    def draw_landmarks_on_image(self, rgb_image, landmarks):
        """Draw hand landmarks with MediaPipe's visualization utilities."""
        if landmarks is None or len(landmarks) == 0:
            return rgb_image

        annotated_image = np.copy(rgb_image)
        height, width, _ = annotated_image.shape

        # Convert Point32 list to MediaPipe NormalizedLandmarkList
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=pt.x, y=pt.y, z=pt.z)
            for pt in landmarks
        ])

        # Draw landmarks and connections using MediaPipe styles
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style()
        )

        return annotated_image

    def destroy_node(self):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarksViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
