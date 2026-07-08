import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class WebcamPublisher(Node):
    """Publish webcam frames as ROS2 Image messages."""

    def __init__(self):
        super().__init__('webcam_publisher')

        # Declare parameters
        # Declare parameters (0 = use native camera value)
        self.declare_parameter('camera_id', 4)
        self.declare_parameter('fps', 0)
        self.declare_parameter('frame_width', 0)
        self.declare_parameter('frame_height', 0)
        self.declare_parameter('selfie_mode', False)

        camera_id = self.get_parameter('camera_id').get_parameter_value().integer_value
        fps_param = self.get_parameter('fps').get_parameter_value().integer_value
        frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value
        self.selfie_mode = self.get_parameter('selfie_mode').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.publisher = self.create_publisher(Image, '/camera/color/image_raw', 10)

        # Open webcam
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(f'Failed to open camera with device ID: {camera_id}')

        # Apply resolution/fps only if explicitly set (non-zero)
        if frame_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        if frame_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        if fps_param > 0:
            self.cap.set(cv2.CAP_PROP_FPS, fps_param)

        # Read actual values from camera
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = actual_fps if actual_fps > 0 else 30.0

        # Timer: publish frames at actual FPS rate
        self.timer = self.create_timer(1.0 / self.fps, self.publish_frame)
        self.frame_count = 0

        self.get_logger().info(
            f'Webcam publisher started. Capturing from camera {camera_id} '
            f'at {self.fps:.1f} FPS, resolution {actual_width}x{actual_height}, '
            f"selfie_mode={'on' if self.selfie_mode else 'off'}."
        )

    def publish_frame(self):
        """Capture and publish a frame from the webcam."""
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn('Failed to capture frame from webcam')
            return

        if self.selfie_mode:
            frame = cv2.flip(frame, 1)

        # Convert BGR to RGB for standard ROS image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create ROS2 Image message
        msg = self.bridge.cv2_to_imgmsg(frame_rgb, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_link'

        self.publisher.publish(msg)
        self.frame_count += 1

        if self.frame_count % max(1, int(self.fps) * 5) == 0:  # Log every 5 seconds
            self.get_logger().debug(f'Published {self.frame_count} frames')

    def destroy_node(self):
        """Clean up camera resources."""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = WebcamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
