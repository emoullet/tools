import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the webcam-to-hand-landmarks pipeline launch description."""
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    hand_landmarks_config_file = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml',
    )
    webcam_publisher_config_file = os.path.join(
        package_share_dir,
        'config',
        'webcam_publisher.yaml',
    )

    # Arguments
    camera_id_arg = DeclareLaunchArgument(
        'camera_id',
        default_value='0',
        description='Camera device ID (default: 0 for primary webcam)',
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='0',
        description='Publishing rate in Hz (0 = use native camera FPS)',
    )

    frame_width_arg = DeclareLaunchArgument(
        'frame_width',
        default_value='0',
        description='Webcam frame width in pixels (0 = use native camera width)',
    )

    frame_height_arg = DeclareLaunchArgument(
        'frame_height',
        default_value='0',
        description='Webcam frame height in pixels (0 = use native camera height)',
    )

    # Hand landmarks node
    hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='hand_landmarks_node',
        name='hand_landmarks_node',
        output='screen',
        parameters=[hand_landmarks_config_file],
    )

    # Webcam publisher node
    webcam_publisher_node = Node(
        package='mediapipe_mocap',
        executable='webcam_publisher',
        name='webcam_publisher',
        output='screen',
        parameters=[
            webcam_publisher_config_file,
            {
                'camera_id': LaunchConfiguration('camera_id'),
                'fps': LaunchConfiguration('fps'),
                'frame_width': LaunchConfiguration('frame_width'),
                'frame_height': LaunchConfiguration('frame_height'),
            },
        ],
    )

    return LaunchDescription([
        camera_id_arg,
        fps_arg,
        frame_width_arg,
        frame_height_arg,
        webcam_publisher_node,
        hand_landmarks_node,
    ])
