import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Create the offline-video hand landmarks test launch description."""
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    hand_landmarks_config_file = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml',
    )

    offline_media_package_share_dir = get_package_share_directory('offline_media_publisher')
    video_config_file = os.path.join(
        offline_media_package_share_dir,
        'config',
        'video_publisher.yaml',
    )

    folder_path_arg = DeclareLaunchArgument(
        'folder_path',
        default_value='',
        description='Path to folder containing videos (required)',
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='50',
        description='Publishing rate in Hz (overrides native video FPS)',
    )

    hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='hand_landmarks_node',
        name='hand_landmarks_node',
        output='screen',
        parameters=[hand_landmarks_config_file],
    )

    viewer_node = Node(
        package='mediapipe_mocap',
        executable='viewer_node',
        name='hand_landmarks_viewer',
        output='screen',
        parameters=[
            {
                'image_topic': '/camera/color/image_raw',
                'landmarks_topic': '/hand_landmarks',
                'window_name': 'Hand Landmarks Viewer',
            },
        ],
    )

    offline_video_node = Node(
        package='offline_media_publisher',
        executable='video_publisher',
        name='video_publisher',
        output='screen',
        parameters=[
            video_config_file,
            {
                'folder_path': LaunchConfiguration('folder_path'),
                'fps': LaunchConfiguration('fps'),
            },
        ],
    )

    return LaunchDescription([
        folder_path_arg,
        fps_arg,
        hand_landmarks_node,
        viewer_node,
        offline_video_node,
    ])
