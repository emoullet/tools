import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    config_file = os.path.join(package_share_dir, 'config', 'hand_landmarks_node.yaml')

    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='false',
        description='Show local OpenCV visualization window in hand_landmarks_node',
    )

    window_name_arg = DeclareLaunchArgument(
        'window_name',
        default_value='Hand Landmarks (Node)',
        description='OpenCV window title when visualize is enabled',
    )

    running_mode_arg = DeclareLaunchArgument(
        'running_mode',
        default_value='VIDEO',
        description='MediaPipe running mode: VIDEO or LIVE_STREAM',
    )

    return LaunchDescription([
        visualize_arg,
        window_name_arg,
        running_mode_arg,
        Node(
            package='mediapipe_mocap',
            executable='hand_landmarks_node',
            name='hand_landmarks_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'visualize': LaunchConfiguration('visualize'),
                    'window_name': LaunchConfiguration('window_name'),
                    'running_mode': LaunchConfiguration('running_mode'),
                },
            ]
        )
    ])
