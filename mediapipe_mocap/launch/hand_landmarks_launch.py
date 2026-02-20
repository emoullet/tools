from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    config_file = os.path.join(package_share_dir, 'config', 'hand_landmarks_node.yaml')

    use_depth = LaunchConfiguration('use_depth')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    depth_time_tolerance_ms = LaunchConfiguration('depth_time_tolerance_ms')
    depth_min_m = LaunchConfiguration('depth_min_m')
    depth_max_m = LaunchConfiguration('depth_max_m')
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_depth',
            default_value='false',
            description='Enable RGB+depth fusion for metric 3D landmarks'
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/aligned_depth_to_color/image_raw',
            description='Depth image topic aligned with RGB image_topic'
        ),
        DeclareLaunchArgument(
            'camera_info_topic',
            default_value='/camera/color/camera_info',
            description='CameraInfo topic used for depth projection'
        ),
        DeclareLaunchArgument(
            'depth_time_tolerance_ms',
            default_value='10.0',
            description='Maximum allowed RGB/depth timestamp mismatch (ms)'
        ),
        DeclareLaunchArgument(
            'depth_min_m',
            default_value='0.05',
            description='Minimum valid depth in meters'
        ),
        DeclareLaunchArgument(
            'depth_max_m',
            default_value='2.0',
            description='Maximum valid depth in meters'
        ),
        Node(
            package='mediapipe_mocap',
            executable='hand_landmarks_node',
            name='hand_landmarks_node',
            output='screen',
            parameters=[
                config_file,
                {
                    'use_depth': use_depth,
                    'depth_topic': depth_topic,
                    'camera_info_topic': camera_info_topic,
                    'depth_time_tolerance_ms': depth_time_tolerance_ms,
                    'depth_min_m': depth_min_m,
                    'depth_max_m': depth_max_m,
                }
            ]
        )
    ])
