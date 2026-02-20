from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    hand_landmarks_config_file = os.path.join(package_share_dir, 'config', 'hand_landmarks_node.yaml')
    
    offline_media_package_share_dir = get_package_share_directory('offline_media_publisher')
    video_config_file = os.path.join(offline_media_package_share_dir, 'config', 'video_publisher.yaml')
    
    folder_path_arg = DeclareLaunchArgument(
        'folder_path',
        default_value='',
        description='Path to folder containing videos (required)'
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='50',
        description='Publishing rate in Hz (overrides native video FPS)'
    )

    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value='false',
        description='Enable RGB+depth fusion for metric 3D landmarks'
    )

    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera/aligned_depth_to_color/image_raw',
        description='Depth image topic aligned with RGB image_topic'
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera/color/camera_info',
        description='CameraInfo topic used for depth projection'
    )

    depth_time_tolerance_ms_arg = DeclareLaunchArgument(
        'depth_time_tolerance_ms',
        default_value='10.0',
        description='Maximum allowed RGB/depth timestamp mismatch (ms)'
    )

    depth_min_m_arg = DeclareLaunchArgument(
        'depth_min_m',
        default_value='0.05',
        description='Minimum valid depth in meters'
    )

    depth_max_m_arg = DeclareLaunchArgument(
        'depth_max_m',
        default_value='2.0',
        description='Maximum valid depth in meters'
    )
    
    hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='hand_landmarks_node',
        name='hand_landmarks_node',
        output='screen',
        parameters=[
            hand_landmarks_config_file,
            {
                'use_depth': LaunchConfiguration('use_depth'),
                'depth_topic': LaunchConfiguration('depth_topic'),
                'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                'depth_time_tolerance_ms': LaunchConfiguration('depth_time_tolerance_ms'),
                'depth_min_m': LaunchConfiguration('depth_min_m'),
                'depth_max_m': LaunchConfiguration('depth_max_m'),
            }
        ]
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
            }
            ]
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
                }
            ]
        )

    
    return LaunchDescription([
        folder_path_arg,
        fps_arg,
        use_depth_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        depth_time_tolerance_ms_arg,
        depth_min_m_arg,
        depth_max_m_arg,
        hand_landmarks_node, 
        viewer_node, 
        offline_video_node
    ])
