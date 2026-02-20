from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    hand_landmarks_config_file = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml'
    )

    image_topic = LaunchConfiguration('image_topic')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    depth_time_tolerance_ms = LaunchConfiguration('depth_time_tolerance_ms')
    depth_min_m = LaunchConfiguration('depth_min_m')
    depth_max_m = LaunchConfiguration('depth_max_m')

    camera_name = LaunchConfiguration('camera_name')
    camera_namespace = LaunchConfiguration('camera_namespace')
    serial_no = LaunchConfiguration('serial_no')
    enable_sync = LaunchConfiguration('enable_sync')
    color_profile = LaunchConfiguration('color_profile')
    depth_profile = LaunchConfiguration('depth_profile')
    fps = LaunchConfiguration('fps')

    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare('realsense2_camera'),
                'launch',
                'rs_launch.py'
            ])
        ),
        launch_arguments={
            'camera_name': camera_name,
            'camera_namespace': camera_namespace,
            'serial_no': serial_no,
            'enable_color': 'true',
            'enable_depth': 'true',
            'align_depth.enable': 'true',
            'enable_sync': enable_sync,
            'rgb_camera.profile': color_profile,
            'depth_module.profile': depth_profile,
            'fps': fps,
        }.items()
    )

    hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='hand_landmarks_node',
        name='hand_landmarks_node',
        output='screen',
        parameters=[
            hand_landmarks_config_file,
            {
                'image_topic': image_topic,
                'use_depth': True,
                'depth_topic': depth_topic,
                'camera_info_topic': camera_info_topic,
                'depth_time_tolerance_ms': depth_time_tolerance_ms,
                'depth_min_m': depth_min_m,
                'depth_max_m': depth_max_m,
            }
        ]
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_name',
            default_value='camera',
            description='RealSense camera node name'
        ),
        DeclareLaunchArgument(
            'camera_namespace',
            default_value='',
            description='RealSense camera namespace'
        ),
        DeclareLaunchArgument(
            'serial_no',
            default_value='',
            description='RealSense serial number (empty = first detected camera)'
        ),
        DeclareLaunchArgument(
            'enable_sync',
            default_value='true',
            description='Enable hardware stream synchronization in RealSense driver'
        ),
        DeclareLaunchArgument(
            'fps',
            default_value='30',
            description='Camera FPS used by RealSense driver'
        ),
        DeclareLaunchArgument(
            'color_profile',
            default_value='640x480x30',
            description='RealSense RGB profile (WxHxFPS)'
        ),
        DeclareLaunchArgument(
            'depth_profile',
            default_value='640x480x30',
            description='RealSense depth profile (WxHxFPS)'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/color/image_raw',
            description='RGB image topic consumed by hand_landmarks_node'
        ),
        DeclareLaunchArgument(
            'depth_topic',
            default_value='/camera/aligned_depth_to_color/image_raw',
            description='Depth image topic aligned with image_topic'
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
        realsense_launch,
        hand_landmarks_node,
    ])
