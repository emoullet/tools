from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    hand_landmarks_config_file = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml'
    )

    video_device = LaunchConfiguration('video_device')
    image_topic = LaunchConfiguration('image_topic')
    camera_name = LaunchConfiguration('camera_name')
    container_executable = LaunchConfiguration('container_executable')
    use_depth = LaunchConfiguration('use_depth')
    depth_topic = LaunchConfiguration('depth_topic')
    camera_info_topic = LaunchConfiguration('camera_info_topic')
    depth_time_tolerance_ms = LaunchConfiguration('depth_time_tolerance_ms')
    depth_min_m = LaunchConfiguration('depth_min_m')
    depth_max_m = LaunchConfiguration('depth_max_m')

    return LaunchDescription([
        DeclareLaunchArgument(
            'video_device',
            default_value='/dev/video2',
            description='Video device used by usb_cam'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/image_raw',
            description='Image topic published by usb_cam and consumed by hand_landmarks_node'
        ),
        DeclareLaunchArgument(
            'camera_name',
            default_value='webcam',
            description='Camera name for usb_cam'
        ),
        DeclareLaunchArgument(
            'container_executable',
            default_value='component_container_mt',
            description='Composable container executable (e.g. component_container or component_container_mt)'
        ),
        DeclareLaunchArgument(
            'use_depth',
            default_value='false',
            description='Enable RGB+depth fusion for metric 3D landmarks'
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

        ComposableNodeContainer(
            name='vision_container',
            namespace='',
            package='rclcpp_components',
            executable=container_executable,
            composable_node_descriptions=[
                ComposableNode(
                    package='usb_cam',
                    plugin='usb_cam::UsbCamNode',
                    name='camera_webcam',
                    parameters=[{
                        'video_device': video_device,
                        'camera_name': camera_name,
                        'frame_id': camera_name,
                        'image_width': 640,
                        'image_height': 480,
                        'framerate': 30.0,
                        'pixel_format': 'mjpeg2rgb'
                    }],
                    remappings=[
                        ('/image_raw', image_topic),
                    ],
                    extra_arguments=[{'use_intra_process_comms': True}]
                )
            ],
            output='screen',
        ),

        Node(
            package='mediapipe_mocap',
            executable='hand_landmarks_node',
            name='hand_landmarks_node',
            output='screen',
            parameters=[
                hand_landmarks_config_file,
                {
                    'image_topic': image_topic,
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
