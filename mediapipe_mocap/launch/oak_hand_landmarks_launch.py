from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
import os


def generate_launch_description():
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    oak_hand_landmarks_config = os.path.join(
        package_share_dir,
        'config',
        '3d_hand_landmarks_oak_node.yaml',
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30.0',
        description='OAK camera FPS',
    )

    rgb_width_arg = DeclareLaunchArgument(
        'rgb_width',
        default_value='640',
        description='OAK RGB/depth output width in pixels',
    )

    rgb_height_arg = DeclareLaunchArgument(
        'rgb_height',
        default_value='400',
        description='OAK RGB/depth output height in pixels',
    )

    visualize_arg = DeclareLaunchArgument(
        'visualize',
        default_value='true',
        description='Show local OpenCV visualization window in 3d_hand_landmarks_oak_node',
    )

    window_name_arg = DeclareLaunchArgument(
        'window_name',
        default_value='3D Hand Landmarks OAK',
        description='OpenCV window title when visualize is enabled',
    )

    oak_config_file_arg = DeclareLaunchArgument(
        'oak_config_file',
        default_value=oak_hand_landmarks_config,
        description='Path to OAK 3D hand landmarks config file',
    )

    publish_normalized_landmarks_arg = DeclareLaunchArgument(
        'publish_normalized_landmarks',
        default_value='true',
        description='Publish normalized control landmarks instead of metric 3D landmarks',
    )

    raw_landmarks_topic_arg = DeclareLaunchArgument(
        'raw_landmarks_topic',
        default_value='',
        description='Optional topic for metric camera-frame landmarks before normalization',
    )

    dead_zone_arg = DeclareLaunchArgument(
        'dead_zone',
        default_value='0.05',
        description='Dead zone radius used by the OAK feedback overlay',
    )

    saturation_zone_arg = DeclareLaunchArgument(
        'saturation_zone',
        default_value='0.4',
        description='XYZ saturation distance used by the OAK feedback overlay',
    )

    landmark_index_arg = DeclareLaunchArgument(
        'landmark_index',
        default_value='0',
        description='Tracked landmark index (0-20) for OAK feedback overlay',
    )

    oak_hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='3d_hand_landmarks_oak_node',
        name='hand_landmarks_oak_3d_node',
        output='screen',
        parameters=[oak_hand_landmarks_config]
    )

    return LaunchDescription([
        fps_arg,
        rgb_width_arg,
        rgb_height_arg,
        visualize_arg,
        window_name_arg,
        oak_config_file_arg,
        publish_normalized_landmarks_arg,
        raw_landmarks_topic_arg,
        dead_zone_arg,
        saturation_zone_arg,
        landmark_index_arg,
        oak_hand_landmarks_node,
    ])
