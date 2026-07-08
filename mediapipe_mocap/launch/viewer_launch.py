from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    """Create the hand landmarks viewer launch description."""
    return LaunchDescription([
        Node(
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
    ])
