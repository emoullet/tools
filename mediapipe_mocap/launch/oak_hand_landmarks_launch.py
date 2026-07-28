# Copyright 2026 Etienne Moullet
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the Etienne Moullet nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    """Create the standalone OAK 3D hand landmarks launch description."""
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    oak_hand_landmarks_config = os.path.join(
        package_share_dir,
        'config',
        'oak_hand_landmarks_node.yaml',
    )

    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='50.0',
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
        description='Show local OpenCV visualization window in oak_hand_landmarks_node',
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
        description='XYZ saturation distance used by normalization and feedback overlay',
    )

    landmark_index_arg = DeclareLaunchArgument(
        'landmark_index',
        default_value='0',
        description='Tracked landmark index (0-20) for OAK feedback overlay',
    )

    oak_hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='oak_hand_landmarks_node',
        name='oak_hand_landmarks_node',
        output='screen',
        parameters=[
            LaunchConfiguration('oak_config_file'),
            {
                'fps': ParameterValue(LaunchConfiguration('fps'), value_type=float),
                'rgb_width': ParameterValue(
                    LaunchConfiguration('rgb_width'),
                    value_type=int,
                ),
                'rgb_height': ParameterValue(
                    LaunchConfiguration('rgb_height'),
                    value_type=int,
                ),
                'visualize': ParameterValue(
                    LaunchConfiguration('visualize'),
                    value_type=bool,
                ),
                'window_name': LaunchConfiguration('window_name'),
                'publish_normalized_landmarks': ParameterValue(
                    LaunchConfiguration('publish_normalized_landmarks'),
                    value_type=bool,
                ),
                'raw_landmarks_topic': LaunchConfiguration('raw_landmarks_topic'),
                'dead_zone': ParameterValue(
                    LaunchConfiguration('dead_zone'),
                    value_type=float,
                ),
                'saturation_zone': ParameterValue(
                    LaunchConfiguration('saturation_zone'),
                    value_type=float,
                ),
                'tracked_landmark_index': ParameterValue(
                    LaunchConfiguration('landmark_index'),
                    value_type=int,
                ),
            },
        ],
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
