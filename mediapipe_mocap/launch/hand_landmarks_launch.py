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


def generate_launch_description():
    """Create the standalone 2D hand landmarks launch description."""
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

    return LaunchDescription([
        visualize_arg,
        window_name_arg,
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
                },
            ]
        )
    ])
