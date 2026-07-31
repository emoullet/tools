# Copyright 2026 ISIR-EXTENDER
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
# * Neither the name of the ISIR-EXTENDER nor the names of its
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

"""Launch the RGB MediaPipe hand-landmarks producer."""

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


USE_YAML_DEFAULT = '__use_yaml__'


def _parse_bool(value):
    """Parse a resolved ROS launch boolean."""
    return value.lower() in ('1', 'true', 'yes', 'on')


def _create_node(context):
    """Create the producer with only explicit terminal overrides."""
    overrides = {}
    for argument_name, parameter_name, value_type in (
        ('visualize', 'visualize', _parse_bool),
        ('window_name', 'window_name', str),
        ('show_control_overlay', 'show_control_overlay', _parse_bool),
        ('overlay_dead_zone', 'overlay_dead_zone', float),
        ('overlay_saturation_zone', 'overlay_saturation_zone', float),
        ('overlay_normalization_mode', 'overlay_normalization_mode', str),
    ):
        value = LaunchConfiguration(argument_name).perform(context)
        if value != USE_YAML_DEFAULT:
            overrides[parameter_name] = value_type(value)

    return [
        Node(
            package='mediapipe_mocap',
            executable='hand_landmarks_node',
            name='hand_landmarks_node',
            output='screen',
            parameters=[
                LaunchConfiguration('config_file'),
                overrides,
            ],
        )
    ]


def generate_launch_description():
    """Create the standalone 2D hand landmarks launch description."""
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    config_file = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml',
    )

    arguments = [
        DeclareLaunchArgument(
            'config_file',
            default_value=config_file,
            description='Path to the producer YAML parameter file',
        ),
        DeclareLaunchArgument(
            'visualize',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML OpenCV visualization setting',
        ),
        DeclareLaunchArgument(
            'window_name',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML OpenCV window title',
        ),
        DeclareLaunchArgument(
            'show_control_overlay',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML control-overlay visibility',
        ),
        DeclareLaunchArgument(
            'overlay_dead_zone',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML display-only dead-zone radius',
        ),
        DeclareLaunchArgument(
            'overlay_saturation_zone',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML display-only saturation radius',
        ),
        DeclareLaunchArgument(
            'overlay_normalization_mode',
            default_value=USE_YAML_DEFAULT,
            description='Override the YAML axis or vector overlay mode',
        ),
    ]

    return LaunchDescription([
        *arguments,
        OpaqueFunction(function=_create_node),
    ])
