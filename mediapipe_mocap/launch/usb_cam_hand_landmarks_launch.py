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

"""Launch usb_cam and the MediaPipe hand-landmarks detector."""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _launch_pipeline(context):
    """Create nodes after resolving optional usb_cam parameter overrides."""
    usb_cam_overrides = {}
    override_specs = (
        ('video_device', str),
        ('framerate', float),
        ('image_width', int),
        ('image_height', int),
        ('pixel_format', str),
        ('frame_id', str),
    )
    for parameter_name, value_type in override_specs:
        value = LaunchConfiguration(parameter_name).perform(context)
        if value:
            usb_cam_overrides[parameter_name] = value_type(value)

    image_topic = LaunchConfiguration('image_topic').perform(context)
    topic_prefix, separator, _ = image_topic.rpartition('/')
    camera_info_topic = (
        f'{topic_prefix}/camera_info' if separator else 'camera_info'
    )

    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[
            LaunchConfiguration('usb_cam_params_file'),
            usb_cam_overrides,
        ],
        remappings=[
            ('image_raw', image_topic),
            ('camera_info', camera_info_topic),
        ],
    )

    hand_landmarks_node = Node(
        package='mediapipe_mocap',
        executable='hand_landmarks_node',
        name='hand_landmarks_node',
        output='screen',
        parameters=[
            LaunchConfiguration('hand_landmarks_params_file'),
            {
                'image_topic': image_topic,
                'selfie_mode': ParameterValue(
                    LaunchConfiguration('selfie_mode'),
                    value_type=bool,
                ),
            },
        ],
    )

    return [usb_cam_node, hand_landmarks_node]


def generate_launch_description():
    """Create the usb_cam-to-hand-landmarks pipeline launch description."""
    package_share_dir = get_package_share_directory('mediapipe_mocap')
    default_usb_cam_config = os.path.join(
        package_share_dir,
        'config',
        'usb_cam.yaml',
    )
    default_hand_landmarks_config = os.path.join(
        package_share_dir,
        'config',
        'hand_landmarks_node.yaml',
    )

    arguments = [
        DeclareLaunchArgument(
            'usb_cam_params_file',
            default_value=default_usb_cam_config,
            description='usb_cam YAML parameter file',
        ),
        DeclareLaunchArgument(
            'hand_landmarks_params_file',
            default_value=default_hand_landmarks_config,
            description='Hand-landmarks YAML parameter file',
        ),
        DeclareLaunchArgument(
            'video_device',
            default_value='',
            description='Override usb_cam video device; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'framerate',
            default_value='',
            description='Override camera frame rate; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'image_width',
            default_value='',
            description='Override image width; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'image_height',
            default_value='',
            description='Override image height; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'pixel_format',
            default_value='',
            description='Override usb_cam pixel format; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'frame_id',
            default_value='',
            description='Override camera frame ID; empty uses YAML value',
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/color/image_raw',
            description='Raw image topic consumed by the detector',
        ),
        DeclareLaunchArgument(
            'selfie_mode',
            default_value='true',
            description='Mirror detector input horizontally',
        ),
    ]

    return LaunchDescription([
        *arguments,
        OpaqueFunction(function=_launch_pipeline),
    ])
