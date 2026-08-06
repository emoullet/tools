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

"""Camera-free tests for installed producer launch contracts."""

import importlib.util
import os
from pathlib import Path

from ament_index_python.packages import (
    get_package_prefix,
    get_package_share_directory,
)
import yaml


_PACKAGE_NAME = 'mediapipe_mocap'


def _load_launch_module(launch_name):
    """Load one installed launch file as an isolated Python module."""
    launch_path = (
        Path(get_package_share_directory(_PACKAGE_NAME))
        / 'launch'
        / launch_name
    )
    spec = importlib.util.spec_from_file_location(
        f'_mediapipe_mocap_{launch_path.stem}',
        launch_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _resolve_launch(launch_name, overrides=None):
    """Resolve launch arguments and opaque factories without executing nodes."""
    from launch import LaunchContext
    from launch.actions import DeclareLaunchArgument, OpaqueFunction
    from launch_ros.actions import Node

    module = _load_launch_module(launch_name)
    description = module.generate_launch_description()
    context = LaunchContext()
    context.launch_configurations.update(overrides or {})

    for entity in description.entities:
        if isinstance(entity, DeclareLaunchArgument):
            entity.execute(context)

    nodes = []
    for entity in description.entities:
        if isinstance(entity, OpaqueFunction):
            nodes.extend(entity.execute(context) or [])

    assert nodes
    assert all(isinstance(node, Node) for node in nodes)
    return context, nodes


def _evaluated_parameters(node, context):
    """Evaluate a launch node's normalized parameter inputs in memory."""
    from launch_ros.utilities import evaluate_parameters

    normalized_parameters = vars(node)['_Node__parameters']
    return evaluate_parameters(context, normalized_parameters)


def _parameters_from_file(path, node_name):
    """Read the parameter section applicable to an unnamespaced node."""
    document = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    parameters = {}
    for selector in ('/**', node_name, f'/{node_name}'):
        selected = document.get(selector)
        if selected is not None:
            parameters.update(selected.get('ros__parameters', {}))
    return parameters


def _effective_parameters(node, context):
    """Apply parameter files and dictionaries in launch precedence order."""
    effective = {}
    node_name = vars(node)['_Node__node_name']
    for parameter_source in _evaluated_parameters(node, context):
        if isinstance(parameter_source, Path):
            effective.update(
                _parameters_from_file(parameter_source, node_name)
            )
        else:
            effective.update(parameter_source)
    return effective


def _expanded_remappings(node, context):
    """Evaluate remapping substitutions without executing the node."""
    from launch.utilities import perform_substitutions

    normalized_remappings = vars(node)['_Node__remappings']
    return [
        (
            perform_substitutions(context, source),
            perform_substitutions(context, destination),
        )
        for source, destination in normalized_remappings
    ]


def _node_by_executable(nodes, executable):
    """Return the single resolved node matching an executable name."""
    matching = [node for node in nodes if node.node_executable == executable]
    assert len(matching) == 1
    return matching[0]


def test_installed_entry_points_and_launch_assets_resolve():
    """Ensure installed scripts, launch files, and default configs exist."""
    package_prefix = Path(get_package_prefix(_PACKAGE_NAME))
    package_share = Path(get_package_share_directory(_PACKAGE_NAME))

    for executable in ('hand_landmarks_node', 'oak_hand_landmarks_node'):
        executable_path = (
            package_prefix / 'lib' / _PACKAGE_NAME / executable
        )
        assert executable_path.is_file()
        assert os.access(executable_path, os.X_OK)

    for relative_path in (
        'launch/hand_landmarks_launch.py',
        'launch/usb_cam_hand_landmarks_launch.py',
        'launch/oak_hand_landmarks_launch.py',
        'config/hand_landmarks_node.yaml',
        'config/usb_cam.yaml',
        'config/oak_hand_landmarks_node.yaml',
    ):
        assert (package_share / relative_path).is_file()


def test_rgb_standalone_launch_uses_effective_yaml_defaults():
    """Resolve the RGB producer launch without opening a camera."""
    context, nodes = _resolve_launch('hand_landmarks_launch.py')
    node = _node_by_executable(nodes, 'hand_landmarks_node')
    parameters = _effective_parameters(node, context)

    assert node.node_package == _PACKAGE_NAME
    assert parameters['image_topic'] == '/camera/color/image_raw'
    assert parameters['landmarks_topic'] == '/hand_landmarks'
    assert parameters['num_hands'] == 1
    assert parameters['enable_one_euro_filter'] is True
    assert parameters['one_euro_frequency'] == 30.0
    assert parameters['visualize'] is True
    assert parameters['show_control_overlay'] is True
    assert parameters['overlay_normalization_mode'] == 'vector'
    assert 'running_mode' not in parameters


def test_oak_standalone_launch_uses_effective_yaml_defaults():
    """Resolve the OAK producer launch without constructing a pipeline."""
    context, nodes = _resolve_launch('oak_hand_landmarks_launch.py')
    node = _node_by_executable(nodes, 'oak_hand_landmarks_node')
    parameters = _effective_parameters(node, context)

    assert node.node_package == _PACKAGE_NAME
    assert parameters['landmarks_topic'] == '/hand_landmarks'
    assert parameters['camera_frame_id'] == 'oak_rgb_camera_optical_frame'
    assert parameters['rgb_width'] == 640
    assert parameters['rgb_height'] == 400
    assert parameters['fps'] == 50.0
    assert parameters['enable_one_euro_filter'] is True
    assert parameters['visualize'] is True
    assert parameters['show_control_overlay'] is False
    assert parameters['auto_reference_on_first_detection'] is True
    assert 'running_mode' not in parameters


def test_usb_pipeline_resolves_camera_and_producer_defaults():
    """Resolve both USB pipeline nodes and their default topic wiring."""
    context, nodes = _resolve_launch('usb_cam_hand_landmarks_launch.py')
    camera_node = _node_by_executable(nodes, 'usb_cam_node_exe')
    producer_node = _node_by_executable(nodes, 'hand_landmarks_node')

    camera_parameters = _effective_parameters(camera_node, context)
    producer_parameters = _effective_parameters(producer_node, context)
    camera_remappings = _expanded_remappings(camera_node, context)

    assert camera_node.node_package == 'usb_cam'
    assert camera_parameters['video_device'] == '/dev/video0'
    assert camera_parameters['framerate'] == 30.0
    assert camera_parameters['image_width'] == 640
    assert camera_parameters['image_height'] == 480
    assert camera_parameters['pixel_format'] == 'mjpeg2rgb'
    assert camera_remappings == [
        ('image_raw', '/camera/color/image_raw'),
        ('camera_info', '/camera/color/camera_info'),
    ]

    assert producer_parameters['image_topic'] == '/camera/color/image_raw'
    assert producer_parameters['selfie_mode'] is True
    assert producer_parameters['enable_one_euro_filter'] is True
    assert producer_parameters['visualize'] is True


def test_terminal_overrides_win_without_starting_nodes():
    """Exercise launch override parsing and parameter precedence in memory."""
    context, nodes = _resolve_launch(
        'oak_hand_landmarks_launch.py',
        {
            'fps': '42.0',
            'rgb_width': '800',
            'rgb_height': '600',
            'visualize': 'false',
            'show_control_overlay': 'true',
            'overlay_dead_zone': '0.08',
            'overlay_saturation_zone': '0.4',
            'overlay_normalization_mode': 'axis',
            'landmark_index': '8',
        },
    )
    parameters = _effective_parameters(
        _node_by_executable(nodes, 'oak_hand_landmarks_node'),
        context,
    )

    assert parameters['fps'] == 42.0
    assert parameters['rgb_width'] == 800
    assert parameters['rgb_height'] == 600
    assert parameters['visualize'] is False
    assert parameters['show_control_overlay'] is True
    assert parameters['overlay_dead_zone'] == 0.08
    assert parameters['overlay_saturation_zone'] == 0.4
    assert parameters['overlay_normalization_mode'] == 'axis'
    assert parameters['tracked_landmark_index'] == 8
