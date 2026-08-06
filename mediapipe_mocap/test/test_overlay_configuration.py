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

"""Check producer overlay configuration and standalone launch defaults."""

import ast
from pathlib import Path

import yaml


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_OVERLAY_KEYS = {
    'show_control_overlay',
    'overlay_dead_zone',
    'overlay_saturation_zone',
    'overlay_normalization_mode',
}


def _launch_defaults(path):
    """Return literal defaults for launch arguments in one Python file."""
    defaults = {}
    tree = ast.parse(path.read_text(encoding='utf-8'))
    constants = {
        target.id: assignment.value.value
        for assignment in tree.body
        if isinstance(assignment, ast.Assign)
        and isinstance(assignment.value, ast.Constant)
        for target in assignment.targets
        if isinstance(target, ast.Name)
    }
    for call in (
        node for node in ast.walk(tree) if isinstance(node, ast.Call)
    ):
        if not (
            isinstance(call.func, ast.Name)
            and call.func.id == 'DeclareLaunchArgument'
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            continue
        default = next(
            (
                (
                    keyword.value.value
                    if isinstance(keyword.value, ast.Constant)
                    else constants.get(keyword.value.id)
                )
                for keyword in call.keywords
                if keyword.arg == 'default_value'
                and isinstance(keyword.value, (ast.Constant, ast.Name))
            ),
            None,
        )
        defaults[call.args[0].value] = default
    return defaults


def test_producer_yaml_disables_optional_overlay_by_default():
    """Both shipped producer configurations should use only overlay names."""
    for config_name, node_name in (
        ('hand_landmarks_node.yaml', 'hand_landmarks_node'),
        ('oak_hand_landmarks_node.yaml', 'oak_hand_landmarks_node'),
    ):
        document = yaml.safe_load(
            (_PACKAGE_ROOT / 'config' / config_name).read_text(
                encoding='utf-8'
            )
        )
        parameters = document[node_name]['ros__parameters']
        assert _OVERLAY_KEYS <= parameters.keys()
        assert isinstance(parameters['show_control_overlay'], bool)
        assert parameters['overlay_normalization_mode'] == 'vector'


def test_standalone_launches_defer_parameter_values_to_yaml():
    """Standalone launch arguments should override YAML only when supplied."""
    for launch_name in (
        'hand_landmarks_launch.py',
        'oak_hand_landmarks_launch.py',
    ):
        defaults = _launch_defaults(_PACKAGE_ROOT / 'launch' / launch_name)
        assert defaults['show_control_overlay'] == '__use_yaml__'
        assert defaults['overlay_dead_zone'] == '__use_yaml__'
        assert defaults['overlay_saturation_zone'] == '__use_yaml__'
        assert defaults['overlay_normalization_mode'] == '__use_yaml__'
