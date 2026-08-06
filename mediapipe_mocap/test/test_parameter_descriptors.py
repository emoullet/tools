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

"""Tests for producer parameter declaration metadata."""

from mediapipe_mocap.parameter_descriptors import (
    hand_landmarks_parameter_declarations,
    oak_hand_landmarks_parameter_declarations,
)


RGB_PARAMETER_NAMES = {
    'image_topic',
    'landmarks_topic',
    'model_path',
    'num_hands',
    'min_hand_detection_confidence',
    'min_hand_presence_confidence',
    'min_tracking_confidence',
    'delegate',
    'selfie_mode',
    'enable_one_euro_filter',
    'one_euro_frequency',
    'one_euro_mincutoff',
    'one_euro_beta',
    'one_euro_derivative_cutoff',
    'visualize',
    'window_name',
    'reset_reference_topic',
    'reset_reference_cooldown_sec',
    'initial_reference',
    'show_control_overlay',
    'overlay_dead_zone',
    'overlay_saturation_zone',
    'overlay_normalization_mode',
    'tracked_landmark_index',
}

OAK_PARAMETER_NAMES = {
    'landmarks_topic',
    'model_path',
    'num_hands',
    'min_hand_detection_confidence',
    'min_hand_presence_confidence',
    'min_tracking_confidence',
    'delegate',
    'camera_frame_id',
    'rgb_width',
    'rgb_height',
    'fps',
    'rgb_socket',
    'left_socket',
    'right_socket',
    'stereo_preset',
    'stereo_left_right_check',
    'stereo_subpixel',
    'stereo_extended_disparity',
    'stereo_rectify_edge_fill_color',
    'sync_threshold_ms',
    'sync_attempts',
    'sync_run_on_host',
    'depth_sample_radius_px',
    'min_depth_m',
    'max_depth_m',
    'depth_percentile',
    'missing_depth_strategy',
    'max_missing_depth_landmarks',
    'show_control_overlay',
    'overlay_dead_zone',
    'overlay_saturation_zone',
    'overlay_normalization_mode',
    'tracked_landmark_index',
    'initial_reference',
    'auto_reference_on_first_detection',
    'reset_reference_topic',
    'reset_reference_cooldown_sec',
    'enable_one_euro_filter',
    'one_euro_mincutoff',
    'one_euro_beta',
    'one_euro_derivative_cutoff',
    'visualize',
    'window_name',
}


def _declarations_by_name(declarations):
    """Convert declaration tuples into a mapping and reject duplicates."""
    result = {name: (default, descriptor) for name, default, descriptor in declarations}
    assert len(result) == len(declarations)
    return result


def test_rgb_declarations_cover_the_public_parameter_surface():
    """Keep every RGB producer parameter described."""
    declarations = _declarations_by_name(
        hand_landmarks_parameter_declarations('/test/model.task')
    )

    assert set(declarations) == RGB_PARAMETER_NAMES
    assert declarations['model_path'][0] == '/test/model.task'
    assert declarations['one_euro_frequency'][0] == 30.0
    assert declarations['show_control_overlay'][0] is False


def test_oak_declarations_cover_the_public_parameter_surface():
    """Keep every OAK producer parameter described."""
    declarations = _declarations_by_name(
        oak_hand_landmarks_parameter_declarations('/test/model.task')
    )

    assert set(declarations) == OAK_PARAMETER_NAMES
    assert declarations['model_path'][0] == '/test/model.task'
    assert declarations['fps'][0] == 50.0
    assert declarations['visualize'][0] is False


def test_every_parameter_has_a_description():
    """Ensure ``ros2 param describe`` never returns empty producer help."""
    declarations = [
        *hand_landmarks_parameter_declarations('/test/model.task'),
        *oak_hand_landmarks_parameter_declarations('/test/model.task'),
    ]

    assert all(descriptor.description for _, _, descriptor in declarations)


def test_bounded_parameters_expose_numeric_ranges():
    """Expose natural bounded domains through standard ROS range fields."""
    rgb = _declarations_by_name(
        hand_landmarks_parameter_declarations('/test/model.task')
    )
    oak = _declarations_by_name(
        oak_hand_landmarks_parameter_declarations('/test/model.task')
    )

    confidence_range = rgb['min_hand_detection_confidence'][1].floating_point_range[0]
    assert confidence_range.from_value == 0.0
    assert confidence_range.to_value == 1.0

    landmark_range = rgb['tracked_landmark_index'][1].integer_range[0]
    assert landmark_range.from_value == 0
    assert landmark_range.to_value == 20

    percentile_range = oak['depth_percentile'][1].floating_point_range[0]
    assert percentile_range.from_value == 0.0
    assert percentile_range.to_value == 100.0


def test_enum_parameters_list_supported_values():
    """Make enum-like string choices discoverable from the ROS descriptor."""
    rgb = _declarations_by_name(
        hand_landmarks_parameter_declarations('/test/model.task')
    )
    oak = _declarations_by_name(
        oak_hand_landmarks_parameter_declarations('/test/model.task')
    )

    assert "'AUTO', 'CPU', 'GPU'" in rgb['delegate'][1].additional_constraints
    assert "'axis', 'vector'" in (
        rgb['overlay_normalization_mode'][1].additional_constraints
    )
    assert "'skip_frame', 'reuse_last', 'hand_median'" in (
        oak['missing_depth_strategy'][1].additional_constraints
    )


def test_overlay_descriptors_are_explicitly_display_only():
    """Prevent producer overlay metadata from implying wire-level normalization."""
    for declarations in (
        hand_landmarks_parameter_declarations('/test/model.task'),
        oak_hand_landmarks_parameter_declarations('/test/model.task'),
    ):
        by_name = _declarations_by_name(declarations)
        constraints = by_name['show_control_overlay'][1].additional_constraints
        assert 'Display only' in constraints
        assert 'never changes the published PointCloud' in constraints
