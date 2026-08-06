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

"""Parameter contracts shared by the RGB and OAK hand-landmark nodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from rcl_interfaces.msg import (
    FloatingPointRange,
    IntegerRange,
    ParameterDescriptor,
)


ParameterDeclaration = tuple[str, Any, ParameterDescriptor]


def _descriptor(
    description: str,
    *,
    constraints: str = '',
    floating_range: tuple[float, float] | None = None,
    integer_range: tuple[int, int] | None = None,
) -> ParameterDescriptor:
    """Build a parameter descriptor with optional inclusive numeric bounds."""
    descriptor = ParameterDescriptor(
        description=description,
        additional_constraints=constraints,
    )
    if floating_range is not None:
        descriptor.floating_point_range = [
            FloatingPointRange(
                from_value=floating_range[0],
                to_value=floating_range[1],
                step=0.0,
            )
        ]
    if integer_range is not None:
        descriptor.integer_range = [
            IntegerRange(
                from_value=integer_range[0],
                to_value=integer_range[1],
                step=1,
            )
        ]
    return descriptor


def _parameter(
    name: str,
    default: Any,
    description: str,
    *,
    constraints: str = '',
    floating_range: tuple[float, float] | None = None,
    integer_range: tuple[int, int] | None = None,
) -> ParameterDeclaration:
    """Create one declaration tuple accepted by ``declare_parameters``."""
    return (
        name,
        default,
        _descriptor(
            description,
            constraints=constraints,
            floating_range=floating_range,
            integer_range=integer_range,
        ),
    )


def _landmarker_parameters(
    default_model_path: str,
) -> list[ParameterDeclaration]:
    """Return parameters common to both MediaPipe producer nodes."""
    confidence_range = (0.0, 1.0)
    return [
        _parameter(
            'landmarks_topic',
            '/hand_landmarks',
            'Output sensor_msgs/PointCloud topic containing 21 filtered landmarks.',
        ),
        _parameter(
            'model_path',
            default_model_path,
            'MediaPipe hand-landmarker task model path.',
            constraints='An empty value selects the bundled model.',
        ),
        _parameter(
            'num_hands',
            1,
            'Maximum number of hands detected by MediaPipe; only the first is published.',
            constraints='Must be at least 1.',
        ),
        _parameter(
            'min_hand_detection_confidence',
            0.5,
            'Minimum MediaPipe palm-detection confidence.',
            floating_range=confidence_range,
        ),
        _parameter(
            'min_hand_presence_confidence',
            0.5,
            'Minimum MediaPipe hand-presence confidence.',
            floating_range=confidence_range,
        ),
        _parameter(
            'min_tracking_confidence',
            0.5,
            'Minimum MediaPipe landmark-tracking confidence.',
            floating_range=confidence_range,
        ),
        _parameter(
            'delegate',
            'AUTO',
            'MediaPipe inference delegate selection.',
            constraints="One of: 'AUTO', 'CPU', 'GPU'.",
        ),
    ]


def _filter_parameters(
    *,
    include_frequency: bool,
) -> list[ParameterDeclaration]:
    """Return the shared One Euro filter parameter declarations."""
    parameters = [
        _parameter(
            'enable_one_euro_filter',
            True,
            'Enable One Euro filtering before reference, publication, and display.',
        ),
    ]
    if include_frequency:
        parameters.append(
            _parameter(
                'one_euro_frequency',
                30.0,
                'Nominal One Euro filter sampling frequency in hertz.',
                constraints='Must be greater than 0 Hz.',
            )
        )
    parameters.extend(
        [
            _parameter(
                'one_euro_mincutoff',
                1.0,
                'One Euro minimum cutoff frequency in hertz.',
                constraints='Must be greater than 0 Hz.',
            ),
            _parameter(
                'one_euro_beta',
                0.1,
                'Dimensionless One Euro speed coefficient.',
                constraints='Must be greater than or equal to 0.',
            ),
            _parameter(
                'one_euro_derivative_cutoff',
                1.0,
                'One Euro derivative cutoff frequency in hertz.',
                constraints='Must be greater than 0 Hz.',
            ),
        ]
    )
    return parameters


def _viewer_parameters(
    window_name: str,
    *,
    metric_overlay: bool,
) -> list[ParameterDeclaration]:
    """Return visualization and display-only control-overlay declarations."""
    overlay_units = 'meters' if metric_overlay else 'aspect-corrected normalized coordinates'
    return [
        _parameter(
            'visualize',
            False,
            'Open the producer OpenCV diagnostics window.',
        ),
        _parameter(
            'window_name',
            window_name,
            'OpenCV diagnostics window title.',
            constraints='Used only when visualize is true.',
        ),
        _parameter(
            'show_control_overlay',
            False,
            'Display control dead-zone, saturation, and normalized preview.',
            constraints='Display only; never changes the published PointCloud.',
        ),
        _parameter(
            'overlay_dead_zone',
            0.05,
            f'Display-only control dead-zone radius in {overlay_units}.',
            constraints='Must be greater than or equal to 0; ignored when overlay is disabled.',
        ),
        _parameter(
            'overlay_saturation_zone',
            0.3,
            f'Display-only control saturation radius in {overlay_units}.',
            constraints='Must be greater than 0; ignored when overlay is disabled.',
        ),
        _parameter(
            'overlay_normalization_mode',
            'vector',
            'Display-only control normalization geometry.',
            constraints=(
                "One of: 'axis', 'vector'; ignored when control overlay is disabled."
            ),
        ),
        _parameter(
            'tracked_landmark_index',
            0,
            'MediaPipe landmark index used for reference reset and control preview.',
            integer_range=(0, 20),
        ),
    ]


def hand_landmarks_parameter_declarations(
    default_model_path: str,
) -> Sequence[ParameterDeclaration]:
    """Return the complete RGB producer parameter declaration contract."""
    return [
        _parameter(
            'image_topic',
            '/camera/color/image_raw',
            'Input sensor_msgs/Image topic consumed with latest-frame reliable QoS.',
        ),
        *_landmarker_parameters(default_model_path),
        _parameter(
            'selfie_mode',
            False,
            'Mirror input images horizontally before detection.',
        ),
        *_filter_parameters(include_frequency=True),
        _parameter(
            'reset_reference_topic',
            '/reset_reference',
            'Input std_msgs/Bool topic; a rising edge recenters on the next valid hand.',
        ),
        _parameter(
            'reset_reference_cooldown_sec',
            0.25,
            'Minimum time between accepted reference-reset requests in seconds.',
            constraints='Must be greater than or equal to 0 s.',
        ),
        _parameter(
            'initial_reference',
            [0.5, 0.5, 0.5],
            'Initial [x, y, z] reference in normalized image coordinates.',
            constraints='Exactly three floating-point values; z is retained but output is planar.',
        ),
        *_viewer_parameters(
            'Hand Landmarks (Node)',
            metric_overlay=False,
        ),
    ]


def oak_hand_landmarks_parameter_declarations(
    default_model_path: str,
) -> Sequence[ParameterDeclaration]:
    """Return the complete OAK producer parameter declaration contract."""
    return [
        *_landmarker_parameters(default_model_path),
        _parameter(
            'camera_frame_id',
            'oak_rgb_camera_optical_frame',
            'RGB optical camera frame assigned to published PointCloud headers.',
        ),
        _parameter(
            'rgb_width',
            640,
            'Requested aligned RGB and depth image width in pixels.',
            constraints='Must be at least 1 px.',
        ),
        _parameter(
            'rgb_height',
            400,
            'Requested aligned RGB and depth image height in pixels.',
            constraints='Must be at least 1 px.',
        ),
        _parameter(
            'fps',
            50.0,
            'Requested OAK capture and One Euro sampling frequency in hertz.',
            constraints='Must be greater than 0 Hz.',
        ),
        _parameter(
            'rgb_socket',
            'CAM_A',
            'DepthAI camera-board socket used for RGB.',
            constraints='Must name a depthai.CameraBoardSocket member.',
        ),
        _parameter(
            'left_socket',
            'CAM_B',
            'DepthAI camera-board socket used for the left mono camera.',
            constraints='Must name a depthai.CameraBoardSocket member.',
        ),
        _parameter(
            'right_socket',
            'CAM_C',
            'DepthAI camera-board socket used for the right mono camera.',
            constraints='Must name a depthai.CameraBoardSocket member.',
        ),
        _parameter(
            'stereo_preset',
            'FAST_DENSITY',
            'DepthAI StereoDepth preset mode.',
            constraints='Must name a depthai StereoDepth.PresetMode member.',
        ),
        _parameter(
            'stereo_left_right_check',
            True,
            'Enable DepthAI stereo left-right consistency checking.',
        ),
        _parameter(
            'stereo_subpixel',
            False,
            'Enable DepthAI stereo subpixel disparity.',
        ),
        _parameter(
            'stereo_extended_disparity',
            False,
            'Enable DepthAI extended disparity for closer minimum depth.',
        ),
        _parameter(
            'stereo_rectify_edge_fill_color',
            0,
            'DepthAI stereo rectification edge-fill color.',
            integer_range=(0, 255),
        ),
        _parameter(
            'sync_threshold_ms',
            15.0,
            'Maximum RGB/depth synchronization offset in milliseconds.',
            constraints='Must be greater than or equal to 0 ms.',
        ),
        _parameter(
            'sync_attempts',
            -1,
            'DepthAI synchronization attempt policy.',
            constraints='Must be -1 or greater.',
        ),
        _parameter(
            'sync_run_on_host',
            True,
            'Run the DepthAI RGB/depth synchronizer on the host.',
        ),
        _parameter(
            'depth_sample_radius_px',
            2,
            'Pixel radius sampled around each landmark for valid depth.',
            constraints='Must be greater than or equal to 0 px.',
        ),
        _parameter(
            'min_depth_m',
            0.12,
            'Minimum accepted aligned depth in meters.',
            constraints='Must be greater than or equal to 0 m.',
        ),
        _parameter(
            'max_depth_m',
            3.0,
            'Maximum accepted aligned depth in meters.',
            constraints='Must be greater than min_depth_m.',
        ),
        _parameter(
            'depth_percentile',
            50.0,
            'Percentile selected from valid depth samples around a landmark.',
            floating_range=(0.0, 100.0),
        ),
        _parameter(
            'missing_depth_strategy',
            'reuse_last',
            'Fallback strategy for landmarks without a direct valid depth sample.',
            constraints="One of: 'skip_frame', 'reuse_last', 'hand_median'.",
        ),
        _parameter(
            'max_missing_depth_landmarks',
            8,
            'Maximum missing direct-depth samples accepted in one 21-point hand.',
            integer_range=(0, 21),
        ),
        *_viewer_parameters(
            '3D Hand Landmarks OAK',
            metric_overlay=True,
        ),
        _parameter(
            'initial_reference',
            [0.0, 0.0, 0.6],
            'Initial [x, y, z] reference in RGB-camera coordinates, in meters.',
            constraints='Exactly three floating-point values.',
        ),
        _parameter(
            'auto_reference_on_first_detection',
            True,
            'Initialize the reference from the first hand with valid metric depth.',
        ),
        _parameter(
            'reset_reference_topic',
            '/reset_reference',
            'Input std_msgs/Bool topic; a rising edge recenters on the next valid 3D hand.',
        ),
        _parameter(
            'reset_reference_cooldown_sec',
            0.25,
            'Minimum time between accepted reference-reset requests in seconds.',
            constraints='Must be greater than or equal to 0 s.',
        ),
        *_filter_parameters(include_frequency=False),
    ]
