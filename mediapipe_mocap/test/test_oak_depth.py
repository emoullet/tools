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

"""Tests for hardware-independent OAK depth processing."""

from types import SimpleNamespace

from mediapipe_mocap.oak_depth import (
    back_project,
    CameraIntrinsics,
    DepthProcessingConfig,
    MissingDepthStrategy,
    OakDepthProjector,
    parse_missing_depth_strategy,
    prepare_metric_hand_for_output,
    project_to_normalized_image,
)
import numpy as np
import pytest


def _config(
    strategy: MissingDepthStrategy = MissingDepthStrategy.REUSE_LAST,
    max_missing: int = 8,
    sample_radius: int = 0,
) -> DepthProcessingConfig:
    return DepthProcessingConfig(
        sample_radius_px=sample_radius,
        min_depth_m=0.5,
        max_depth_m=2.0,
        percentile=50.0,
        missing_strategy=strategy,
        max_missing_landmarks=max_missing,
    )


def _projector(
    strategy: MissingDepthStrategy = MissingDepthStrategy.REUSE_LAST,
    max_missing: int = 8,
) -> OakDepthProjector:
    return OakDepthProjector(
        CameraIntrinsics(fx=100.0, fy=200.0, cx=1.0, cy=1.0),
        _config(strategy, max_missing),
        hand_slots=1,
    )


def _landmarks(*coordinates):
    return [
        SimpleNamespace(x=x_coordinate, y=y_coordinate)
        for x_coordinate, y_coordinate in coordinates
    ]


def test_camera_intrinsics_are_extracted_from_calibration_matrix():
    """Calibration conversion should retain focal lengths and principal point."""
    intrinsics = CameraIntrinsics.from_matrix(
        [
            [100.0, 0.0, 20.0],
            [0.0, 200.0, 30.0],
            [0.0, 0.0, 1.0],
        ]
    )

    assert intrinsics == CameraIntrinsics(100.0, 200.0, 20.0, 30.0)


def test_depth_sampling_filters_range_and_uses_percentile():
    """Sampling should ignore invalid values and convert millimeters to meters."""
    depth_mm = np.array(
        [
            [0, 400, 0],
            [500, 1000, 1500],
            [2000, 2500, 0],
        ],
        dtype=np.uint16,
    )

    projector = OakDepthProjector(
        CameraIntrinsics(fx=100.0, fy=200.0, cx=1.0, cy=1.0),
        _config(sample_radius=1),
        hand_slots=1,
    )

    sampled = projector.sample_depth_m(depth_mm, 1, 1)

    assert sampled == pytest.approx(1.25)


def test_back_projection_uses_rgb_camera_intrinsics():
    """Pinhole projection should return metric camera-frame coordinates."""
    point = back_project(
        u=11.0,
        v=21.0,
        depth_m=2.0,
        intrinsics=CameraIntrinsics(100.0, 200.0, 1.0, 1.0),
    )

    assert point.x == pytest.approx(0.2)
    assert point.y == pytest.approx(0.2)
    assert point.z == pytest.approx(2.0)


def test_metric_projection_is_inverse_of_back_projection():
    """Filtered camera points should reproject to their source image location."""
    intrinsics = CameraIntrinsics(100.0, 200.0, 1.0, 1.0)
    metric_point = back_project(
        u=21.0,
        v=31.0,
        depth_m=1.5,
        intrinsics=intrinsics,
    )

    image_point = project_to_normalized_image(
        metric_point,
        intrinsics,
        image_width=101,
        image_height=201,
    )

    assert image_point is not None
    assert image_point.x == pytest.approx(0.21)
    assert image_point.y == pytest.approx(0.155)
    assert image_point.z == pytest.approx(1.5)


@pytest.mark.parametrize(
    ('point', 'intrinsics', 'width', 'height'),
    [
        (
            SimpleNamespace(x=0.0, y=0.0, z=0.0),
            CameraIntrinsics(100.0, 200.0, 1.0, 1.0),
            101,
            201,
        ),
        (
            SimpleNamespace(x=float('nan'), y=0.0, z=1.0),
            CameraIntrinsics(100.0, 200.0, 1.0, 1.0),
            101,
            201,
        ),
        (
            SimpleNamespace(x=0.0, y=0.0, z=1.0),
            CameraIntrinsics(0.0, 200.0, 1.0, 1.0),
            101,
            201,
        ),
        (
            SimpleNamespace(x=0.0, y=0.0, z=1.0),
            CameraIntrinsics(100.0, 200.0, 1.0, 1.0),
            1,
            201,
        ),
    ],
)
def test_metric_projection_rejects_invalid_inputs(
    point,
    intrinsics,
    width,
    height,
):
    """Invalid filtered points should skip display instead of using raw pixels."""
    assert project_to_normalized_image(
        point,
        intrinsics,
        width,
        height,
    ) is None


def test_filtered_metric_points_are_reprojected_without_mutating_input():
    """Output and display should share one transformed metric collection."""
    source_points = [
        SimpleNamespace(x=0.0, y=0.0, z=1.0),
        SimpleNamespace(x=0.1, y=0.2, z=1.0),
    ]
    calls = []

    def transform(index, point):
        calls.append(index)
        return SimpleNamespace(
            x=point.x + 0.1,
            y=point.y,
            z=point.z + 1.0,
        )

    prepared = prepare_metric_hand_for_output(
        source_points,
        CameraIntrinsics(100.0, 100.0, 0.0, 0.0),
        image_width=101,
        image_height=101,
        transform=transform,
    )

    assert prepared is not None
    metric_points, image_points = prepared
    assert calls == [0, 1]
    assert metric_points[0].x == pytest.approx(0.1)
    assert metric_points[0].z == pytest.approx(2.0)
    assert image_points[0].x == pytest.approx(0.05)
    assert image_points[0].y == pytest.approx(0.0)
    assert image_points[0].z == pytest.approx(metric_points[0].z)
    assert source_points[0].x == pytest.approx(0.0)
    assert source_points[0].z == pytest.approx(1.0)


def test_invalid_filtered_projection_rejects_the_entire_display_hand():
    """A bad post-filter point should never fall back to raw image landmarks."""
    prepared = prepare_metric_hand_for_output(
        [SimpleNamespace(x=0.0, y=0.0, z=1.0)],
        CameraIntrinsics(100.0, 100.0, 0.0, 0.0),
        image_width=101,
        image_height=101,
        transform=lambda _index, point: SimpleNamespace(
            x=point.x,
            y=point.y,
            z=0.0,
        ),
    )

    assert prepared is None


def test_skip_frame_strategy_rejects_any_missing_depth():
    """The skip strategy should not produce partially recovered hands."""
    result = _projector(
        MissingDepthStrategy.SKIP_FRAME
    ).project_hand(
        _landmarks((0.0, 0.0), (1.0, 1.0)),
        np.array([[1000, 0], [0, 0]], dtype=np.uint16),
        hand_index=0,
    )

    assert not result.valid
    assert result.missing_depth_count == 1


def test_hand_median_strategy_fills_missing_depth():
    """The median strategy should use valid samples from the current hand."""
    result = _projector(
        MissingDepthStrategy.HAND_MEDIAN
    ).project_hand(
        _landmarks((0.0, 0.0), (1.0, 1.0)),
        np.array([[1000, 0], [0, 0]], dtype=np.uint16),
        hand_index=0,
    )

    assert result.valid
    assert result.missing_depth_count == 1
    assert result.metric_points[1].z == pytest.approx(1.0)


def test_reuse_last_strategy_uses_landmark_history():
    """The reuse strategy should prefer the same landmark's prior depth."""
    projector = _projector(MissingDepthStrategy.REUSE_LAST)
    landmarks = _landmarks((0.0, 0.0), (1.0, 1.0))
    projector.project_hand(
        landmarks,
        np.array([[1000, 0], [0, 1500]], dtype=np.uint16),
        hand_index=0,
    )

    result = projector.project_hand(
        landmarks,
        np.array([[1000, 0], [0, 0]], dtype=np.uint16),
        hand_index=0,
    )

    assert result.valid
    assert result.missing_depth_count == 1
    assert result.metric_points[1].z == pytest.approx(1.5)


def test_missing_limit_rejects_hand_before_fallback():
    """Too many missing samples should reject even recoverable landmarks."""
    projector = _projector(
        MissingDepthStrategy.REUSE_LAST,
        max_missing=0,
    )
    result = projector.project_hand(
        _landmarks((0.0, 0.0), (1.0, 1.0)),
        np.array([[1000, 0], [0, 0]], dtype=np.uint16),
        hand_index=0,
    )

    assert not result.valid
    assert result.missing_depth_count == 1


def test_invalid_strategy_falls_back_to_reuse_last():
    """Unknown strategy parameters should retain the existing fallback."""
    warnings = []

    strategy = parse_missing_depth_strategy('unknown', warnings.append)

    assert strategy is MissingDepthStrategy.REUSE_LAST
    assert warnings == [
        "Invalid missing_depth_strategy 'unknown', "
        "falling back to 'reuse_last'."
    ]
