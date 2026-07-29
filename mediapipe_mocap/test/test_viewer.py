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

"""Tests for the ROS-independent hand-landmark viewer."""

from types import SimpleNamespace

from mediapipe_mocap import viewer
import numpy as np  # noqa: I100,I201


class _Clock:
    """Controllable monotonic clock used by viewer performance tests."""

    def __init__(self):
        """Start at zero seconds."""
        self.now = 0.0

    def __call__(self):
        """Return the current test time."""
        return self.now


def _patch_opencv(monkeypatch, key=-1):
    """Replace OpenCV GUI and drawing calls with recording mocks."""
    calls = {
        name: []
        for name in (
            'circle',
            'ellipse',
            'line',
            'putText',
            'drawMarker',
            'rectangle',
            'imshow',
            'destroyWindow',
        )
    }

    for name in calls:
        monkeypatch.setattr(
            viewer.cv2,
            name,
            lambda *args, _name=name, **kwargs: calls[_name].append(
                (args, kwargs)
            ),
        )
    monkeypatch.setattr(viewer.cv2, 'waitKey', lambda delay: key)
    return calls


def test_show_2d_draws_landmarks_connections_and_reference(monkeypatch):
    """Sensor diagnostics remain visible when the control overlay is disabled."""
    calls = _patch_opencv(monkeypatch, key=ord('q'))
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    hand = [
        SimpleNamespace(x=0.25, y=0.5, z=0.0)
        for _ in range(21)
    ]
    hand_viewer = viewer.HandLandmarksViewer('2D')

    exit_requested = hand_viewer.show_2d(
        image,
        [hand],
        mediapipe_time_sec=0.012,
        reference_xyz=(0.5, 0.5, 0.0),
        tracked_landmark_index=0,
    )

    landmark_circles = [
        args for args, _ in calls['circle']
        if len(args) == 5 and args[2] == 3
    ]
    assert len(landmark_circles) == 21
    assert landmark_circles[0][1] == (50, 50)
    assert len(calls['line']) == len(viewer.HAND_CONNECTIONS)
    assert calls['drawMarker'][0][0][1] == (100, 50)
    assert not any(
        args[1].startswith('CTRL') for args, _ in calls['putText']
    )
    assert calls['imshow'][0][0][0] == '2D'
    assert calls['imshow'][0][0][1] is not image
    assert not np.any(image)
    assert exit_requested


def test_vector_overlay_draws_circles_and_aspect_corrected_preview(monkeypatch):
    """Vector preview should match the producer's aspect-corrected cloud."""
    calls = _patch_opencv(monkeypatch)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    hand = [SimpleNamespace(x=0.65, y=0.5, z=0.0)]
    config = viewer.ControlOverlayConfig(
        dead_zone=0.1,
        saturation_zone=0.5,
        normalization_mode=viewer.OverlayNormalizationMode.VECTOR,
    )

    viewer.HandLandmarksViewer('2D').show_2d(
        image,
        [hand],
        reference_xyz=(0.5, 0.5, 0.0),
        control_overlay=config,
        displacement_scale=(2.0, 1.0, 0.0),
    )

    assert len(calls['rectangle']) == 0
    zone_radii = [
        args[2] for args, _ in calls['circle']
        if len(args) >= 6 and args[4] == 2
    ]
    assert zone_radii == [10, 50, 7]
    text = [args[1] for args, _ in calls['putText']]
    assert 'CTRL vector  DZ: 0.10  SAT: 0.50' in text
    assert (
        'LM[0] ACTIVE d=(0.30, 0.00, 0.00) '
        'n=(0.50, 0.00, 0.00)'
    ) in text


def test_axis_overlay_draws_rectangles_and_per_axis_preview(monkeypatch):
    """Axis preview should use rectangular zones and independent ramps."""
    calls = _patch_opencv(monkeypatch)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    hand = [SimpleNamespace(x=0.85, y=0.55, z=0.0)]
    config = viewer.ControlOverlayConfig(
        dead_zone=0.1,
        saturation_zone=0.3,
        normalization_mode=viewer.OverlayNormalizationMode.AXIS,
    )

    viewer.HandLandmarksViewer('2D').show_2d(
        image,
        [hand],
        reference_xyz=(0.5, 0.5, 0.0),
        control_overlay=config,
    )

    assert len(calls['rectangle']) == 2
    text = [args[1] for args, _ in calls['putText']]
    assert 'CTRL axis  DZ: 0.10  SAT: 0.30' in text
    assert (
        'LM[0] SATURATED d=(0.35, 0.05, 0.00) '
        'n=(1.00, 0.00, 0.00)'
    ) in text


def test_show_3d_draws_vector_preview_and_projected_spheres(monkeypatch):
    """The 3D vector overlay should project metric spheres and preview status."""
    calls = _patch_opencv(monkeypatch)
    clock = _Clock()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    image_hand = [SimpleNamespace(x=0.5, y=0.5, z=0.0)]
    metric_hand = [SimpleNamespace(x=0.2, y=0.0, z=0.5)]
    hand_viewer = viewer.HandLandmarksViewer('3D', clock=clock)
    clock.now = 0.5

    exit_requested = hand_viewer.show_3d(
        image,
        [image_hand],
        primary_metric_hand=metric_hand,
        missing_depth_count=2,
        mediapipe_time_sec=0.01,
        reference_metric=(0.0, 0.0, 0.5),
        reference_image=(0.5, 0.5, 0.0),
        reference_initialized=True,
        tracked_landmark_index=0,
        control_overlay=viewer.ControlOverlayConfig(
            dead_zone=0.05,
            saturation_zone=0.4,
            normalization_mode=viewer.OverlayNormalizationMode.VECTOR,
        ),
        camera_intrinsics=(100.0, 100.0, 100.0, 50.0),
    )

    assert [args[2] for args, _ in calls['ellipse']] == [(10, 10), (80, 80)]
    text = [args[1] for args, _ in calls['putText']]
    assert 'FPS: 2.0' in text
    assert 'MP avg: 10.0ms  missing depth: 2' in text
    assert 'Ref3D: (0.00, 0.00, 0.50) m' in text
    assert (
        'LM[0] ACTIVE d=(0.20, 0.00, 0.00)m '
        'n=(0.43, 0.00, 0.00)'
    ) in text
    assert not exit_requested


def test_3d_axis_overlay_projects_cuboids(monkeypatch):
    """Axis mode should draw cuboid edges rather than sphere ellipses."""
    calls = _patch_opencv(monkeypatch)
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    config = viewer.ControlOverlayConfig(
        dead_zone=0.05,
        saturation_zone=0.1,
        normalization_mode=viewer.OverlayNormalizationMode.AXIS,
    )

    viewer.HandLandmarksViewer('3D').show_3d(
        image,
        [],
        reference_metric=(0.0, 0.0, 0.5),
        reference_image=(0.5, 0.5, 0.5),
        reference_initialized=True,
        control_overlay=config,
        camera_intrinsics=(100.0, 100.0, 100.0, 50.0),
    )

    assert len(calls['ellipse']) == 0
    assert len(calls['line']) == 2 * len(viewer._CUBOID_EDGES)


def test_invalid_overlay_mode_warns_and_falls_back_to_vector():
    """Unknown overlay modes should not prevent the producer from starting."""
    warnings = []

    mode = viewer.parse_overlay_normalization_mode('diagonal', warnings.append)

    assert mode is viewer.OverlayNormalizationMode.VECTOR
    assert warnings == [
        "Invalid overlay_normalization_mode 'diagonal', "
        "falling back to 'vector'."
    ]


def test_performance_overlay_updates_once_per_window(monkeypatch):
    """Viewer metrics should remain stable until the next window completes."""
    calls = _patch_opencv(monkeypatch)
    clock = _Clock()
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    hand_viewer = viewer.HandLandmarksViewer('Performance', clock=clock)

    clock.now = 0.25
    hand_viewer.show_2d(image, [], mediapipe_time_sec=0.01)
    assert calls['putText'] == []

    clock.now = 0.5
    hand_viewer.show_2d(image, [], mediapipe_time_sec=0.03)
    text = [args[1] for args, _ in calls['putText']]
    assert text[-2:] == ['FPS: 4.0', 'MP avg: 20.0ms']

    clock.now = 0.6
    hand_viewer.show_2d(image, [], mediapipe_time_sec=0.10)
    text = [args[1] for args, _ in calls['putText']]
    assert text[-2:] == ['FPS: 4.0', 'MP avg: 20.0ms']


def test_close_destroys_only_the_configured_window(monkeypatch):
    """Closing a viewer should target its configured window."""
    calls = _patch_opencv(monkeypatch)
    hand_viewer = viewer.HandLandmarksViewer('Hand Window')

    hand_viewer.close()

    assert calls['destroyWindow'] == [(('Hand Window',), {})]
