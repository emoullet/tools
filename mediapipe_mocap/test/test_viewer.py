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
            'line',
            'putText',
            'drawMarker',
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
    """The 2D renderer should draw a complete hand and report a quit key."""
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
        dead_zone=0.05,
        saturation_zone=0.3,
    )

    landmark_circles = [
        args for args, _ in calls['circle']
        if len(args) == 5 and args[2] == 3
    ]
    assert len(landmark_circles) == 21
    assert landmark_circles[0][1] == (50, 50)
    assert len(calls['line']) == len(viewer.HAND_CONNECTIONS) + 1
    assert calls['drawMarker'][0][0][1] == (100, 50)
    assert calls['imshow'][0][0][0] == '2D'
    assert calls['imshow'][0][0][1] is not image
    assert exit_requested


def test_show_3d_draws_metric_status_and_projected_zone(monkeypatch):
    """The 3D renderer should project its metric saturation zone and status."""
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
        dead_zone=0.05,
        saturation_zone=0.4,
        focal_length_px=100.0,
    )

    assert any(args[2] == 80 for args, _ in calls['circle'])
    text = [args[1] for args, _ in calls['putText']]
    assert 'FPS: 2.0' in text
    assert 'MP avg: 10.0ms  missing depth: 2' in text
    assert 'Ref3D: (0.00, 0.00, 0.50) m' in text
    assert 'LM[0] ACTIVE d=(0.20, 0.00, 0.00) m' in text
    assert not exit_requested


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
