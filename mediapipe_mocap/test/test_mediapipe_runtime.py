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

"""Tests for MediaPipe construction and synchronous runtime state."""

from dataclasses import dataclass
from types import SimpleNamespace

from mediapipe_mocap.mediapipe_runtime import (
    create_hand_landmarker,
    create_hand_landmarker_with_delegate,
    DelegateMode,
    HandLandmarkerConfig,
    HandLandmarkerRuntime,
    MonotonicTimestampGenerator,
    parse_delegate_mode,
    PeriodicPerformanceTracker,
    select_delegate,
    timestamp_ms_from_header,
    timestamp_sec_from_header,
)
import pytest


def test_delegate_mode_falls_back_to_auto_with_warning():
    """Invalid delegate values should use the portable AUTO policy."""
    warnings = []

    mode = parse_delegate_mode('accelerator', warnings.append)

    assert mode is DelegateMode.AUTO
    assert warnings == [
        "Invalid delegate 'ACCELERATOR', falling back to AUTO. "
        "Expected 'AUTO', 'CPU', or 'GPU'."
    ]


@pytest.mark.parametrize(
    ('system_name', 'wsl', 'expected'),
    [
        ('Linux', False, DelegateMode.GPU),
        ('Linux', True, DelegateMode.CPU),
        ('Darwin', False, DelegateMode.CPU),
        ('Windows', False, DelegateMode.CPU),
    ],
)
def test_auto_delegate_uses_platform_preference(system_name, wsl, expected):
    """AUTO should retain the existing native-Linux GPU preference."""
    selection = select_delegate(
        DelegateMode.AUTO,
        system_name=system_name,
        wsl=wsl,
    )

    assert selection.preferred is expected


def test_monotonic_timestamp_advances_duplicate_and_older_values():
    """Every emitted timestamp should be strictly greater than the last."""
    timestamps = MonotonicTimestampGenerator()

    assert timestamps.next_timestamp(100) == 100
    assert timestamps.next_timestamp(100) == 101
    assert timestamps.next_timestamp(50) == 102
    assert timestamps.next_timestamp(200) == 200


class _Clock:
    """Controllable monotonic clock used by runtime timing tests."""

    def __init__(self, initial: float = 0.0):
        """Start the clock at ``initial`` seconds."""
        self.now = initial

    def __call__(self):
        """Return the current test time."""
        return self.now


class _FakeRuntimeLandmarker:
    """Record detection calls made through HandLandmarkerRuntime."""

    def __init__(self):
        """Initialize call history and close state."""
        self.calls = []
        self.closed = False

    def detect_for_video(self, image, timestamp_ms):
        """Record and return a synchronous detection."""
        self.calls.append(('video', image, timestamp_ms))
        return f'result:{image}'

    def close(self):
        """Record resource closure."""
        self.closed = True


def test_video_runtime_owns_timestamp_timing_and_close():
    """VIDEO mode should time calls and close the injected landmarker."""
    clock = _Clock(10.0)

    class TimedLandmarker(_FakeRuntimeLandmarker):
        def detect_for_video(self, image, timestamp_ms):
            result = super().detect_for_video(image, timestamp_ms)
            clock.now += 0.25
            return result

    landmarker = TimedLandmarker()
    runtime = HandLandmarkerRuntime(
        landmarker,
        clock=clock,
    )

    clock.now = 12.0
    first = runtime.detect('first', 100)
    clock.now = 15.5
    second = runtime.detect('second', 100)
    runtime.close()

    assert first.result == 'result:first'
    assert first.duration_sec == 0.25
    assert second.result == 'result:second'
    assert second.duration_sec == 0.25
    assert landmarker.calls == [
        ('video', 'first', 100),
        ('video', 'second', 101),
    ]
    assert landmarker.closed


def test_video_runtime_propagates_detection_errors():
    """Synchronous detection errors should remain visible to the node."""
    class FailingLandmarker(_FakeRuntimeLandmarker):
        def detect_for_video(self, image, timestamp_ms):
            raise RuntimeError('detection failed')

    runtime = HandLandmarkerRuntime(FailingLandmarker())

    with pytest.raises(RuntimeError, match='detection failed'):
        runtime.detect('image', 100)


def test_periodic_performance_tracker_reports_completed_windows():
    """Rate and average duration should reset after each completed interval."""
    clock = _Clock(10.0)
    tracker = PeriodicPerformanceTracker(clock=clock)

    clock.now = 10.25
    assert tracker.tick(0.01) is None
    clock.now = 10.5
    first = tracker.tick(0.03)
    clock.now = 11.5
    second = tracker.tick()

    assert first.rate_hz == 4.0
    assert first.average_duration_sec == pytest.approx(0.02)
    assert second.rate_hz == 1.0
    assert second.average_duration_sec is None


@dataclass
class _Options:
    """Capture keyword arguments passed to a fake MediaPipe option type."""

    values: dict

    def __init__(self, **kwargs):
        """Store construction values for assertions."""
        self.values = kwargs


class _Landmarker:
    """Capture options passed to a fake MediaPipe landmarker factory."""

    @classmethod
    def create_from_options(cls, options):
        """Return the options so the test can inspect construction."""
        return options


def test_landmarker_factory_enforces_video_mode():
    """Injected construction should always select MediaPipe VIDEO mode."""
    config = HandLandmarkerConfig(
        model_asset_path='/model.task',
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.7,
    )

    video = create_hand_landmarker(
        config,
        delegate='cpu',
        base_options_type=_Options,
        landmarker_options_type=_Options,
        landmarker_type=_Landmarker,
        running_mode_type=SimpleNamespace(VIDEO='video'),
    )

    assert video.values['running_mode'] == 'video'
    assert 'result_callback' not in video.values
    assert video.values['base_options'].values == {
        'model_asset_path': '/model.task',
        'delegate': 'cpu',
    }


def test_auto_delegate_falls_back_to_cpu_after_gpu_failure(monkeypatch):
    """AUTO should retry CPU when native-Linux GPU initialization fails."""
    attempts = []
    warnings = []
    information = []

    class FailingGpuLandmarker:
        @classmethod
        def create_from_options(cls, options):
            delegate = options.values['base_options'].values['delegate']
            attempts.append(delegate)
            if delegate == 'gpu':
                raise RuntimeError('GPU unavailable')
            return options

    monkeypatch.setattr(
        'mediapipe_mocap.mediapipe_runtime.platform.system',
        lambda: 'Linux',
    )
    monkeypatch.setattr(
        'mediapipe_mocap.mediapipe_runtime._is_wsl',
        lambda: False,
    )
    config = HandLandmarkerConfig(
        model_asset_path='/model.task',
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    options = create_hand_landmarker_with_delegate(
        config,
        requested_delegate=DelegateMode.AUTO,
        base_options_type=_Options,
        delegate_type=SimpleNamespace(CPU='cpu', GPU='gpu'),
        landmarker_options_type=_Options,
        landmarker_type=FailingGpuLandmarker,
        running_mode_type=SimpleNamespace(VIDEO='video'),
        info=information.append,
        warn=warnings.append,
    )

    assert attempts == ['gpu', 'cpu']
    assert options.values['base_options'].values['delegate'] == 'cpu'
    assert warnings == [
        'GPU delegate initialization failed in AUTO mode; '
        'falling back to CPU: GPU unavailable'
    ]
    assert information == [
        'Platform: Linux (native). Using CPU delegate (AUTO fallback).'
    ]


def test_explicit_gpu_failure_is_not_hidden():
    """Explicit GPU selection should surface initialization failures."""
    class FailingLandmarker:
        @classmethod
        def create_from_options(cls, options):
            raise RuntimeError('GPU unavailable')

    config = HandLandmarkerConfig(
        model_asset_path='/model.task',
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with pytest.raises(RuntimeError, match='GPU unavailable'):
        create_hand_landmarker_with_delegate(
            config,
            requested_delegate=DelegateMode.GPU,
            base_options_type=_Options,
            delegate_type=SimpleNamespace(CPU='cpu', GPU='gpu'),
            landmarker_options_type=_Options,
            landmarker_type=FailingLandmarker,
            running_mode_type=SimpleNamespace(VIDEO='video'),
            info=lambda message: None,
            warn=lambda message: None,
        )


def test_header_timestamp_conversions():
    """ROS-compatible stamps should convert without importing ROS messages."""
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=345_678_901)
    )

    assert timestamp_ms_from_header(header) == 12_345
    assert timestamp_sec_from_header(header) == pytest.approx(12.345678901)
