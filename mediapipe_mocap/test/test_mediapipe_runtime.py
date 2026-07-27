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

"""Tests for MediaPipe construction and asynchronous runtime state."""

from dataclasses import dataclass
from types import SimpleNamespace

from mediapipe_mocap.mediapipe_runtime import (
    AsyncContextStore,
    create_hand_landmarker,
    HandLandmarkerConfig,
    MonotonicTimestampGenerator,
    parse_running_mode,
    RuntimeMode,
    timestamp_ms_from_header,
    timestamp_sec_from_header,
)
import pytest


def test_running_mode_falls_back_to_video_with_warning():
    """Invalid parameter values should preserve the current fallback."""
    warnings = []

    mode = parse_running_mode('not-a-mode', warnings.append)

    assert mode is RuntimeMode.VIDEO
    assert warnings == [
        "Invalid running_mode 'NOT-A-MODE', falling back to VIDEO. "
        "Expected 'VIDEO' or 'LIVE_STREAM'."
    ]


def test_monotonic_timestamp_advances_duplicate_and_older_values():
    """Every emitted timestamp should be strictly greater than the last."""
    timestamps = MonotonicTimestampGenerator()

    assert timestamps.next_timestamp(100) == 100
    assert timestamps.next_timestamp(100) == 101
    assert timestamps.next_timestamp(50) == 102
    assert timestamps.next_timestamp(200) == 200


def test_async_context_store_evicts_oldest_context():
    """Bounded storage should retain only the newest pending submissions."""
    contexts = AsyncContextStore[str](max_pending=2)

    contexts.put(10, 'oldest')
    contexts.put(20, 'middle')
    contexts.put(30, 'newest')

    assert contexts.pop(10) is None
    assert contexts.pop(20) == 'middle'
    assert contexts.pop(30) == 'newest'
    assert contexts.pending_count == 0


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


def test_landmarker_factory_adds_callback_only_for_live_stream():
    """Injected construction should map runtime modes to MediaPipe modes."""
    def callback(result, image, timestamp):
        pass

    running_modes = SimpleNamespace(VIDEO='video', LIVE_STREAM='live')
    common = {
        'model_asset_path': '/model.task',
        'num_hands': 1,
        'min_hand_detection_confidence': 0.5,
        'min_hand_presence_confidence': 0.6,
        'min_tracking_confidence': 0.7,
    }

    video = create_hand_landmarker(
        HandLandmarkerConfig(running_mode=RuntimeMode.VIDEO, **common),
        delegate='cpu',
        result_callback=None,
        base_options_type=_Options,
        landmarker_options_type=_Options,
        landmarker_type=_Landmarker,
        running_mode_type=running_modes,
    )
    live = create_hand_landmarker(
        HandLandmarkerConfig(running_mode=RuntimeMode.LIVE_STREAM, **common),
        delegate='gpu',
        result_callback=callback,
        base_options_type=_Options,
        landmarker_options_type=_Options,
        landmarker_type=_Landmarker,
        running_mode_type=running_modes,
    )

    assert video.values['running_mode'] == 'video'
    assert 'result_callback' not in video.values
    assert live.values['running_mode'] == 'live'
    assert live.values['result_callback'] is callback
    assert live.values['base_options'].values == {
        'model_asset_path': '/model.task',
        'delegate': 'gpu',
    }


def test_header_timestamp_conversions():
    """ROS-compatible stamps should convert without importing ROS messages."""
    header = SimpleNamespace(
        stamp=SimpleNamespace(sec=12, nanosec=345_678_901)
    )

    assert timestamp_ms_from_header(header) == 12_345
    assert timestamp_sec_from_header(header) == pytest.approx(12.345678901)
