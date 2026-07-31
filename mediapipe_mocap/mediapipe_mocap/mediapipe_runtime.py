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

"""
Own MediaPipe construction, timestamps, timing, and resource lifecycle.

This module deliberately avoids importing MediaPipe or ROS. Callers inject the
MediaPipe option and landmarker types, which keeps synchronous VIDEO-mode
runtime behavior directly unit-testable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
import platform
from threading import Lock
import time
from typing import Any, Generic, Protocol, TypeVar


class _Stamp(Protocol):
    """Structural type for ROS-compatible timestamp fields."""

    sec: int
    nanosec: int


class _Header(Protocol):
    """Structural type for messages carrying a ROS-compatible stamp."""

    stamp: _Stamp


class DelegateMode(Enum):
    """User-selectable MediaPipe execution delegate policy."""

    AUTO = 'AUTO'
    CPU = 'CPU'
    GPU = 'GPU'


@dataclass(frozen=True)
class DelegateSelection:
    """Resolved delegate attempt and the platform label used for logging."""

    requested: DelegateMode
    preferred: DelegateMode
    platform_label: str


def parse_delegate_mode(
    value: str,
    warn: Callable[[str], None] | None = None,
) -> DelegateMode:
    """Parse a delegate parameter, falling back to AUTO when invalid."""
    normalized = str(value).upper()
    try:
        return DelegateMode(normalized)
    except ValueError:
        if warn is not None:
            warn(
                f"Invalid delegate '{normalized}', falling back to AUTO. "
                "Expected 'AUTO', 'CPU', or 'GPU'."
            )
        return DelegateMode.AUTO


def select_delegate(
    requested: DelegateMode,
    *,
    system_name: str | None = None,
    wsl: bool | None = None,
) -> DelegateSelection:
    """Resolve AUTO to GPU on native Linux and CPU on other platforms."""
    detected_system = system_name if system_name is not None else platform.system()
    detected_wsl = (
        wsl
        if wsl is not None
        else detected_system == 'Linux' and _is_wsl()
    )

    if detected_system == 'Linux' and detected_wsl:
        platform_label = 'WSL (Windows Subsystem for Linux)'
    elif detected_system == 'Linux':
        platform_label = 'Linux (native)'
    elif detected_system == 'Darwin':
        platform_label = 'macOS'
    else:
        platform_label = detected_system

    if requested is DelegateMode.AUTO:
        preferred = (
            DelegateMode.GPU
            if detected_system == 'Linux' and not detected_wsl
            else DelegateMode.CPU
        )
    else:
        preferred = requested

    return DelegateSelection(
        requested=requested,
        preferred=preferred,
        platform_label=platform_label,
    )


def _is_wsl() -> bool:
    """Return whether the Linux kernel version identifies a WSL environment."""
    try:
        with open('/proc/version', 'r', encoding='utf-8') as proc_version:
            version = proc_version.read().lower()
    except OSError:
        return False
    return 'microsoft' in version or 'wsl' in version


@dataclass(frozen=True)
class HandLandmarkerConfig:
    """Values shared by USB and OAK MediaPipe hand-landmarker options."""

    model_asset_path: str
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float


def create_hand_landmarker(
    config: HandLandmarkerConfig,
    *,
    delegate: Any,
    base_options_type: Callable[..., Any],
    landmarker_options_type: Callable[..., Any],
    landmarker_type: Any,
    running_mode_type: Any,
) -> Any:
    """Construct a VIDEO-mode landmarker from injected MediaPipe types."""
    options_kwargs: dict[str, Any] = {
        'base_options': base_options_type(
            model_asset_path=config.model_asset_path,
            delegate=delegate,
        ),
        'running_mode': running_mode_type.VIDEO,
        'num_hands': config.num_hands,
        'min_hand_detection_confidence': (
            config.min_hand_detection_confidence
        ),
        'min_hand_presence_confidence': config.min_hand_presence_confidence,
        'min_tracking_confidence': config.min_tracking_confidence,
    }
    options = landmarker_options_type(**options_kwargs)
    return landmarker_type.create_from_options(options)


def create_hand_landmarker_with_delegate(
    config: HandLandmarkerConfig,
    *,
    requested_delegate: DelegateMode,
    base_options_type: Callable[..., Any],
    delegate_type: Any,
    landmarker_options_type: Callable[..., Any],
    landmarker_type: Any,
    running_mode_type: Any,
    info: Callable[[str], None],
    warn: Callable[[str], None],
) -> Any:
    """Create a landmarker and apply AUTO's GPU-to-CPU fallback policy."""
    selection = select_delegate(requested_delegate)

    def create_with(delegate_mode: DelegateMode) -> Any:
        return create_hand_landmarker(
            config,
            delegate=getattr(delegate_type, delegate_mode.value),
            base_options_type=base_options_type,
            landmarker_options_type=landmarker_options_type,
            landmarker_type=landmarker_type,
            running_mode_type=running_mode_type,
        )

    try:
        landmarker = create_with(selection.preferred)
    except Exception as error:
        can_fallback = (
            selection.requested is DelegateMode.AUTO
            and selection.preferred is DelegateMode.GPU
        )
        if not can_fallback:
            raise
        warn(
            'GPU delegate initialization failed in AUTO mode; '
            f'falling back to CPU: {error}'
        )
        landmarker = create_with(DelegateMode.CPU)
        info(
            f'Platform: {selection.platform_label}. '
            'Using CPU delegate (AUTO fallback).'
        )
        return landmarker

    requested_label = (
        'AUTO'
        if selection.requested is DelegateMode.AUTO
        else 'explicit'
    )
    info(
        f'Platform: {selection.platform_label}. '
        f'Using {selection.preferred.value} delegate ({requested_label}).'
    )
    return landmarker


@dataclass
class MonotonicTimestampGenerator:
    """Generate strictly increasing integer timestamps across detection calls."""

    _last_timestamp_ms: int = field(default=-1, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def next_timestamp(self, candidate_ms: int) -> int:
        """Return ``candidate_ms`` or advance the previous value by one."""
        timestamp_ms = int(candidate_ms)
        with self._lock:
            if timestamp_ms <= self._last_timestamp_ms:
                timestamp_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = timestamp_ms
            return timestamp_ms


ResultT = TypeVar('ResultT')


@dataclass(frozen=True)
class DetectionResult(Generic[ResultT]):
    """Synchronous MediaPipe result paired with its processing duration."""

    result: ResultT
    duration_sec: float


class HandLandmarkerRuntime:
    """Own VIDEO-mode timestamp, timing, and landmarker lifecycle state."""

    def __init__(
        self,
        landmarker: Any,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create runtime state around an injected MediaPipe landmarker."""
        self._landmarker = landmarker
        self._clock = clock
        self._timestamps = MonotonicTimestampGenerator()

    def detect(
        self,
        image: Any,
        candidate_timestamp_ms: int,
    ) -> DetectionResult[Any]:
        """
        Detect synchronously using a strictly increasing MediaPipe timestamp.

        Detection errors propagate to the caller. Durations use a monotonic
        clock and are clamped to zero if an injected clock moves backwards.
        """
        timestamp_ms = self._timestamps.next_timestamp(candidate_timestamp_ms)
        started_at_sec = self._clock()
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return DetectionResult(
            result=result,
            duration_sec=max(self._clock() - started_at_sec, 0.0),
        )

    def close(self) -> None:
        """Close the injected landmarker and release its native resources."""
        self._landmarker.close()


@dataclass(frozen=True)
class PerformanceSnapshot:
    """Event rate and optional average duration for one completed window."""

    rate_hz: float
    average_duration_sec: float | None


class PeriodicPerformanceTracker:
    """Aggregate event rate and durations over fixed monotonic-clock windows."""

    def __init__(
        self,
        interval_sec: float = 0.5,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Create a tracker whose :meth:`tick` returns rates periodically."""
        if interval_sec <= 0.0:
            raise ValueError('interval_sec must be greater than zero')
        self._interval_sec = float(interval_sec)
        self._clock = clock
        self._window_start_sec = self._clock()
        self._event_count = 0
        self._duration_total_sec = 0.0
        self._duration_count = 0

    def tick(
        self,
        duration_sec: float | None = None,
    ) -> PerformanceSnapshot | None:
        """Record one event and return aggregate metrics when the window ends."""
        self._event_count += 1
        if duration_sec is not None:
            self._duration_total_sec += max(float(duration_sec), 0.0)
            self._duration_count += 1

        now_sec = self._clock()
        elapsed_sec = now_sec - self._window_start_sec
        if elapsed_sec < self._interval_sec:
            return None

        average_duration_sec = (
            self._duration_total_sec / self._duration_count
            if self._duration_count > 0
            else None
        )
        snapshot = PerformanceSnapshot(
            rate_hz=self._event_count / elapsed_sec,
            average_duration_sec=average_duration_sec,
        )
        self._window_start_sec = now_sec
        self._event_count = 0
        self._duration_total_sec = 0.0
        self._duration_count = 0
        return snapshot


def timestamp_ms_from_header(header: _Header) -> int:
    """Convert a ROS-compatible header timestamp to integer milliseconds."""
    return (
        int(header.stamp.sec) * 1000
        + int(header.stamp.nanosec) // 1_000_000
    )


def timestamp_sec_from_header(header: _Header) -> float:
    """Convert a ROS-compatible header timestamp to floating-point seconds."""
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
