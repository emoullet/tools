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

"""
Own MediaPipe construction, timestamps, and asynchronous result contexts.

This module deliberately avoids importing MediaPipe or ROS. Callers inject the
MediaPipe option and landmarker types, which keeps the runtime state directly
unit-testable. Timestamp generators and context stores own their locks and may
be shared by submission and MediaPipe callback threads.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Generic, Protocol, TypeVar


IMAGE_ASYNC_CONTEXT_LIMIT = 120
OAK_ASYNC_CONTEXT_LIMIT = 8


class _Stamp(Protocol):
    """Structural type for ROS-compatible timestamp fields."""

    sec: int
    nanosec: int


class _Header(Protocol):
    """Structural type for messages carrying a ROS-compatible stamp."""

    stamp: _Stamp


class RuntimeMode(Enum):
    """MediaPipe hand-landmarker running modes supported by the nodes."""

    VIDEO = 'VIDEO'
    LIVE_STREAM = 'LIVE_STREAM'


def parse_running_mode(
    value: str,
    warn: Callable[[str], None] | None = None,
) -> RuntimeMode:
    """Parse a mode parameter, preserving the existing VIDEO fallback."""
    normalized = str(value).upper()
    try:
        return RuntimeMode(normalized)
    except ValueError:
        if warn is not None:
            warn(
                f"Invalid running_mode '{normalized}', falling back to VIDEO. "
                "Expected 'VIDEO' or 'LIVE_STREAM'."
            )
        return RuntimeMode.VIDEO


@dataclass(frozen=True)
class HandLandmarkerConfig:
    """Values shared by USB and OAK MediaPipe hand-landmarker options."""

    model_asset_path: str
    running_mode: RuntimeMode
    num_hands: int
    min_hand_detection_confidence: float
    min_hand_presence_confidence: float
    min_tracking_confidence: float


def create_hand_landmarker(
    config: HandLandmarkerConfig,
    *,
    delegate: Any,
    result_callback: Callable[[Any, Any, int], None] | None,
    base_options_type: Callable[..., Any],
    landmarker_options_type: Callable[..., Any],
    landmarker_type: Any,
    running_mode_type: Any,
) -> Any:
    """Construct a landmarker from injected MediaPipe runtime types."""
    running_mode = (
        running_mode_type.LIVE_STREAM
        if config.running_mode is RuntimeMode.LIVE_STREAM
        else running_mode_type.VIDEO
    )
    options_kwargs: dict[str, Any] = {
        'base_options': base_options_type(
            model_asset_path=config.model_asset_path,
            delegate=delegate,
        ),
        'running_mode': running_mode,
        'num_hands': config.num_hands,
        'min_hand_detection_confidence': (
            config.min_hand_detection_confidence
        ),
        'min_hand_presence_confidence': config.min_hand_presence_confidence,
        'min_tracking_confidence': config.min_tracking_confidence,
    }
    if config.running_mode is RuntimeMode.LIVE_STREAM:
        if result_callback is None:
            raise ValueError('LIVE_STREAM mode requires a result callback')
        options_kwargs['result_callback'] = result_callback

    options = landmarker_options_type(**options_kwargs)
    return landmarker_type.create_from_options(options)


@dataclass
class MonotonicTimestampGenerator:
    """Generate strictly increasing integer timestamps across callback threads."""

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


ContextT = TypeVar('ContextT')


class AsyncContextStore(Generic[ContextT]):
    """Own a bounded timestamp-to-context map shared across callback threads."""

    def __init__(self, max_pending: int) -> None:
        """Create a store that retains at most ``max_pending`` contexts."""
        if max_pending < 1:
            raise ValueError('max_pending must be at least one')
        self._max_pending = int(max_pending)
        self._contexts: dict[int, ContextT] = {}
        self._lock = Lock()

    def put(self, timestamp_ms: int, context: ContextT) -> None:
        """Store a context and evict the oldest pending entries if necessary."""
        with self._lock:
            self._contexts[int(timestamp_ms)] = context
            while len(self._contexts) > self._max_pending:
                oldest_timestamp = next(iter(self._contexts))
                self._contexts.pop(oldest_timestamp, None)

    def pop(self, timestamp_ms: int) -> ContextT | None:
        """Atomically remove and return one completed context."""
        with self._lock:
            return self._contexts.pop(int(timestamp_ms), None)

    def discard(self, timestamp_ms: int) -> None:
        """Remove a failed submission context when it is still pending."""
        with self._lock:
            self._contexts.pop(int(timestamp_ms), None)

    @property
    def pending_count(self) -> int:
        """Return the number of contexts currently awaiting results."""
        with self._lock:
            return len(self._contexts)


def timestamp_ms_from_header(header: _Header) -> int:
    """Convert a ROS-compatible header timestamp to integer milliseconds."""
    return (
        int(header.stamp.sec) * 1000
        + int(header.stamp.nanosec) // 1_000_000
    )


def timestamp_sec_from_header(header: _Header) -> float:
    """Convert a ROS-compatible header timestamp to floating-point seconds."""
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9
