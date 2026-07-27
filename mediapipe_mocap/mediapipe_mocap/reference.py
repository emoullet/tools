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
Thread-safe reference position and reset state shared by hand trackers.

ROS callbacks request resets while MediaPipe result callbacks consume them.
All mutable state is therefore owned by :class:`ReferenceState` and guarded by
one internal lock. Callers receive immutable snapshots instead of references
to mutable internal state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Sequence


Point3D = tuple[float, float, float]


class ResetTriggerMode(Enum):
    """Boolean signal interpretation used for reference reset requests."""

    RISING_EDGE = 'rising_edge'
    TRUE_MESSAGE = 'true_message'


class ResetRequestResult(Enum):
    """Outcome of processing one boolean reset signal."""

    ACCEPTED = 'accepted'
    INACTIVE = 'inactive'
    COOLDOWN = 'cooldown'


@dataclass(frozen=True)
class ReferenceSnapshot:
    """Immutable copy of the current reference state."""

    position: Point3D
    image_position: Point3D | None
    initialized: bool
    pending_reset: bool


@dataclass
class ReferenceState:
    """Own a reference point and coordinate reset requests across callbacks."""

    initial_position: Sequence[float]
    cooldown_sec: float
    trigger_mode: ResetTriggerMode
    initially_initialized: bool = True
    initial_image_position: Sequence[float] | None = None
    _position: Point3D = field(init=False, repr=False)
    _image_position: Point3D | None = field(init=False, repr=False)
    _initialized: bool = field(init=False, repr=False)
    _pending_reset: bool = field(default=False, init=False, repr=False)
    _last_signal: bool = field(default=False, init=False, repr=False)
    _last_accepted_time_sec: float | None = field(default=None, init=False, repr=False)
    _lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        """Normalize initial values before the state is shared across threads."""
        self._position = _point3(self.initial_position)
        self._image_position = (
            _point3(self.initial_image_position)
            if self.initial_image_position is not None
            else None
        )
        self.cooldown_sec = max(0.0, float(self.cooldown_sec))
        self._initialized = bool(self.initially_initialized)

    def request_reset(self, signal: bool, now_sec: float) -> ResetRequestResult:
        """
        Process a reset signal and queue an accepted request.

        Parameters
        ----------
        signal : bool
            Current boolean reset signal.
        now_sec : float
            Current clock value in seconds. It must use the same clock domain
            for every call.

        Returns
        -------
        ResetRequestResult
            Whether the signal was inactive, rejected by the cooldown, or
            accepted and queued.

        """
        current_signal = bool(signal)
        timestamp_sec = float(now_sec)

        with self._lock:
            previous_signal = self._last_signal
            self._last_signal = current_signal

            if self.trigger_mode is ResetTriggerMode.RISING_EDGE:
                triggered = current_signal and not previous_signal
            else:
                triggered = current_signal

            if not triggered:
                return ResetRequestResult.INACTIVE

            if (
                self._last_accepted_time_sec is not None
                and timestamp_sec - self._last_accepted_time_sec < self.cooldown_sec
            ):
                return ResetRequestResult.COOLDOWN

            self._last_accepted_time_sec = timestamp_sec
            self._pending_reset = True
            return ResetRequestResult.ACCEPTED

    def update_reference(
        self,
        position: Sequence[float],
        *,
        image_position: Sequence[float] | None = None,
        initialize_if_needed: bool = False,
    ) -> ReferenceSnapshot | None:
        """
        Consume a pending reset or initialize the reference.

        Parameters
        ----------
        position : Sequence[float]
            New reference coordinates.
        image_position : Sequence[float] | None
            Optional matching image-space coordinates used by visualization.
        initialize_if_needed : bool
            Update when the reference is uninitialized, even if no explicit
            reset is pending.

        Returns
        -------
        ReferenceSnapshot | None
            The updated snapshot, or ``None`` when no update was required.

        """
        new_position = _point3(position)
        new_image_position = (
            _point3(image_position) if image_position is not None else None
        )

        with self._lock:
            if not self._pending_reset and not (
                initialize_if_needed and not self._initialized
            ):
                return None

            self._position = new_position
            if image_position is not None:
                self._image_position = new_image_position
            self._initialized = True
            self._pending_reset = False
            return self._snapshot_unlocked()

    def snapshot(self) -> ReferenceSnapshot:
        """Return a consistent immutable view of the current state."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> ReferenceSnapshot:
        """Build a snapshot while the caller owns the state lock."""
        return ReferenceSnapshot(
            position=self._position,
            image_position=self._image_position,
            initialized=self._initialized,
            pending_reset=self._pending_reset,
        )


def _point3(value: Sequence[float]) -> Point3D:
    """Convert a three-element coordinate sequence to an immutable tuple."""
    if len(value) != 3:
        raise ValueError(f'Expected three coordinates, got {len(value)}')
    return (float(value[0]), float(value[1]), float(value[2]))
