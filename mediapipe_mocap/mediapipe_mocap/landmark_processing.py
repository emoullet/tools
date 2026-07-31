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
Coordinate transforms and stateful filtering for hand landmarks.

The functions in this module are stateless. ``LandmarkFilterBank`` owns the
mutable One Euro filter state for a fixed number of MediaPipe hand slots.
Instances are callback-owned and are not safe to share across threads without
external synchronization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from geometry_msgs.msg import Point32
from signal_processing import OneEuroFilter


LANDMARK_COUNT = 21
"""Number of landmarks in one MediaPipe hand result."""

_AXIS_COUNT = 3


@dataclass(frozen=True)
class OneEuroFilterConfig:
    """Validated One Euro tuning values, expressed in seconds and hertz."""

    frequency: float
    min_cutoff: float
    beta: float
    derivative_cutoff: float = 1.0

    def __post_init__(self) -> None:
        """Reject values that cannot produce a stable filter."""
        if self.frequency <= 0.0:
            raise ValueError('frequency must be greater than zero')
        if self.min_cutoff <= 0.0:
            raise ValueError('min_cutoff must be greater than zero')
        if self.beta < 0.0:
            raise ValueError('beta must be non-negative')
        if self.derivative_cutoff <= 0.0:
            raise ValueError('derivative_cutoff must be greater than zero')


class LandmarkFilterBank:
    """Own independent scalar filters for every hand, landmark, and axis."""

    def __init__(self, hand_slots: int, config: OneEuroFilterConfig) -> None:
        """Allocate filters for ``hand_slots`` MediaPipe result positions."""
        if hand_slots < 1:
            raise ValueError('hand_slots must be at least one')

        self._filters = [
            [
                OneEuroFilter(
                    config.frequency,
                    config.min_cutoff,
                    config.beta,
                    config.derivative_cutoff,
                )
                for _ in range(LANDMARK_COUNT * _AXIS_COUNT)
            ]
            for _ in range(hand_slots)
        ]

    def filter_point(
        self,
        hand_index: int,
        landmark_index: int,
        point: Point32,
        timestamp_sec: float,
    ) -> Point32:
        """Filter one point using its stable hand and landmark slots."""
        if not 0 <= landmark_index < LANDMARK_COUNT:
            raise IndexError(
                f'landmark_index must be in [0, {LANDMARK_COUNT - 1}]'
            )

        hand_filters = self._filters[hand_index]
        base_index = landmark_index * _AXIS_COUNT
        return Point32(
            x=hand_filters[base_index].filter(
                float(point.x), timestamp_sec
            ),
            y=hand_filters[base_index + 1].filter(
                float(point.y), timestamp_sec
            ),
            z=hand_filters[base_index + 2].filter(
                float(point.z), timestamp_sec
            ),
        )

    def reset(self) -> None:
        """Clear all history so the next sample warm-starts each filter."""
        for hand_filters in self._filters:
            for filter_instance in hand_filters:
                filter_instance.reset()


def relative_points(
    hand_landmarks: Sequence[Point32],
    reference_xyz: Sequence[float],
    scale_xyz: Sequence[float] = (1.0, 1.0, 1.0),
) -> list[Point32]:
    """Subtract a reference and apply per-axis scaling to each landmark."""
    ref_x, ref_y, ref_z = reference_xyz
    scale_x, scale_y, scale_z = scale_xyz
    return [
        Point32(
            x=(float(landmark.x) - ref_x) * scale_x,
            y=(float(landmark.y) - ref_y) * scale_y,
            z=(float(landmark.z) - ref_z) * scale_z,
        )
        for landmark in hand_landmarks
    ]
