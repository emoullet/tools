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
Depth sampling and RGB-camera back-projection for OAK hand landmarks.

The projector is independent of DepthAI and MediaPipe runtime types. It owns
only the last-valid depth history used by the ``reuse_last`` strategy. Callers
must serialize access if results can be processed concurrently.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from geometry_msgs.msg import Point32
from mediapipe_mocap.landmark_processing import LANDMARK_COUNT
import numpy as np


_MILLIMETERS_PER_METER = 1000.0
_METERS_PER_MILLIMETER = 0.001


class _NormalizedLandmark(Protocol):
    """Structural type for MediaPipe-compatible normalized landmarks."""

    x: float
    y: float


class MissingDepthStrategy(Enum):
    """Fallback policy for landmarks without a valid direct depth sample."""

    SKIP_FRAME = 'skip_frame'
    REUSE_LAST = 'reuse_last'
    HAND_MEDIAN = 'hand_median'


def parse_missing_depth_strategy(
    value: str,
    warn: Callable[[str], None] | None = None,
) -> MissingDepthStrategy:
    """Parse a strategy parameter, preserving the ``reuse_last`` fallback."""
    normalized = str(value).lower()
    try:
        return MissingDepthStrategy(normalized)
    except ValueError:
        if warn is not None:
            warn(
                'Invalid missing_depth_strategy '
                f"'{normalized}', falling back to 'reuse_last'."
            )
        return MissingDepthStrategy.REUSE_LAST


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics for the aligned RGB camera, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_matrix(cls, matrix: Sequence[Sequence[float]]) -> CameraIntrinsics:
        """Extract focal lengths and principal point from a 3×3 matrix."""
        return cls(
            fx=float(matrix[0][0]),
            fy=float(matrix[1][1]),
            cx=float(matrix[0][2]),
            cy=float(matrix[1][2]),
        )


@dataclass(frozen=True)
class DepthProcessingConfig:
    """Sampling limits and missing-depth behavior for one OAK stream."""

    sample_radius_px: int
    min_depth_m: float
    max_depth_m: float
    percentile: float
    missing_strategy: MissingDepthStrategy
    max_missing_landmarks: int


@dataclass(frozen=True)
class HandProjection:
    """Result of projecting one normalized hand into metric camera space."""

    metric_points: list[Point32] | None
    image_points: list[Point32] | None
    missing_depth_count: int

    @property
    def valid(self) -> bool:
        """Return whether complete point lists are available."""
        return self.metric_points is not None and self.image_points is not None


class OakDepthProjector:
    """Sample aligned depth and back-project normalized hand landmarks."""

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: DepthProcessingConfig,
        hand_slots: int,
    ) -> None:
        """Initialize immutable calibration and per-hand depth history."""
        if hand_slots < 1:
            raise ValueError('hand_slots must be at least one')
        self.intrinsics = intrinsics
        self.config = config
        self._last_depth_by_hand: list[list[float | None]] = [
            [None for _ in range(LANDMARK_COUNT)]
            for _ in range(hand_slots)
        ]

    def sample_depth_m(
        self,
        depth_mm: np.ndarray,
        u: int,
        v: int,
    ) -> float | None:
        """Return the configured percentile of valid nearby depth in meters."""
        radius = self.config.sample_radius_px
        height, width = depth_mm.shape[:2]
        x0 = max(0, u - radius)
        x1 = min(width, u + radius + 1)
        y0 = max(0, v - radius)
        y1 = min(height, v + radius + 1)
        region = depth_mm[y0:y1, x0:x1]
        if region.size == 0:
            return None

        min_mm = self.config.min_depth_m * _MILLIMETERS_PER_METER
        max_mm = self.config.max_depth_m * _MILLIMETERS_PER_METER
        valid = region[(region >= min_mm) & (region <= max_mm)]
        if valid.size == 0:
            return None
        return (
            float(np.percentile(valid, self.config.percentile))
            * _METERS_PER_MILLIMETER
        )

    def project_hand(
        self,
        hand_landmarks: Sequence[_NormalizedLandmark],
        depth_mm: np.ndarray,
        hand_index: int,
    ) -> HandProjection:
        """Project one hand or return an invalid result under its fallback policy."""
        height, width = depth_mm.shape[:2]
        sampled_depths: list[float | None] = []
        missing_indices: list[int] = []

        for index, landmark in enumerate(hand_landmarks):
            x_normalized = float(landmark.x)
            y_normalized = float(landmark.y)
            if not (
                0.0 <= x_normalized <= 1.0
                and 0.0 <= y_normalized <= 1.0
            ):
                sampled_depths.append(None)
                missing_indices.append(index)
                continue

            u = int(round(x_normalized * (width - 1)))
            v = int(round(y_normalized * (height - 1)))
            depth_m = self.sample_depth_m(depth_mm, u, v)
            if depth_m is None:
                missing_indices.append(index)
            sampled_depths.append(depth_m)

        missing_count = len(missing_indices)
        if missing_count > self.config.max_missing_landmarks:
            return HandProjection(None, None, missing_count)

        if missing_indices:
            if self.config.missing_strategy is MissingDepthStrategy.SKIP_FRAME:
                return HandProjection(None, None, missing_count)
            if not self._fill_missing_depths(
                sampled_depths,
                missing_indices,
                hand_index,
            ):
                return HandProjection(None, None, missing_count)

        metric_points: list[Point32] = []
        image_points: list[Point32] = []
        for index, landmark in enumerate(hand_landmarks):
            u = float(landmark.x) * (width - 1)
            v = float(landmark.y) * (height - 1)
            depth_m = float(sampled_depths[index])
            self._last_depth_by_hand[hand_index][index] = depth_m
            metric_point = back_project(
                u,
                v,
                depth_m,
                self.intrinsics,
            )
            metric_points.append(metric_point)
            image_points.append(
                Point32(
                    x=float(landmark.x),
                    y=float(landmark.y),
                    z=depth_m,
                )
            )

        return HandProjection(metric_points, image_points, missing_count)

    def _fill_missing_depths(
        self,
        sampled_depths: list[float | None],
        missing_indices: Sequence[int],
        hand_index: int,
    ) -> bool:
        """Fill missing samples in place and report whether all were resolved."""
        current_valid = [
            depth for depth in sampled_depths if depth is not None
        ]
        hand_median = (
            float(np.median(current_valid))
            if current_valid
            else None
        )
        for index in missing_indices:
            fallback_depth = None
            if (
                self.config.missing_strategy
                is MissingDepthStrategy.REUSE_LAST
            ):
                fallback_depth = self._last_depth_by_hand[hand_index][index]
            if fallback_depth is None and hand_median is not None:
                fallback_depth = hand_median
            if fallback_depth is None:
                return False
            sampled_depths[index] = fallback_depth
        return True


def back_project(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> Point32:
    """Back-project one RGB pixel and metric depth into camera coordinates."""
    return Point32(
        x=(float(u) - intrinsics.cx) * float(depth_m) / intrinsics.fx,
        y=(float(v) - intrinsics.cy) * float(depth_m) / intrinsics.fy,
        z=float(depth_m),
    )
