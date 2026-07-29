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

"""Tests for hand-landmark coordinate transforms and filtering."""

from geometry_msgs.msg import Point32
from mediapipe_mocap.landmark_processing import (
    LandmarkFilterBank,
    OneEuroFilterConfig,
    relative_points,
)
import pytest


def test_relative_points_subtracts_reference_and_scales_axes():
    """Relative conversion should retain its existing per-axis behavior."""
    points = [Point32(x=0.6, y=0.2, z=0.8)]

    relative = relative_points(
        points,
        reference_xyz=(0.5, 0.4, 0.3),
        scale_xyz=(2.0, 3.0, 0.0),
    )

    assert relative[0].x == pytest.approx(0.2)
    assert relative[0].y == pytest.approx(-0.6)
    assert relative[0].z == pytest.approx(0.0)


def test_filter_bank_uses_independent_hand_and_landmark_state():
    """Samples in one slot must not warm-start another slot."""
    filters = LandmarkFilterBank(
        hand_slots=2,
        config=OneEuroFilterConfig(30.0, 1.0, 0.1),
    )

    filters.filter_point(0, 0, Point32(x=0.0, y=0.0, z=0.0), 1.0)
    filtered = filters.filter_point(
        0, 0, Point32(x=1.0, y=1.0, z=1.0), 2.0
    )
    other_hand = filters.filter_point(
        1, 0, Point32(x=1.0, y=1.0, z=1.0), 2.0
    )
    other_landmark = filters.filter_point(
        0, 1, Point32(x=1.0, y=1.0, z=1.0), 2.0
    )

    assert filtered.x < 1.0
    assert other_hand.x == pytest.approx(1.0)
    assert other_landmark.x == pytest.approx(1.0)


def test_filter_bank_reset_warm_starts_after_lost_hand():
    """Reset should prevent old motion history leaking into reacquisition."""
    filters = LandmarkFilterBank(
        hand_slots=1,
        config=OneEuroFilterConfig(30.0, 1.0, 0.1),
    )
    filters.filter_point(0, 0, Point32(x=0.0, y=0.0, z=0.0), 1.0)
    filters.filter_point(0, 0, Point32(x=1.0, y=1.0, z=1.0), 2.0)

    filters.reset()
    reacquired = filters.filter_point(
        0, 0, Point32(x=0.25, y=0.5, z=0.75), 3.0
    )

    assert reacquired.x == pytest.approx(0.25)
    assert reacquired.y == pytest.approx(0.5)
    assert reacquired.z == pytest.approx(0.75)
