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

"""Tests for shared hand landmark signal helpers."""

from geometry_msgs.msg import Point32
from mediapipe_mocap.hand_landmarks_common import (
    normalized_control_points,
    OneEuroFilter,
)
import pytest
from signal_processing import OneEuroFilter as SignalProcessingOneEuroFilter


def test_one_euro_filter_uses_signal_processing_package():
    """Canonical filter should re-export the signal_processing implementation."""
    assert OneEuroFilter is SignalProcessingOneEuroFilter


def test_normalized_control_points_axis_mode():
    """Axis mode should clamp each axis independently to [-1, 1]."""
    points = [Point32(x=0.6, y=-0.2, z=0.03)]

    normalized = normalized_control_points(points, 0.3, mode='axis')

    assert normalized[0].x == pytest.approx(1.0)
    assert normalized[0].y == pytest.approx(-2.0 / 3.0)
    assert normalized[0].z == pytest.approx(0.1)


def test_normalized_control_points_vector_mode():
    """Vector mode should preserve direction and limit the normalized norm."""
    points = [
        Point32(x=0.3, y=0.4, z=0.0),
        Point32(x=0.1, y=0.0, z=0.0),
    ]

    normalized = normalized_control_points(points, 0.5, mode='vector')

    assert normalized[0].x == pytest.approx(0.6)
    assert normalized[0].y == pytest.approx(0.8)
    assert normalized[0].z == pytest.approx(0.0)
    assert normalized[1].x == pytest.approx(0.2)
    assert normalized[1].y == pytest.approx(0.0)
    assert normalized[1].z == pytest.approx(0.0)
