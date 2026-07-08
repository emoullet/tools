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

    normalized = normalized_control_points(points, (0.3, 0.1, 0.1), mode='axis')

    assert normalized[0].x == pytest.approx(1.0)
    assert normalized[0].y == pytest.approx(-1.0)
    assert normalized[0].z == pytest.approx(0.3)


def test_normalized_control_points_vector_mode():
    """Vector mode should preserve direction and limit the normalized norm."""
    points = [
        Point32(x=0.3, y=0.4, z=0.0),
        Point32(x=0.1, y=0.0, z=0.0),
    ]

    normalized = normalized_control_points(points, (0.5, 0.5, 0.5), mode='vector')

    assert normalized[0].x == pytest.approx(0.6)
    assert normalized[0].y == pytest.approx(0.8)
    assert normalized[0].z == pytest.approx(0.0)
    assert normalized[1].x == pytest.approx(0.2)
    assert normalized[1].y == pytest.approx(0.0)
    assert normalized[1].z == pytest.approx(0.0)
