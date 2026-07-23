"""QoS profiles used by the low-latency hand tracking pipeline."""

from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)

def latest_reliable_qos() -> QoSProfile:
    """Return a reliable, one-sample profile for high-rate images, landmarks and commands."""
    return QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )