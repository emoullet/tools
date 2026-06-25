import math
import os
import platform
import sys
from typing import Iterable, Sequence

from geometry_msgs.msg import Point32
import numpy as np


# MediaPipe hand landmark graph edges (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def prepare_runtime_imports():
    """Prepare optional GPU and virtual-env paths before heavy CV imports."""
    _force_nvidia_prime_render_offload()
    _add_venv_site_packages_to_path()


def _force_nvidia_prime_render_offload():
    """Request NVIDIA PRIME render offload before OpenGL users are imported."""
    if sys.platform.startswith('linux') and (
        os.path.exists('/dev/nvidiactl') or os.path.isdir('/proc/driver/nvidia')
    ):
        os.environ['__NV_PRIME_RENDER_OFFLOAD'] = '1'
        os.environ['__GLX_VENDOR_LIBRARY_NAME'] = 'nvidia'


def _add_venv_site_packages_to_path():
    candidate_roots = []

    venv_root = os.environ.get('VIRTUAL_ENV')
    if venv_root:
        candidate_roots.append(venv_root)

    current_dir = os.getcwd()
    for _ in range(4):
        candidate = os.path.join(current_dir, '.venv_mediapipe')
        if os.path.isdir(candidate):
            candidate_roots.append(candidate)
            break
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    for root in candidate_roots:
        site_packages = os.path.join(
            root,
            'lib',
            f'python{sys.version_info.major}.{sys.version_info.minor}',
            'site-packages',
        )
        if os.path.isdir(site_packages) and site_packages not in sys.path:
            sys.path.insert(0, site_packages)
            return


def is_wsl() -> bool:
    """Check if running on Windows Subsystem for Linux."""
    try:
        with open('/proc/version', 'r') as f:
            proc_version = f.read().lower()
            return 'microsoft' in proc_version or 'wsl' in proc_version
    except Exception:
        return False


def get_best_mediapipe_delegate(mp_module, logger):
    """Choose the fastest MediaPipe delegate known to work on this platform."""
    system = platform.system()
    delegate = mp_module.tasks.BaseOptions.Delegate.CPU
    delegate_name = 'CPU'
    if system == 'Linux':
        if not is_wsl():
            system = 'Linux (native)'
            delegate = mp_module.tasks.BaseOptions.Delegate.GPU
            delegate_name = 'GPU'
        else:
            system = 'WSL (Windows Subsystem for Linux)'
    elif system == 'Darwin':
        system = 'macOS'
    logger.info(f'Platform: {system}. Using {delegate_name} delegate.')
    return delegate


def timestamp_sec_from_header(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def ensure_3_tuple(values: Iterable[float], fallback, logger=None, parameter_name='value'):
    values = list(values)
    if len(values) < 3:
        if logger is not None:
            logger.warning(
                f"Parameter '{parameter_name}' must contain at least 3 values (x, y, z). "
                f'Falling back to [{fallback[0]}, {fallback[1]}, {fallback[2]}].'
            )
        values = fallback
    return (float(values[0]), float(values[1]), float(values[2]))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def saturate_axis(value: float, saturation: float) -> float:
    saturation = max(float(saturation), 1e-9)
    return clamp(float(value) / saturation, -1.0, 1.0)


def saturate_vector_norm(point: Point32, saturation: float) -> Point32:
    saturation = max(float(saturation), 1e-9)
    norm = math.sqrt(
        float(point.x) * float(point.x)
        + float(point.y) * float(point.y)
        + float(point.z) * float(point.z)
    )
    if norm <= 1e-12:
        return Point32(x=0.0, y=0.0, z=0.0)

    scale = min(norm / saturation, 1.0) / norm
    return Point32(
        x=float(point.x) * scale,
        y=float(point.y) * scale,
        z=float(point.z) * scale,
    )


def relative_points(
    hand_landmarks: Sequence[Point32],
    reference_xyz,
    scale_xyz=(1.0, 1.0, 1.0),
):
    ref_x, ref_y, ref_z = reference_xyz
    scale_x, scale_y, scale_z = scale_xyz
    return [
        Point32(
            x=(float(lm.x) - ref_x) * scale_x,
            y=(float(lm.y) - ref_y) * scale_y,
            z=(float(lm.z) - ref_z) * scale_z,
        )
        for lm in hand_landmarks
    ]


def normalized_control_points(
    points: Sequence[Point32],
    saturation_xyz,
    mode: str = 'axis',
):
    mode = str(mode).lower()
    if mode == 'vector':
        # Use one scalar radius for the whole vector; the first component is the
        # canonical saturation when a per-axis tuple was supplied.
        return [saturate_vector_norm(pt, saturation_xyz[0]) for pt in points]

    sat_x, sat_y, sat_z = saturation_xyz
    return [
        Point32(
            x=saturate_axis(pt.x, sat_x),
            y=saturate_axis(pt.y, sat_y),
            z=saturate_axis(pt.z, sat_z),
        )
        for pt in points
    ]


def reset_filter_bank(filters):
    for hand_filters in filters:
        for filter_instance in hand_filters:
            filter_instance.reset()


def draw_hand_on_image(image: np.ndarray, hand_landmarks):
    """Draw one hand's landmarks and connections using normalized image coords."""
    import cv2

    height, width = image.shape[:2]

    points_px = []
    for lm in hand_landmarks:
        x = int(float(lm.x) * width)
        y = int(float(lm.y) * height)
        points_px.append((x, y))
        cv2.circle(image, (x, y), 3, (0, 255, 0), -1)

    for start_idx, end_idx in HAND_CONNECTIONS:
        if start_idx < len(points_px) and end_idx < len(points_px):
            cv2.line(image, points_px[start_idx], points_px[end_idx], (0, 200, 255), 2)


def draw_reference_overlay(
    image: np.ndarray,
    reference_xyz,
    hand_landmarks=None,
    tracked_landmark_index: int = 0,
    show_control_zones: bool = True,
    dead_zone: float = 0.0,
    saturation_xyz=(0.3, 0.3, 0.3),
    label='SAT_XYZ',
):
    import cv2

    if reference_xyz is None:
        return

    x_norm, y_norm, z_norm = reference_xyz
    height, width = image.shape[:2]
    x_px = int(np.clip(x_norm, 0.0, 1.0) * width)
    y_px = int(np.clip(y_norm, 0.0, 1.0) * height)

    cv2.drawMarker(
        image,
        (x_px, y_px),
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=16,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f'Ref: ({x_norm:.2f}, {y_norm:.2f}, {z_norm:.2f})',
        (10, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 255),
        2,
    )

    sat_x, sat_y, sat_z = saturation_xyz
    if show_control_zones:
        dead_radius_px = max(1, int(float(dead_zone) * min(width, height)))
        cv2.circle(image, (x_px, y_px), dead_radius_px, (0, 255, 255), 2, cv2.LINE_AA)

        sat_radius_px = max(1, int(float(sat_x) * min(width, height)))
        cv2.circle(image, (x_px, y_px), sat_radius_px, (255, 128, 0), 2, cv2.LINE_AA)

        cv2.putText(
            image,
            f'DZ: {dead_zone:.2f}  {label}: ({sat_x:.2f}, {sat_y:.2f}, {sat_z:.2f})',
            (10, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    if hand_landmarks and 0 <= tracked_landmark_index < len(hand_landmarks):
        lm = hand_landmarks[tracked_landmark_index]
        lm_x = int(np.clip(float(lm.x), 0.0, 1.0) * width)
        lm_y = int(np.clip(float(lm.y), 0.0, 1.0) * height)
        dx = float(lm.x) - x_norm
        dy = float(lm.y) - y_norm
        dz = float(lm.z) - z_norm

        in_dead_zone = (dx * dx + dy * dy + dz * dz) ** 0.5 < float(dead_zone)
        x_sat = abs(dx) >= float(sat_x)
        y_sat = abs(dy) >= float(sat_y)
        z_sat = abs(dz) >= float(sat_z)

        cv2.circle(image, (lm_x, lm_y), 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.line(image, (x_px, y_px), (lm_x, lm_y), (255, 0, 255), 1, cv2.LINE_AA)

        status = 'DEAD' if in_dead_zone else 'ACTIVE'
        sat_flags = f'SAT[x:{int(x_sat)} y:{int(y_sat)} z:{int(z_sat)}]'
        cv2.putText(
            image,
            f'LM[{tracked_landmark_index}] {status} {sat_flags}',
            (10, 178),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 255),
            2,
        )


class OneEuroFilter:
    """Simple scalar One Euro filter."""

    def __init__(self, frequency: float = 30.0, mincutoff: float = 1.0, beta: float = 0.1):
        self.frequency = float(frequency)
        self.mincutoff = float(mincutoff)
        self.beta = float(beta)
        self.last_value = 0.0
        self.last_derivative = 0.0
        self.last_timestamp = -1.0

    def reset(self):
        self.last_value = 0.0
        self.last_derivative = 0.0
        self.last_timestamp = -1.0

    def filter(self, value: float, timestamp_sec: float) -> float:  # noqa: A003
        if self.last_timestamp < 0.0:
            self.last_value = float(value)
            self.last_derivative = 0.0
            self.last_timestamp = float(timestamp_sec)
            return float(value)

        dt = float(timestamp_sec) - self.last_timestamp
        if dt <= 0.0:
            return self.last_value

        derivative = (float(value) - self.last_value) / dt
        cutoff = self.mincutoff + self.beta * abs(derivative)
        alpha = cutoff / (cutoff + self.frequency)

        filtered_value = alpha * float(value) + (1.0 - alpha) * self.last_value
        filtered_derivative = alpha * derivative + (1.0 - alpha) * self.last_derivative

        self.last_value = filtered_value
        self.last_derivative = filtered_derivative
        self.last_timestamp = float(timestamp_sec)
        return filtered_value
