import math
import os
import platform
import sys
from typing import Iterable, Sequence

from geometry_msgs.msg import Point32


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
    """Prefer the active or workspace MediaPipe virtualenv on sys.path."""
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
    """
    Check if running on Windows Subsystem for Linux.

    Returns
    -------
    bool
        True when the Linux kernel version reports WSL, otherwise False.

    """
    try:
        with open('/proc/version', 'r') as f:
            proc_version = f.read().lower()
            return 'microsoft' in proc_version or 'wsl' in proc_version
    except Exception:
        return False


def get_best_mediapipe_delegate(mp_module, logger):
    """
    Choose the fastest MediaPipe delegate known to work on this platform.

    Parameters
    ----------
    mp_module : module
        Imported ``mediapipe`` module. It must expose
        ``tasks.BaseOptions.Delegate`` so the selected delegate enum can be
        returned directly to MediaPipe task options.
    logger : object
        ROS-style logger with an ``info`` method. The function logs the
        detected platform label and whether CPU or GPU execution was selected.

    Returns
    -------
    object
        MediaPipe ``BaseOptions.Delegate`` enum value to pass to task options.

    """
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
    """
    Convert a ROS message header timestamp to floating-point seconds.

    Parameters
    ----------
    header : std_msgs.msg.Header
        ROS message header whose ``stamp.sec`` and ``stamp.nanosec`` fields
        contain the source timestamp.

    Returns
    -------
    float
        Timestamp in seconds.

    """
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def ensure_3_tuple(values: Iterable[float], fallback, logger=None, parameter_name='value'):
    """
    Return three float values, warning and using fallback when incomplete.

    Parameters
    ----------
    values : Iterable[float]
        Parameter or config value expected to provide at least three entries
        ordered as ``x, y, z``. Extra entries are ignored.
    fallback : Sequence[float]
        Three fallback values used when ``values`` contains fewer than three
        entries.
    logger : object, optional
        ROS-style logger with a ``warning`` method. When provided, invalid
        input is reported before falling back.
    parameter_name : str
        Human-readable parameter name included in the warning text.

    Returns
    -------
    tuple[float, float, float]
        Tuple containing exactly three float values.

    """
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
    """
    Scale one axis by saturation and clamp it into [-1, 1].

    Parameters
    ----------
    value : float
        Reference-relative displacement for one coordinate axis. It is assumed
        to use the same unit as ``saturation``.
    saturation : float
        Positive displacement magnitude that maps to ``+1`` or ``-1``. Very
        small values are clamped internally to avoid division by zero.

    Returns
    -------
    float
        Normalized axis value in [-1, 1].

    """
    saturation = max(float(saturation), 1e-9)
    return clamp(float(value) / saturation, -1.0, 1.0)


def saturate_vector_norm(point: Point32, saturation: float) -> Point32:
    """
    Scale a point by saturation and limit its vector norm to 1.

    Parameters
    ----------
    point : geometry_msgs.msg.Point32
        Reference-relative displacement to normalize. The x/y/z values are
        interpreted as a single 3D vector.
    saturation : float
        Positive vector magnitude that maps to norm ``1``. Very small values
        are clamped internally to avoid division by zero.

    Returns
    -------
    Point32
        Point32 with normalized x/y/z components and vector norm limited to
        ``1``.

    """
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
    """
    Return landmarks relative to reference_xyz with per-axis scaling.

    Parameters
    ----------
    hand_landmarks : Sequence[geometry_msgs.msg.Point32]
        Input landmarks in a shared coordinate frame, such as normalized image
        coordinates or metric camera coordinates.
    reference_xyz : Sequence[float]
        Reference position ordered as ``x, y, z`` in the same coordinate frame
        as ``hand_landmarks``. It is subtracted from each landmark.
    scale_xyz : Sequence[float]
        Per-axis scale ordered as ``x, y, z`` and applied after subtracting
        ``reference_xyz``.

    Returns
    -------
    list[Point32]
        Reference-relative landmarks after per-axis scaling.

    """
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
    saturation_zone: float,
    mode: str = 'axis',
):
    """
    Map relative landmarks into normalized control space.

    Axis mode clamps each coordinate independently. Vector mode preserves
    direction and limits the whole vector norm.

    Parameters
    ----------
    points : Sequence[geometry_msgs.msg.Point32]
        Reference-relative input points to map into control space.
    saturation_zone : float
        Scalar displacement that maps to the saturated limit. In ``axis`` mode
        it is applied to each coordinate independently; in ``vector`` mode it
        is applied to the full 3D vector norm.
    mode : str
        Normalization strategy. ``'axis'`` clips each coordinate independently,
        while ``'vector'`` preserves vector direction and clips only the norm.

    Returns
    -------
    list[Point32]
        Normalized control inputs with values constrained by the selected mode.

    """
    mode = str(mode).lower()
    if mode == 'vector':
        return [saturate_vector_norm(pt, saturation_zone) for pt in points]

    return [
        Point32(
            x=saturate_axis(pt.x, saturation_zone),
            y=saturate_axis(pt.y, saturation_zone),
            z=saturate_axis(pt.z, saturation_zone),
        )
        for pt in points
    ]


def reset_filter_bank(filters):
    """
    Reset all filters in a nested per-hand filter bank.

    Parameters
    ----------
    filters : Iterable[Iterable[object]]
        Nested filter bank arranged as hands, then per-landmark coordinate
        filters. Each filter instance must provide a ``reset`` method.

    """
    for hand_filters in filters:
        for filter_instance in hand_filters:
            filter_instance.reset()


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
