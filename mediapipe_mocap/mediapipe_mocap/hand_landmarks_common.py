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

import os
import platform
import sys
from typing import Iterable

from signal_processing import OneEuroFilter as SignalProcessingOneEuroFilter


OneEuroFilter = SignalProcessingOneEuroFilter


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
