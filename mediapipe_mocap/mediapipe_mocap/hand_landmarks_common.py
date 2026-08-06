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

import os
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
        if os.path.isdir(site_packages):
            if site_packages not in sys.path:
                sys.path.insert(0, site_packages)
            return


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
