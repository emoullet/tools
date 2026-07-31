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

"""Tests for shared reference state and reset semantics."""

from mediapipe_mocap.reference import (
    ReferenceState,
    ResetRequestResult,
)


def test_rising_edge_mode_requires_false_between_requests():
    """Rising-edge mode should ignore inactive and repeated true signals."""
    state = ReferenceState(
        initial_position=(0.5, 0.5, 0.0),
        cooldown_sec=0.0,
    )

    assert state.request_reset(False, 0.0) is ResetRequestResult.INACTIVE
    assert state.request_reset(True, 1.0) is ResetRequestResult.ACCEPTED
    assert state.request_reset(True, 2.0) is ResetRequestResult.INACTIVE
    assert state.request_reset(False, 3.0) is ResetRequestResult.INACTIVE
    assert state.request_reset(True, 4.0) is ResetRequestResult.ACCEPTED


def test_cooldown_rejects_early_rising_edge():
    """An edge inside the cooldown should not replace the pending request."""
    state = ReferenceState(
        initial_position=(0.5, 0.5, 0.0),
        cooldown_sec=0.5,
    )

    assert state.request_reset(True, 1.0) is ResetRequestResult.ACCEPTED
    assert state.request_reset(True, 1.1) is ResetRequestResult.INACTIVE
    assert state.request_reset(False, 1.2) is ResetRequestResult.INACTIVE
    assert state.snapshot().pending_reset

    assert state.request_reset(True, 1.3) is ResetRequestResult.COOLDOWN
    assert state.request_reset(False, 1.4) is ResetRequestResult.INACTIVE
    assert state.request_reset(True, 1.5) is ResetRequestResult.ACCEPTED


def test_pending_reset_is_consumed_by_reference_update():
    """Updating should atomically replace the reference and clear pending state."""
    state = ReferenceState(
        initial_position=(0.5, 0.5, 0.0),
        cooldown_sec=0.0,
    )

    assert state.update_reference((0.1, 0.2, 0.3)) is None
    assert state.request_reset(True, 1.0) is ResetRequestResult.ACCEPTED

    updated = state.update_reference((0.1, 0.2, 0.3))

    assert updated is not None
    assert updated.position == (0.1, 0.2, 0.3)
    assert updated.initialized
    assert not updated.pending_reset


def test_uninitialized_reference_can_be_set_automatically():
    """OAK auto-reference should initialize metric and image coordinates once."""
    state = ReferenceState(
        initial_position=(0.0, 0.0, 0.6),
        cooldown_sec=0.25,
        initially_initialized=False,
    )

    updated = state.update_reference(
        (0.1, -0.2, 0.7),
        image_position=(0.4, 0.6, 0.7),
        initialize_if_needed=True,
    )

    assert updated is not None
    assert updated.position == (0.1, -0.2, 0.7)
    assert updated.image_position == (0.4, 0.6, 0.7)
    assert updated.initialized
    assert state.update_reference(
        (0.2, -0.1, 0.8),
        image_position=(0.5, 0.5, 0.8),
        initialize_if_needed=True,
    ) is None
