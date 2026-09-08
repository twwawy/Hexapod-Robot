"""Scale policy residuals without replacing the archived firmware gait.

Imported only after the isolated v3 controller. The original step owns phase,
contact handling, foot memory, posture, Cartesian residual semantics and IK.
No geometric landing target or odom stance anchor enters that controller.
"""
from collections import namedtuple

import jax.numpy as jp
import firmware_mjx_controller as firmware

_firmware_step = firmware.step
ResidualState = namedtuple('ResidualState', firmware.FirmwareState._fields + (
    'terrain_features', 'observation_fraction', 'residual_scale',
    'residual_gain', 'applied_action',
))


def extend_state(base):
    return ResidualState(*base, jp.zeros(15), jp.zeros(()), jp.ones(()),
                         jp.zeros(()), jp.zeros(18))


def step(state, *, policy_action, **kwargs):
    """Run the original nominal + residual controller, with a smooth gain."""
    coverage = jp.clip(state.observation_fraction, 0., 1.)
    target = jp.clip(state.residual_scale, 0., 1.) * coverage
    # Gain grows continuously with sensor availability; it never changes gait
    # phase or switches a foot into an externally prescribed trajectory.
    alpha = jp.exp(-firmware.FIRMWARE_CONTROL_DT / .5)
    gain = alpha * state.residual_gain + (1.-alpha) * target
    enabled = (coverage > 0.) & (state.residual_scale > 0.)
    gain = jp.where(enabled, gain, 0.)
    state = state._replace(residual_filter=jp.where(enabled, state.residual_filter, 0.))
    applied = jp.clip(policy_action, -1., 1.) * gain
    next_state, output = _firmware_step(state, policy_action=applied, **kwargs)
    return next_state._replace(residual_gain=gain, applied_action=applied), output
