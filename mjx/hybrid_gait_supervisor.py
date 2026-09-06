"""Geometry-owned NORMAL -> SHORT -> WAVE -> HOLD; never an RL gait action."""
from typing import NamedTuple
from adaptive_contract import TRIPOD_PHASE_LIMITS, WAVE_PHASE_LIMITS
import jax.numpy as jp
from wave_gait_scheduler import TRIPOD, WAVE, TRIPOD_PHASE_S, WAVE_PHASE_S

NORMAL, SHORT, WAVE_MODE, HOLD = range(4)
MODE_NAMES = ('TRIPOD_NORMAL', 'TRIPOD_SHORT', 'WAVE', 'HOLD')
STRIDE_SCALES = jp.array((1.3, 1., .75, .5, .25, .125))
RETURN_CONFIRM_S = .5
MIN_WAVE_DWELL_S = 2.


class SupervisorState(NamedTuple):
    return_time: object
    wave_time: object
    decision: object


def initial_supervisor():
    return SupervisorState(jp.asarray(0.), jp.asarray(0.), jp.asarray(HOLD))


def phase_duration(scale, mode):
    # L = v * nominal_duration * scale. Low scales hit T_min and consequently
    # reduce velocity; Wave has its own one-second baseline and five-phase stance.
    return jp.where(mode == WAVE, jp.clip(WAVE_PHASE_S*scale, *WAVE_PHASE_LIMITS),
                    jp.clip(TRIPOD_PHASE_S*scale, *TRIPOD_PHASE_LIMITS))


def decide(s, *, tripod_feasible, tripod_known_bad, wave_feasible, two_tripod_phases,
           current_mode, requested_scale, dt, fixed_mode='hybrid'):
    return_time = jp.where(two_tripod_phases, s.return_time+dt, 0.)
    wave_time = jp.where(current_mode == WAVE, s.wave_time+dt, 0.)
    # Oversize stride is permitted only if normal stride also passed preflight.
    normal = tripod_feasible[1]
    eligible = tripod_feasible & (STRIDE_SCALES <= requested_scale+1e-6)
    eligible &= (STRIDE_SCALES <= 1.) | normal
    has_tripod = jp.any(eligible)
    index = jp.argmax(eligible.astype(jp.int32))
    normal_or_short = jp.where(STRIDE_SCALES[index] >= 1., NORMAL, SHORT)
    all_known_bad = jp.all(tripod_known_bad[1:])
    decision = jp.where(has_tripod, normal_or_short,
                        jp.where(all_known_bad & wave_feasible, WAVE_MODE, HOLD))
    can_return = (return_time >= RETURN_CONFIRM_S) & (wave_time >= MIN_WAVE_DWELL_S)
    decision = jp.where((current_mode == WAVE) & ~can_return,
                        jp.where(wave_feasible, WAVE_MODE, HOLD), decision)
    if fixed_mode == 'tripod':
        decision = jp.where(has_tripod, normal_or_short, HOLD)
    elif fixed_mode == 'wave':
        decision = jp.where(wave_feasible, WAVE_MODE, HOLD)
    return SupervisorState(return_time, wave_time, decision), index
