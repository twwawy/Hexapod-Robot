"""JAX contact scheduler port of main 9752c760 (also de77791d) gait-manager semantics.

Hardware raw/confirmed FSR is represented by policy-rate MJX contact input.
The environment supplies confirmed contacts; stale 5-ms repeats are not samples.
"""
from typing import NamedTuple
import jax.numpy as jp

TRIPOD, WAVE = 0, 1
WAVE_ORDER = jp.array((0, 5, 1, 3, 2, 4), dtype=jp.int32)
from adaptive_contract import TRIPOD_PHASE_S, WAVE_PHASE_S
WAVE_SPEED_SCALE = .2
WAVE_STANCE_PHASES = 5.
START_DELAY_S = .1
LATE_SPEED = .12
LATE_INWARD_SPEED = .096
LATE_DISTANCE = .10
RECONTACT_SPEED = .03
RECONTACT_DISTANCE = .03
RECONTACT_TIMEOUT_S = 1.
STANCE, SWING, LATE, TOUCHDOWN, HOLD = range(5)


class SchedulerState(NamedTuple):
    mode: object
    phase: object
    epoch: object
    running: object
    elapsed: object
    landed: object
    airborne: object
    late_distance: object
    fault: object
    start_wait: object
    completed: object
    switched: object
    recontact_active: object
    recontact_mask: object
    recontact_time: object
    recontact_distance: object
    recontact_exhausted: object
    recontact_ik_blocked: object
    wave_completed: object
    leg_age: object


def initial_scheduler():
    return SchedulerState(jp.asarray(TRIPOD), jp.asarray(0), jp.asarray(0), jp.asarray(False),
        jp.asarray(0.), jp.zeros(6, dtype=jp.bool_), jp.zeros(6, dtype=jp.bool_),
        jp.zeros(6), jp.asarray(False), jp.asarray(0.), jp.asarray(False), jp.asarray(False),
        jp.asarray(False), jp.zeros(6, dtype=jp.bool_), jp.asarray(0.), jp.zeros(6),
        jp.asarray(False), jp.asarray(False), jp.asarray(0), jp.zeros(6, dtype=jp.int32))


def swing_mask(mode, phase):
    return jp.where(mode == WAVE, jp.arange(6) == WAVE_ORDER[phase % 6],
                    (jp.arange(6) % 2) == (phase % 2))


def advance(s, *, requested_mode, permit, proposal_epoch, command_active, contacts, raw_contacts, duration, dt, proposed_wave_phase=-1):
    all_contact = jp.all(contacts)
    fault = s.fault & command_active  # releasing the command rearms exhausted search
    wait = jp.where(~s.running & command_active & all_contact, s.start_wait+dt, 0.)
    # Recontact is independent of permission to start a NEW swing. Never lower
    # all feet at initial reset; require at least one completed gait phase.
    lost_at_boundary = ~s.running & command_active & (s.epoch > 0) & ~all_contact & ~fault
    recovering = (s.recontact_active | lost_at_boundary) & command_active & ~fault
    recovery_started = recovering & ~s.recontact_active
    recovery_time = jp.where(recovery_started, 0., s.recontact_time)
    recovery_distance = jp.where(recovery_started | ~command_active, jp.zeros(6), s.recontact_distance)
    recovery_mask = jp.where(recovering, jp.where(recovery_started, ~contacts, s.recontact_mask | ~contacts), False)
    settled = recovering & all_contact & (wait >= START_DELAY_S)
    exhausted = recovering & ~all_contact & ((recovery_time >= RECONTACT_TIMEOUT_S) |
        jp.any(recovery_mask & ~contacts & (recovery_distance >= RECONTACT_DISTANCE)))
    fault |= exhausted
    launch = (~s.running & command_active & all_contact & ~fault & ~recovering & permit &
              (proposal_epoch == s.epoch) & (wait >= START_DELAY_S))
    # Pattern changes can only happen at a new, all-contact phase boundary.
    switched = launch & (requested_mode != s.mode)
    mode = jp.where(launch, requested_mode, s.mode)
    phase = jp.where(switched, 0, s.phase)
    # Select only at a validated boundary; map updates cannot change an active leg.
    phase = jp.where(launch & (mode == WAVE) & (proposed_wave_phase >= 0),
                     proposed_wave_phase % 6, phase)
    wave_completed = jp.where(switched, 0, s.wave_completed)
    mask = swing_mask(mode, phase)
    running = s.running | launch
    airborne = jp.where(launch, False, s.airborne) | (running & mask & ~raw_contacts)
    landed = jp.where(launch, False, s.landed)
    support = ~mask | landed
    missing_support = running & support & ~contacts
    recovery = jp.any(missing_support)
    raw_landing = running & mask & airborne & raw_contacts & ~contacts & (s.elapsed/duration >= .5)
    frozen = recovery | jp.any(raw_landing)
    elapsed = jp.where(launch, 0., jp.where(running & ~frozen, s.elapsed+dt, s.elapsed))
    progress = jp.clip(elapsed/duration, 0., 1.)
    landed |= running & mask & airborne & contacts & (progress >= .5)
    complete = running & (progress >= 1.) & jp.all(~mask | landed) & all_contact
    states = jp.where(running & mask & ~landed, jp.where(progress >= 1., LATE, SWING), STANCE)
    states = jp.where(raw_landing, TOUCHDOWN, states)
    states = jp.where(recovery, jp.where(missing_support, jp.where(raw_contacts, TOUCHDOWN, LATE), HOLD), states)
    states = jp.where(~running | complete, HOLD, states)
    # Apply after the idle HOLD override, otherwise the recovery is erased.
    states = jp.where(recovering & ~settled,
        jp.where(recovery_mask & ~contacts, jp.where(raw_contacts, TOUCHDOWN, LATE), HOLD), states)
    distance = jp.where(launch, 0., s.late_distance) + ((states == LATE) & ~recovering)*LATE_SPEED*dt
    fault |= jp.any(distance >= LATE_DISTANCE)
    states = jp.where(fault, HOLD, states)
    running &= ~complete & ~fault
    next_state = SchedulerState(mode, phase+complete.astype(jp.int32), s.epoch+complete.astype(jp.int32),
        running, jp.where(complete, 0., elapsed), landed, airborne, distance, fault,
        jp.where(launch | complete, 0., wait), complete, switched,
        recovering & ~settled & ~fault, recovery_mask,
        jp.where(recovering, recovery_time+dt, 0.), recovery_distance,
        (s.recontact_exhausted | exhausted) & command_active,
        s.recontact_ik_blocked & command_active,
        wave_completed + (complete & (mode == WAVE)).astype(jp.int32),
        jp.where(complete, jp.where(mask, 0, jp.minimum(s.leg_age+1, 100)), s.leg_age))
    return next_state, dict(state=states.astype(jp.int32), progress=jp.full(6, progress),
        startup=jp.where(mode == WAVE, wave_completed < 6, phase < 1), enabled=running,
        entering=launch & mask, frozen=frozen | complete | fault | ~running,
        swing_mask=mask & running, recontact_active=recovering & ~settled & ~fault)
