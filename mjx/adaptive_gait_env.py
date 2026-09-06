"""MJX parameter-policy environment: LiDAR actor and privileged terrain critic.

No CPU raycast is called from reset/step. The actor never receives ground-truth
terrain in lidar mode. Simulator pose/contact act as ideal state estimation.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import adaptive_gait_controller as adaptive
import firmware_mjx_controller as fw
import wave_gait_scheduler as scheduler
import hybrid_gait_supervisor as supervisor
from foothold_feasibility import phase_feasibility, support_margin
from adaptive_gait_perception import AngularLidar, SENSOR_PERIOD, MAX_AGE, initial_map, sample
from rough_terrain_env import (
    HexapodRoughTerrainEnv, MODEL_FORWARD, MODEL_LATERAL,
    _quat_rotate_inverse, default_config,
)
from terrain_curriculum import terrain_level as terrain_spec
from adaptive_foothold_estimator import (
    CANDIDATE_COUNT, CANDIDATE_OFFSETS, CANDIDATE_FEATURES,
    local_candidate_xy, project_local_candidates, residual_extent, LOCAL_CENTER_INDEX,
    FOOT_RADIUS, evaluate_candidates,
)

OBSERVATION_CONTRACT = 'adaptive_hybrid_geometry_local25_extent_24_v4'
REWARD_CONTRACT = 'adaptive_hybrid_efficient_progress_v4'
PROPRIO_SIZE = 157
GLOBAL_SIZE = 23
REFERENCE_SIZE = 6*9
BOOTSTRAP_MAX_PHASES = 24
ACTOR_SIZE = PROPRIO_SIZE + GLOBAL_SIZE + REFERENCE_SIZE + 6*CANDIDATE_COUNT*len(CANDIDATE_FEATURES)
CRITIC_SIZE = ACTOR_SIZE + 6*CANDIDATE_COUNT*2 + 15


class AdaptiveGaitEnv(HexapodRoughTerrainEnv):
    def __init__(self, *, terrain_level=0, perception='lidar', config=None,
                 azimuths=90, elevations=8, dropout=.05, noise=.005, gait_mode='hybrid', diagnostics=False,
                 bootstrap_unmapped=True):
        if perception not in ('lidar', 'teacher', 'blind', 'oracle'):
            raise ValueError('perception must be lidar, teacher, blind or debug-only oracle')
        if terrain_spec(terrain_level).kind == 'rough':
            raise ValueError('MJX ray does not support hfields: use level 0, 3, 4 or 5..16')
        self.perception = perception
        if gait_mode not in ('hybrid', 'tripod', 'wave'):
            raise ValueError('gait_mode must be hybrid, tripod or wave')
        self.gait_mode, self.diagnostics = gait_mode, diagnostics
        self.bootstrap_unmapped = bool(bootstrap_unmapped)
        config = default_config() if config is None else config
        # Posture is owned by the parameter policy, not a terrain-kind sampler.
        config.command.height_min = config.command.height_max = 0.
        config.command.pitch_min_deg = config.command.pitch_max_deg = 0.
        super().__init__(config=config, terrain_level=terrain_level)
        self._root_id = self.mj_model.body('hexapod').id
        self._home_qpos = self._home_qpos.at[2].set(-fw.BASE_FOOT_Z + FOOT_RADIUS)
        self.sensor = AngularLidar(self.mjx_model, self._root_id, azimuths, elevations, dropout, noise)

    def _configure_collision_masks(self):
        # Use the old training skeleton endpoint (230 mm, spherical foot centre)
        # consistently in collision geometry, controller IK and touchdown Z.
        model = self._mj_model
        for leg in adaptive.LEG_ORDER:
            outward = model.body(f'{leg}_motor_horn_1_1').pos.copy()
            outward[2] = 0.
            outward /= np.linalg.norm(outward)
            foot = model.geom(f'{leg}_foot_collision').id
            tibia = model.geom(f'{leg}_tibia_collision').id
            model.geom_pos[foot] = .230*outward
            model.site_pos[model.site(f'{leg}_foot_site').id] = .230*outward
            model.geom_pos[tibia] = .115*outward
            model.geom_size[tibia, 1] = .115
            mujoco.mju_quatZ2Vec(model.geom_quat[tibia], outward)
        super()._configure_collision_masks()

    def _terrain_pitch_ff(self, *args):
        return jp.asarray(0.)

    def _terrain_swing_boost(self, *args):
        return jp.asarray(0.)

    def _initialize_controller_info(self, data, info):
        info['command'] = info['command'].at[2:].set(0.)
        info['controller_state'] = adaptive.initial_state()
        info['lidar_map'] = initial_map(data.qpos[:2])
        info['projection_m'] = jp.asarray(0.)
        info['supervisor'] = supervisor.initial_supervisor()
        info['contact_confirm_time'] = jp.zeros(6)
        info['confirmed_contacts'] = jp.zeros(6, dtype=jp.bool_)
        info['slip_estimate'] = jp.zeros(6)
        info['bootstrap_complete'] = jp.asarray(self.perception != 'lidar')
        info['foothold_plan'] = self._landing_plan(data, info, jp.zeros(adaptive.ACTION_SIZE))
        return info

    def _query(self, grid, xy, now, *, privileged=False):
        height, known, age, spread = sample(grid, xy, now)
        if self.perception == 'blind':
            known = jp.zeros_like(known)
        if privileged or self.perception in ('teacher', 'oracle'):
            height = self._terrain_height(xy.reshape(-1, 2)).reshape(xy.shape[:-1])
            spread = jp.zeros_like(height)
        if self.perception == 'oracle' or (privileged and self.diagnostics):
            known, age = jp.ones_like(known), jp.zeros_like(age)
        return height, known, age, spread

    def _candidates(self, data, info, stride=1., period=1., *, lift=None, apex_delta=0.,
                    transfer_delta=0., mode=0, privileged=False):
        lift = jp.zeros(6) if lift is None else lift
        rotation = data.xmat[self._root_id]
        cs = info['controller_state']
        # The same speed/phase relationship drives stance and predicts landing.
        base_period = jp.where(mode == scheduler.WAVE, scheduler.WAVE_PHASE_S, scheduler.TRIPOD_PHASE_S)
        speed_scale = stride*base_period/period*jp.where(mode == scheduler.WAVE, scheduler.WAVE_SPEED_SCALE, 1.)
        vx = jp.clip(info['command'][0]*speed_scale,
                     -adaptive.MAX_LINEAR_SPEED, adaptive.MAX_LINEAR_SPEED)
        vy, wz = jp.asarray(0.), info['command'][1]*speed_scale
        displacement = period*jp.stack((-vx+wz*fw.BASE_FEET[:, 1],
                                        -vy-wz*fw.BASE_FEET[:, 0], jp.zeros(6)), axis=-1)
        front = fw.BASE_FEET - .5*jp.where(mode == scheduler.WAVE, scheduler.WAVE_STANCE_PHASES, 1.)*displacement
        body = fw._rotate_inverse(front.at[:, 2].add(-cs.height_applied), cs.posture_command)
        model = jp.stack((body[:, 1], -body[:, 0], body[:, 2]), axis=-1)
        # Predict translation and yaw to touchdown, then freeze the chosen world
        # patch at swing entry. Subsequent sensing cannot retarget that swing.
        heading = jp.arctan2((rotation @ MODEL_FORWARD)[1], (rotation @ MODEL_FORWARD)[0])
        angle = wz*period
        yaw_rot = jp.array(((jp.cos(angle), -jp.sin(angle), 0.),
                            (jp.sin(angle), jp.cos(angle), 0.), (0., 0., 1.)))
        heading_mid = heading+angle/2
        translation = period*jp.array((vx*jp.cos(heading_mid)-vy*jp.sin(heading_mid),
                                        vx*jp.sin(heading_mid)+vy*jp.cos(heading_mid), 0.))
        nominal = data.qpos[:3] + translation + model @ (yaw_rot @ rotation).T
        basis = jp.stack((rotation @ MODEL_FORWARD, rotation @ MODEL_LATERAL))[:, :2]
        xy = nominal[:, None, :2] + CANDIDATE_OFFSETS[None, :, :] @ basis
        return evaluate_candidates(self, data, info, xy, nominal, basis, lift, apex_delta=apex_delta,
                                   transfer_delta=transfer_delta, privileged=privileged)

    def _phase_check(self, plan, data, info, feet, mode, phase, period, stride):
        check = phase_feasibility(plan, feet, info['confirmed_contacts'], data.subtree_com[self._root_id], mode, phase)
        speed_scale = stride*jp.where(mode == scheduler.WAVE, scheduler.WAVE_PHASE_S*scheduler.WAVE_SPEED_SCALE,
                                      scheduler.TRIPOD_PHASE_S)/period
        rotation = data.xmat[self._root_id]
        times = jp.linspace(0., period, 5)
        vx, wz = info['command'][0]*speed_scale, info['command'][1]*speed_scale
        heading = jp.arctan2((rotation @ MODEL_FORWARD)[1], (rotation @ MODEL_FORWARD)[0])
        def stance_at(t):
            angle = wz*t
            rot = jp.array(((jp.cos(angle), -jp.sin(angle), 0.),
                            (jp.sin(angle), jp.cos(angle), 0.), (0., 0., 1.))) @ rotation
            origin = data.qpos[:3]+vx*t*jp.array((jp.cos(heading+angle/2), jp.sin(heading+angle/2), 0.))
            model = (feet-origin) @ rot
            body = jp.stack((-model[:, 1], model[:, 0], model[:, 2]), axis=-1)
            _, valid = fw._solve_ik(body)
            _, limited = fw._limit_foot_reach(body)
            return jp.all(check['swing_mask'] | (valid & ~limited))
        stance_ok = jp.all(jax.vmap(stance_at)(times))
        check['feasible'] &= stance_ok
        check['known_infeasible'] |= check['observed'] & ~stance_ok
        return check

    def _landing_plan(self, data, info, action):
        xy_bias, lift, posture, requested_stride, apex_delta, transfer_delta = adaptive.decode(action)
        cs = info['controller_state']
        phase = jp.where(cs.scheduler.mode == scheduler.TRIPOD, cs.scheduler.phase, 0)
        feet = data.site_xpos[self._foot_site_ids]
        contacts = info['confirmed_contacts']
        com = data.subtree_com[self._root_id]
        periods = jax.vmap(lambda scale: supervisor.phase_duration(scale, scheduler.TRIPOD))(supervisor.STRIDE_SCALES)
        bank = jax.vmap(lambda scale, period: self._candidates(data, info, scale, period))(supervisor.STRIDE_SCALES, periods)
        checks = jax.vmap(lambda plan, period, stride: self._phase_check(plan, data, info, feet,
            scheduler.TRIPOD, phase, period, stride))(bank, periods, supervisor.STRIDE_SCALES)
        normal_plan = jax.tree_util.tree_map(lambda value: value[1], bank)
        # Second Tripod preview uses the first accepted group's projected contacts.
        first_ids = checks['indices'][1]
        first_targets = normal_plan['world'][jp.arange(6), first_ids]
        first_feet = jp.where(scheduler.swing_mask(scheduler.TRIPOD, phase)[:, None],
                             first_targets.at[:, 2].add(FOOT_RADIUS), feet)
        forward = data.xmat[self._root_id] @ MODEL_FORWARD
        shift = forward*info['command'][0]*scheduler.TRIPOD_PHASE_S
        future_data = data.replace(qpos=data.qpos.at[:3].add(shift),
            site_xpos=data.site_xpos.at[self._foot_site_ids].set(first_feet),
            subtree_com=data.subtree_com.at[self._root_id].add(shift))
        model_feet = (first_feet-future_data.qpos[:3]) @ data.xmat[self._root_id]
        body_feet = jp.stack((-model_feet[:, 1], model_feet[:, 0], model_feet[:, 2]), axis=-1)
        preview_memory = (body_feet @ fw._rotation_matrix(cs.posture_command).T).at[:, 2].add(cs.height_applied)
        future_info = dict(info, controller_state=cs._replace(foot_memory=preview_memory),
                           confirmed_contacts=jp.ones(6, dtype=jp.bool_))
        next_plan = self._candidates(future_data, future_info)
        next_check = self._phase_check(next_plan, future_data, future_info, first_feet,
                                       scheduler.TRIPOD, phase+1, scheduler.TRIPOD_PHASE_S, 1.)
        wave_phase = jp.where(cs.scheduler.mode == scheduler.WAVE, cs.scheduler.phase, 0)
        wave_periods = jax.vmap(lambda scale: supervisor.phase_duration(scale, scheduler.WAVE))(supervisor.STRIDE_SCALES)
        wave_bank = jax.vmap(lambda scale, duration: self._candidates(data, info, scale, duration,
            mode=scheduler.WAVE))(supervisor.STRIDE_SCALES, wave_periods)
        wave_checks = jax.vmap(lambda p, duration, scale: self._phase_check(p, data, info, feet,
            scheduler.WAVE, wave_phase, duration, scale))(wave_bank, wave_periods, supervisor.STRIDE_SCALES)
        wave_eligible = wave_checks['feasible'] & (supervisor.STRIDE_SCALES <= requested_stride+1e-6)
        wave_index = jp.argmax(wave_eligible.astype(jp.int32))
        wave_plan = jax.tree_util.tree_map(lambda value: value[wave_index], wave_bank)
        wave_check = jax.tree_util.tree_map(lambda value: value[wave_index], wave_checks)
        wave_check['feasible'] &= jp.any(wave_eligible)
        next_supervisor, bank_index = supervisor.decide(info['supervisor'],
            tripod_feasible=checks['feasible'], tripod_known_bad=checks['known_infeasible'],
            wave_feasible=wave_check['feasible'], two_tripod_phases=checks['feasible'][1] & next_check['feasible'],
            current_mode=cs.scheduler.mode, requested_scale=requested_stride, dt=self.dt, fixed_mode=self.gait_mode)
        decision = next_supervisor.decision
        if self.perception == 'blind':
            decision = jp.asarray(supervisor.WAVE_MODE if self.gait_mode == 'wave' else supervisor.NORMAL)
            bank_index = jp.asarray(1)
        mode = jp.where(decision == supervisor.WAVE_MODE, scheduler.WAVE, scheduler.TRIPOD)
        stride = jp.where(mode == scheduler.WAVE, supervisor.STRIDE_SCALES[wave_index], supervisor.STRIDE_SCALES[bank_index])
        period = supervisor.phase_duration(stride, mode)
        tripod_plan = jax.tree_util.tree_map(lambda value: value[bank_index], bank)
        reference_plan = jax.tree_util.tree_map(lambda t, w: jp.where(mode == scheduler.WAVE, w, t), tripod_plan, wave_plan)
        selected_check = jax.tree_util.tree_map(lambda t, w: jp.where(mode == scheduler.WAVE, w, t[bank_index]), checks, wave_check)
        global_known = selected_check['feasible']
        # p_ref is chosen BEFORE RL XY/lift; search and residual are separate.
        reference_index = reference_plan['reference_index']
        reference_index = jp.where(selected_check['swing_mask'] & selected_check['feasible'], selected_check['indices'], reference_index)
        leg = jp.arange(6)
        ref = reference_plan['world'][leg, jp.maximum(reference_index, 0)]
        ref = jp.where((reference_index >= 0)[:, None], ref,
                       reference_plan['nominal'].at[:, 2].add(-FOOT_RADIUS))
        requested = ref[:, :2] + xy_bias @ reference_plan['basis']
        local_xy = local_candidate_xy(ref, reference_plan['basis'])
        local_nominal = ref.at[:, 2].add(FOOT_RADIUS)
        refined = evaluate_candidates(self, data, info, local_xy, local_nominal, reference_plan['basis'],
                                      lift, apex_delta=apex_delta, transfer_delta=transfer_delta)
        neutral_local = evaluate_candidates(self, data, info, local_xy, local_nominal,
                                             reference_plan['basis'], jp.zeros(6))
        relative_xy = (refined['xy']-ref[:, None, :2]) @ jp.linalg.inv(reference_plan['basis'])
        bounded = jp.all(jp.abs(relative_xy) <= jp.asarray(adaptive.XY_LIMIT_M)+1e-6, axis=-1)
        eligible = refined['safe'] & (refined['path_coverage'] >= .6) & bounded & (reference_index >= 0)[:, None]
        selected, residual_ok = project_local_candidates(refined['xy'], eligible, requested)
        selected = jp.where(residual_ok, selected, jp.where(reference_index >= 0, LOCAL_CENTER_INDEX, -1))
        # If the lift request invalidates every nearby option, use the neutral
        # reference path. A rejected residual cannot erase a safe reference.
        plan = dict(refined)
        for name in ('world', 'pre', 'clearance', 'required', 'apex_phase', 'transfer'):
            values = refined[name][leg, jp.maximum(selected, 0)]
            fallback = reference_plan[name][leg, jp.maximum(reference_index, 0)]
            mask = residual_ok
            while mask.ndim < values.ndim:
                mask = mask[..., None]
            plan['selected_'+name] = jp.where(mask, values, fallback)
        known = selected >= 0
        fallback_mask = (~residual_ok)[:, None] & (jp.arange(CANDIDATE_COUNT)[None, :] == LOCAL_CENTER_INDEX)
        for name in ('safe', 'ik_ok', 'path_ok', 'status'):
            ref_value = reference_plan[name][leg, jp.maximum(reference_index, 0)]
            plan[name] = jp.where(fallback_mask, ref_value[:, None], plan[name])
        observed_hazard = jp.any(reference_plan['unsafe'] | reference_plan['terrain_ok'], axis=1)
        proposal_safe = known | ~observed_hazard
        if self.perception != 'blind':
            proposal_safe &= known
        # Partial observations alone are not filled with GT or treated as flat.
        plan['selected_world'] = jp.where(known[:, None], plan['selected_world'],
                                           reference_plan['nominal'].at[:, 2].add(-FOOT_RADIUS))
        accepted_xy = (plan['selected_world'][:, :2]-ref[:, :2]) @ jp.linalg.inv(reference_plan['basis'])
        accepted = action.at[:12].set(jp.where(known[:, None], jp.clip(accepted_xy/jp.asarray(adaptive.XY_LIMIT_M), -1., 1.), 0.).reshape(-1))
        effective_lift = plan['selected_clearance']-fw.SWING_HEIGHT-plan['selected_required']
        accepted = accepted.at[12:18].set(jp.where(known, jp.clip(effective_lift/.04, -1., 1.), 0.))
        accepted = accepted.at[18:].set(jp.where(global_known, action[18:], 0.))
        gradients = reference_plan['gradient'][leg, jp.maximum(reference_index, 0)] @ reference_plan['basis'].T
        gradient = jp.sum(jp.where((reference_index >= 0)[:, None], gradients, 0.), axis=0)/jp.maximum(jp.sum(reference_index >= 0), 1)
        posture = posture.at[:2].add(jp.clip(jp.array((jp.arctan(gradient[1]), -jp.arctan(gradient[0]))),
                                             -jp.deg2rad(12.), jp.deg2rad(12.)))
        posture = posture.at[:2].set(jp.clip(posture[:2], -jp.deg2rad(15.), jp.deg2rad(15.)))
        posture = jp.where(global_known, posture, jp.zeros(3))
        accepted = accepted.at[21].set(jp.clip(jp.where(stride < 1., (stride-1.)/.5, (stride-1.)/.3), -1., 1.))
        projection = jp.where(known, jp.linalg.norm(plan['selected_world'][:, :2]-requested, axis=-1), 0.)
        max_stride = jp.max(jp.where(checks['feasible'], supervisor.STRIDE_SCALES, 0.))
        # Refinement must preserve the combined landing support margin too.
        future_feet = jp.where(selected_check['swing_mask'][:, None],
            plan['selected_world'].at[:, 2].add(FOOT_RADIUS), feet)
        refined_margin = support_margin(future_feet[:, :2],
            jp.where(mode == scheduler.WAVE, jp.ones(6, dtype=jp.bool_), selected_check['swing_mask']), com[:2])
        combo_ok = refined_margin >= .012
        for name in ('world', 'pre', 'clearance', 'required', 'apex_phase', 'transfer'):
            fallback = reference_plan[name][leg, jp.maximum(reference_index, 0)]
            plan['selected_'+name] = jp.where(combo_ok, plan['selected_'+name], fallback)
        selected = jp.where(combo_ok, selected, jp.where(reference_index >= 0, LOCAL_CENTER_INDEX, -1))
        accepted = accepted.at[:18].set(jp.where(combo_ok, accepted[:18], 0.))
        # Explicit blind mode is nominal controller debugging, never hybrid fallback.
        if self.perception == 'blind':
            nominal_world = reference_plan['nominal'].at[:, 2].add(-FOOT_RADIUS)
            plan['selected_world'] = nominal_world
            model = (nominal_world+jp.array((0., 0., FOOT_RADIUS))-data.qpos[:3]) @ data.xmat[self._root_id]
            body = jp.stack((-model[:, 1], model[:, 0], model[:, 2]), axis=-1)
            plan['selected_pre'] = (body @ fw._rotation_matrix(cs.posture_command).T).at[:, 2].add(cs.height_applied)
            plan['selected_clearance'] = jp.full(6, fw.SWING_HEIGHT)
            plan['selected_apex_phase'], plan['selected_transfer'] = jp.full(6, .5), jp.full(6, .5)
            proposal_safe, accepted, posture = jp.ones(6, dtype=jp.bool_), jp.zeros(adaptive.ACTION_SIZE), jp.zeros(3)

        # The upward-mounted LiDAR may not observe the first landing patches
        # until the body has moved. Break that mapping/locomotion deadlock with
        # conservative classical Tripod phases. This is permitted only until a
        # local support patch has been observed, with all six feet confirmed,
        # and without claiming that unknown map cells are generally safe.
        local_plan_ready = jp.any(checks['feasible'])
        info['bootstrap_complete'] = info['bootstrap_complete'] | local_plan_ready
        bootstrap = (self.bootstrap_unmapped & (self.perception == 'lidar') &
                     ~info['bootstrap_complete'] & ~cs.scheduler.running & jp.all(contacts) &
                     (cs.scheduler.epoch < BOOTSTRAP_MAX_PHASES))
        bootstrap_stride = jp.asarray(.5)
        bootstrap_period = supervisor.phase_duration(bootstrap_stride, scheduler.TRIPOD)
        bootstrap_plan = self._candidates(data, info, bootstrap_stride, bootstrap_period,
                                          mode=scheduler.TRIPOD)
        bootstrap_world = bootstrap_plan['nominal'].at[:, 2].add(-FOOT_RADIUS)
        model = (bootstrap_world+jp.array((0., 0., FOOT_RADIUS))-data.qpos[:3]) @ data.xmat[self._root_id]
        body = jp.stack((-model[:, 1], model[:, 0], model[:, 2]), axis=-1)
        bootstrap_pre = (body @ fw._rotation_matrix(cs.posture_command).T).at[:, 2].add(cs.height_applied)
        plan['selected_world'] = jp.where(bootstrap, bootstrap_world, plan['selected_world'])
        plan['selected_pre'] = jp.where(bootstrap, bootstrap_pre, plan['selected_pre'])
        plan['selected_clearance'] = jp.where(bootstrap, jp.full(6, fw.SWING_HEIGHT), plan['selected_clearance'])
        plan['selected_required'] = jp.where(bootstrap, jp.zeros(6), plan['selected_required'])
        plan['selected_apex_phase'] = jp.where(bootstrap, jp.full(6, .5), plan['selected_apex_phase'])
        plan['selected_transfer'] = jp.where(bootstrap, jp.full(6, .5), plan['selected_transfer'])
        known = jp.where(bootstrap, jp.ones(6, dtype=jp.bool_), known)
        proposal_safe = jp.where(bootstrap, jp.ones(6, dtype=jp.bool_), proposal_safe)
        selected = jp.where(bootstrap, jp.full(6, -1, dtype=jp.int32), selected)
        accepted = jp.where(bootstrap, jp.zeros(adaptive.ACTION_SIZE), accepted)
        posture = jp.where(bootstrap, jp.zeros(3), posture)
        decision = jp.where(bootstrap, supervisor.SHORT, decision)
        mode = jp.where(bootstrap, scheduler.TRIPOD, mode)
        stride = jp.where(bootstrap, bootstrap_stride, stride)
        period = jp.where(bootstrap, bootstrap_period, period)
        global_known = global_known | bootstrap
        projection = jp.where(known, jp.linalg.norm(plan['selected_world'][:, :2]-requested, axis=-1), 0.)
        plan.update(time=data.time, reference_index=reference_index, reference_world=ref, selected_index=selected,
                    wide_xy=reference_plan['xy'], wide_height=reference_plan['height'],
                    wide_status=reference_plan['status'], wide_nominal=reference_plan['nominal'],
                    local_extent=residual_extent(neutral_local['safe'] & (reference_index >= 0)[:, None]),
                    requested_xy=requested, selected_known=known, proposal_safe=proposal_safe,
                    residual_rejected=(reference_index >= 0) & (~residual_ok | ~combo_ok),
                    projection=projection, accepted_action=accepted, posture=posture, stride=stride, period=period,
                    reference_safe=reference_plan['safe'], reference_status=reference_plan['status'],
                    decision=decision, mode=mode, permit=decision != supervisor.HOLD,
                    bootstrap_classical=bootstrap,
                    speed_scale=stride*jp.where(mode == scheduler.WAVE, scheduler.WAVE_PHASE_S*scheduler.WAVE_SPEED_SCALE,
                                               scheduler.TRIPOD_PHASE_S)/period,
                    max_feasible_stride=max_stride, tripod_feasible=checks['feasible'],
                    tripod_known_bad=checks['known_infeasible'], wave_feasible=wave_check['feasible'],
                    two_tripod_phases=checks['feasible'][1] & next_check['feasible'],
                    support_margin=selected_check['support_margin'],
                    return_time=next_supervisor.return_time, wave_time=next_supervisor.wave_time)
        return plan

    def _prepare_controller_state(self, data, info, action):
        plan = self._landing_plan(data, info, action)
        info['foothold_plan'] = plan
        info['supervisor'] = supervisor.SupervisorState(plan['return_time'], plan['wave_time'], plan['decision'])
        info['projection_m'] = jp.mean(plan['projection'])
        return info['controller_state']._replace(request=plan['accepted_action'],
            proposal_end=plan['selected_pre'], proposal_world=plan['selected_world'],
            proposal_known=plan['selected_known'], proposal_safe=plan['proposal_safe'],
            proposal_clearance=plan['selected_clearance'], proposal_posture=plan['posture'],
            proposal_stride=plan['stride'], proposal_period=plan['period'], proposal_index=plan['selected_index'],
            proposal_mode=plan['mode'], proposal_permit=plan['permit'],
            proposal_epoch=info['controller_state'].scheduler.epoch,
            proposal_apex_phase=plan['selected_apex_phase'], proposal_transfer=plan['selected_transfer'],
            proposal_speed_scale=plan['speed_scale'], raw_contacts=info['contact_state'], confirmed_contacts=info['confirmed_contacts'],
            root_rotation=data.xmat[self._root_id], root_position=data.qpos[:3])

    def _controller_step(self, controller_state, **kwargs):
        return adaptive.step(controller_state, **kwargs)

    def _reward_posture_target(self, info, controller_state, pitch_ff):
        return controller_state.adapt_posture[:2]

    def _reward_height_command(self, info, controller_state):
        return controller_state.height_applied

    def _reward_command(self, info, controller_state):
        # Wave/short-step tracking follows the speed the supervisor accepted.
        scale = jp.where(controller_state.scheduler.running, controller_state.speed_scale, 0.)
        return info['command'].at[:2].multiply(scale)

    def _reward_speed_floor(self):
        return .005

    def _get_obs(self, data, info):
        cs, out = info['controller_state'], info['controller_output']
        proprio = jp.concatenate((info['command'],
            _quat_rotate_inverse(data.qpos[3:7], data.qvel[:3]), .2*data.qvel[3:6],
            _quat_rotate_inverse(data.qpos[3:7], jp.array((0., 0., -1.))),
            data.qpos[self._joint_qpos_ids]-self._home_qpos[self._joint_qpos_ids],
            .1*data.qvel[self._joint_qvel_ids], self._feet_controller_body(data).reshape(-1),
            info['contact_state'].astype(jp.float32), out.gait_progress,
            out.gait_state.astype(jp.float32)/4., out.applied_twist,
            out.ik_valid.astype(jp.float32), out.foot_limited.astype(jp.float32),
            cs.accepted_action, info['last_action'], cs.adapt_posture,
            jp.array((cs.stride_scale, cs.phase_duration, cs.plan_rejected.astype(jp.float32), cs.height_applied))))
        assert proprio.shape == (PROPRIO_SIZE,)
        plan = info['foothold_plan']
        candidates = self._candidates(data, info, plan['stride'], plan['period'], mode=plan['mode'],
                                      privileged=self.perception == 'teacher')
        ref_world = plan['reference_world']
        model_ref = (ref_world-data.qpos[:3]) @ data.xmat[self._root_id]
        ref_body = jp.stack((-model_ref[:, 1], model_ref[:, 0], model_ref[:, 2]), axis=-1)
        ref_known = plan['reference_index'] >= 0
        reference_obs = jp.concatenate((jp.where(ref_known[:, None], ref_body, 0.),
            jp.where(ref_known, jp.linalg.norm(ref_world[:, :2]-plan['wide_nominal'][:, :2], axis=-1), 0.)[:, None],
            ref_known.astype(jp.float32)[:, None],
            plan['local_extent']/jp.array((.06, .06, .04, .04))), axis=-1)
        # vy command is explicitly zero in this MJX bridge; do not imply lateral
        # command support that the inherited firmware interface does not provide.
        foot_surface = data.site_xpos[self._foot_site_ids, 2]-FOOT_RADIUS
        contact_count = jp.sum(info['contact_state'])
        estimated_clearance = jp.where(contact_count > 0, data.qpos[2]-jp.sum(jp.where(
            info['contact_state'], foot_surface, 0.))/jp.maximum(contact_count, 1), -fw.BASE_FOOT_Z+FOOT_RADIUS)
        global_obs = jp.concatenate((jp.zeros(1), self._relative_attitude(data)[:2],
            jp.atleast_1d(estimated_clearance), info['slip_estimate'],
            jp.atleast_1d(cs.scheduler.mode).astype(jp.float32),
            (scheduler.swing_mask(cs.scheduler.mode, cs.scheduler.phase) & cs.scheduler.running).astype(jp.float32),
            jp.array((plan['decision'], plan['max_feasible_stride'], plan['tripod_feasible'][1],
                      plan['wave_feasible'], jp.mean(candidates['confidence']), plan['support_margin']))))
        assert global_obs.shape == (GLOBAL_SIZE,)
        actor = jp.concatenate((proprio, global_obs, reference_obs.reshape(-1), candidates['features'].reshape(-1)))
        # Privileged fields are a separate network input, never concatenated into actor.
        xy = candidates['xy'].reshape(-1, 2)
        gt = self._terrain_height(xy).reshape(6, CANDIDATE_COUNT)
        mapped, valid, _, _ = sample(info['lidar_map'], xy, data.time)
        error = jp.where(valid.reshape(6, CANDIDATE_COUNT), mapped.reshape(6, CANDIDATE_COUNT)-gt, 0.)
        critic = jp.concatenate((actor, ((gt-data.qpos[2])/.3).reshape(-1),
                                 (error/.05).reshape(-1), self._terrain_features(data, info['support_height'])))
        assert actor.shape == (ACTOR_SIZE,) and critic.shape == (CRITIC_SIZE,)
        return {'state': actor, 'privileged_state': critic}

    def reset(self, rng):
        state = super().reset(rng)
        state.metrics.update({name: jp.asarray(0.) for name in (
            'map_known_fraction', 'map_mae_m', 'map_compared_count', 'lidar_returns',
            'plan_rejected', 'projection_m', 'touchdown_error_m', 'foot_slip',
            'efficiency_joint_speed', 'efficiency_vertical_speed', 'efficiency_foot_travel', 'efficiency_excess_clearance',
            'efficiency_tiny_stride', 'stride_scale', 'phase_duration_s', 'pitch_target_rad', 'foothold_center_known_fraction',
            'foothold_coverage_fraction', 'foothold_terrain_fraction', 'foothold_ik_fraction',
            'foothold_safe_fraction', 'foothold_selected_fraction', 'foothold_residual_rejected_fraction',
            'gait_mode', 'supervisor_mode', 'max_feasible_stride', 'support_margin', 'gait_switch',
            'scheduler_fault', 'oracle_safe_recall', 'oracle_false_safe', 'oracle_unknown_fraction',
            'oracle_foothold_error_m', 'oracle_compared', 'oracle_edge_recall', 'oracle_edge_precision')})
        return state

    def step(self, state, action):
        if action.shape != (adaptive.ACTION_SIZE,):
            raise ValueError('hybrid controller requires 24 actions; previous adaptive/stage31 policies are incompatible')
        # Copy dictionaries because the legacy environment updates them in-place.
        state = state.replace(info=dict(state.info), metrics=dict(state.metrics))
        raw = self._foot_contacts(state.data)
        confirmed_time = jp.where(raw, state.info['contact_confirm_time']+self.dt, 0.)
        state.info['contact_confirm_time'] = confirmed_time
        state.info['confirmed_contacts'] = confirmed_time >= .01
        state.info['contact_state'] = raw
        idle = jp.all(jp.abs(state.info['command'][:2]) < .001)
        state.info['no_progress_steps'] = jp.where(idle, 0, state.info['no_progress_steps'])
        state.info['progress_anchor_potential'] = jp.where(idle,
            state.data.qpos[0] + .5*state.info['support_height'], state.info['progress_anchor_potential'])
        previous_contacts = state.info['contact_state']
        previous_mode = state.info['controller_state'].scheduler.mode
        previous_feet = state.data.site_xpos[self._foot_site_ids]
        result = super().step(state, action)
        # Publish the scan in the NEXT observation. The current action and its
        # landing projection must use the same map the actor actually received.
        if self.perception in ('lidar', 'teacher'):
            grid, key = jax.lax.cond(result.info['policy_steps'] % SENSOR_PERIOD == 0,
                lambda _: self.sensor.update(result.data, result.info['lidar_map'], result.info['rng'],
                                              result.info['policy_steps']//SENSOR_PERIOD),
                lambda _: (result.info['lidar_map'], result.info['rng']), operand=None)
            result.info['lidar_map'], result.info['rng'] = grid, key
        cs = result.info['controller_state']
        feet = result.data.site_xpos[self._foot_site_ids]
        touchdown = ~previous_contacts & result.info['contact_state'] & cs.active_known
        target = cs.goal_world + jp.array((0., 0., FOOT_RADIUS))
        landing_error = jp.sum(jp.where(touchdown, jp.linalg.norm(feet-target, axis=-1), 0.))/jp.maximum(jp.sum(touchdown), 1)
        stance = previous_contacts & result.info['contact_state']
        slip = jp.mean(jp.where(stance, jp.sum(((feet-previous_feet)[:, :2]/self.dt)**2, axis=-1), 0.))
        result.info['slip_estimate'] = jp.where(stance, jp.linalg.norm((feet-previous_feet)[:, :2]/self.dt, axis=-1), 0.)
        candidates = self._candidates(result.data, result.info)
        xy = candidates['xy'].reshape(-1, 2)
        mapped, valid, _, _ = sample(result.info['lidar_map'], xy, result.data.time)
        gt = self._terrain_height(xy)
        compared = jp.sum(valid)
        mae = jp.sum(jp.where(valid, jp.abs(mapped-gt), 0.))/jp.maximum(compared, 1)
        result.metrics.update(map_known_fraction=jp.mean(candidates['known'].astype(jp.float32)),
            map_mae_m=mae, map_compared_count=compared.astype(jp.float32),
            lidar_returns=result.info['lidar_map'].hits.astype(jp.float32),
            plan_rejected=cs.plan_rejected.astype(jp.float32), projection_m=result.info['projection_m'],
            touchdown_error_m=landing_error, foot_slip=slip, stride_scale=cs.stride_scale,
            phase_duration_s=cs.phase_duration, pitch_target_rad=cs.adapt_posture[1])
        plan = result.info['foothold_plan']
        result.metrics.update(foothold_center_known_fraction=jp.mean(plan['center_known'].astype(jp.float32)),
            foothold_coverage_fraction=jp.mean(plan['coverage_ok'].astype(jp.float32)),
            foothold_terrain_fraction=jp.mean(plan['terrain_ok'].astype(jp.float32)),
            foothold_ik_fraction=jp.mean((plan['terrain_ok'] & plan['ik_ok']).astype(jp.float32)),
            foothold_safe_fraction=jp.mean(plan['safe'].astype(jp.float32)),
            foothold_selected_fraction=jp.mean(plan['selected_known'].astype(jp.float32)),
            foothold_residual_rejected_fraction=jp.mean(plan['residual_rejected'].astype(jp.float32)))
        switched = cs.scheduler.mode != previous_mode
        result.metrics.update(gait_mode=cs.scheduler.mode.astype(jp.float32), supervisor_mode=plan['decision'].astype(jp.float32),
            max_feasible_stride=plan['max_feasible_stride'], support_margin=plan['support_margin'],
            gait_switch=switched.astype(jp.float32), scheduler_fault=cs.scheduler.fault.astype(jp.float32))
        if self.diagnostics:
            # Same pose/command/candidate XY; GT is metrics-only, never fed back
            # into the LiDAR planner or actor. Compiled out in PPO by default.
            sensed = self._candidates(result.data, result.info, plan['stride'], plan['period'], mode=plan['mode'])
            oracle = self._candidates(result.data, result.info, plan['stride'], plan['period'],
                                      mode=plan['mode'], privileged=True)
            truth, predicted = oracle['safe'], sensed['safe']
            both = (oracle['reference_index'] >= 0) & (sensed['reference_index'] >= 0)
            legs = jp.arange(6)
            reference_error = jp.linalg.norm(
                oracle['world'][legs, jp.maximum(oracle['reference_index'], 0)]-
                sensed['world'][legs, jp.maximum(sensed['reference_index'], 0)], axis=-1)
            edge_truth, edge_pred = oracle['edge_rejected'], sensed['edge_rejected']
            result.metrics.update(
                oracle_safe_recall=jp.sum(truth & predicted)/jp.maximum(jp.sum(truth), 1),
                oracle_false_safe=jp.sum(predicted & ~truth)/jp.maximum(jp.sum(predicted), 1),
                oracle_unknown_fraction=jp.mean((~sensed['coverage_ok']).astype(jp.float32)),
                oracle_foothold_error_m=jp.sum(jp.where(both, reference_error, 0.))/jp.maximum(jp.sum(both), 1),
                oracle_compared=jp.sum(both).astype(jp.float32),
                oracle_edge_recall=jp.sum(edge_truth & edge_pred)/jp.maximum(jp.sum(edge_truth), 1),
                oracle_edge_precision=jp.sum(edge_truth & edge_pred)/jp.maximum(jp.sum(edge_pred), 1))
        penalty = self.dt*(.5*result.info['projection_m']/.04 + .2*slip +
                          .5*jp.mean(plan['residual_rejected'].astype(jp.float32))) + 2.*landing_error
        penalty += self.dt*.005*(cs.scheduler.mode == scheduler.WAVE) + .01*switched
        # Dimensionless efficiency proxies, not measured servo electrical power.
        joint_speed = jp.mean((result.data.qvel[self._joint_qvel_ids]/5.)**2)
        vertical_speed = (result.data.qvel[2]/.2)**2
        foot_travel = jp.mean(jp.linalg.norm(feet-previous_feet, axis=-1))/self.dt
        excess_clearance = jp.mean(jp.maximum(cs.swing_clearance-plan['selected_required']-.06, 0.))
        tiny_stride = jp.maximum(1.-cs.stride_scale, 0.) * plan['tripod_feasible'][1]
        penalty += self.dt*(.002*joint_speed + .002*vertical_speed + .003*foot_travel +
                            .02*excess_clearance + .003*tiny_stride)
        result.metrics.update(efficiency_joint_speed=joint_speed, efficiency_vertical_speed=vertical_speed,
            efficiency_foot_travel=foot_travel, efficiency_excess_clearance=excess_clearance,
            efficiency_tiny_stride=tiny_stride)
        return result.replace(reward=result.reward-penalty, obs=self._get_obs(result.data, result.info))

    @property
    def action_size(self):
        return adaptive.ACTION_SIZE

    @property
    def observation_size(self):
        return {'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE}
