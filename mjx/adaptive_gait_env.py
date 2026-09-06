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
from adaptive_gait_perception import AngularLidar, SENSOR_PERIOD, MAX_AGE, initial_map, sample
from rough_terrain_env import (
    HexapodRoughTerrainEnv, MODEL_FORWARD, MODEL_LATERAL,
    _quat_rotate_inverse, default_config,
)
from terrain_curriculum import terrain_level as terrain_spec
from adaptive_foothold_estimator import (
    CANDIDATE_COUNT, CANDIDATE_OFFSETS, CANDIDATE_FEATURES, RESIDUAL_RADIUS,
    FOOT_RADIUS, evaluate_candidates,
)

OBSERVATION_CONTRACT = 'adaptive_reference_candidates25_proprio_23_v2'
REWARD_CONTRACT = 'adaptive_command_progress_touchdown_v1'
PROPRIO_SIZE = 155
ACTOR_SIZE = PROPRIO_SIZE + 6*CANDIDATE_COUNT*len(CANDIDATE_FEATURES)
CRITIC_SIZE = ACTOR_SIZE + 6*CANDIDATE_COUNT*2 + 15


class AdaptiveGaitEnv(HexapodRoughTerrainEnv):
    def __init__(self, *, terrain_level=0, perception='lidar', config=None,
                 azimuths=90, elevations=8, dropout=.05, noise=.005):
        if perception not in ('lidar', 'teacher', 'blind', 'oracle'):
            raise ValueError('perception must be lidar, teacher, blind or debug-only oracle')
        if terrain_spec(terrain_level).kind == 'rough':
            raise ValueError('MJX ray does not support hfields: use level 0, 3, 4 or 5..16')
        self.perception = perception
        config = default_config() if config is None else config
        # Posture is owned by the parameter policy, not a terrain-kind sampler.
        config.command.height_min = 0.
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
        info['foothold_plan'] = self._landing_plan(data, info, jp.zeros(adaptive.ACTION_SIZE))
        return info

    def _query(self, grid, xy, now, *, privileged=False):
        height, known, age, spread = sample(grid, xy, now)
        if self.perception == 'blind':
            known = jp.zeros_like(known)
        if privileged or self.perception in ('teacher', 'oracle'):
            height = self._terrain_height(xy.reshape(-1, 2)).reshape(xy.shape[:-1])
            spread = jp.zeros_like(height)
        if self.perception == 'oracle':
            known, age = jp.ones_like(known), jp.zeros_like(age)
        return height, known, age, spread

    def _candidates(self, data, info, stride=1., period=.5, *, lift=None, privileged=False):
        lift = jp.zeros(6) if lift is None else lift
        rotation = data.xmat[self._root_id]
        cs = info['controller_state']
        # The same speed/phase relationship drives stance and predicts landing.
        active_ratio = cs.stride_scale/(cs.phase_duration/.5)
        vx = jp.clip(cs.gait_applied[0]*(stride/(period/.5))/active_ratio,
                     -fw.MAX_LINEAR_SPEED, fw.MAX_LINEAR_SPEED)
        vy, wz = cs.gait_applied[1], cs.gait_applied[3]
        displacement = period*jp.stack((-vx+wz*fw.BASE_FEET[:, 1],
                                        -vy-wz*fw.BASE_FEET[:, 0], jp.zeros(6)), axis=-1)
        front = fw.BASE_FEET - .5*displacement
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
        return evaluate_candidates(self, data, info, xy, nominal, basis, lift, privileged=privileged)

    def _landing_plan(self, data, info, action):
        xy_bias, lift, posture, stride, period = adaptive.decode(action)
        baseline = self._candidates(data, info)
        # Whole-body parameters still require usable references for all legs.
        global_known = jp.all(jp.any(baseline['safe'], axis=1))
        stride, period = jp.where(global_known, stride, 1.), jp.where(global_known, period, .5)
        changed = self._candidates(data, info, stride, period)
        global_known &= jp.all(jp.any(changed['safe'], axis=1))
        reference_plan = jax.tree_util.tree_map(lambda new, old: jp.where(global_known, new, old), changed, baseline)
        stride, period = jp.where(global_known, stride, 1.), jp.where(global_known, period, .5)
        # p_ref is chosen BEFORE RL XY/lift; search and residual are separate.
        reference_index = reference_plan['reference_index']
        leg = jp.arange(6)
        ref = reference_plan['world'][leg, jp.maximum(reference_index, 0)]
        ref = jp.where((reference_index >= 0)[:, None], ref,
                       reference_plan['nominal'].at[:, 2].add(-FOOT_RADIUS))
        requested = ref[:, :2] + xy_bias @ reference_plan['basis']
        refined = self._candidates(data, info, stride, period, lift=lift)
        relative_xy = (refined['xy']-ref[:, None, :2]) @ jp.linalg.inv(reference_plan['basis'])
        bounded = jp.all(jp.abs(relative_xy) <= RESIDUAL_RADIUS+1e-6, axis=-1)
        eligible = refined['safe'] & bounded & (reference_index >= 0)[:, None]
        distance = jp.linalg.norm(refined['xy']-requested[:, None, :], axis=-1)
        residual_ok = jp.any(eligible, axis=1)
        selected = jp.where(residual_ok, jp.argmin(jp.where(eligible, distance, jp.inf), axis=1), reference_index)
        # If the lift request invalidates every nearby option, use the neutral
        # reference path. A rejected residual cannot erase a safe reference.
        plan = dict(refined)
        for name in ('world', 'pre', 'clearance', 'required'):
            values = refined[name][leg, jp.maximum(selected, 0)]
            fallback = reference_plan[name][leg, jp.maximum(reference_index, 0)]
            mask = residual_ok
            while mask.ndim < values.ndim:
                mask = mask[..., None]
            plan['selected_'+name] = jp.where(mask, values, fallback)
        known = selected >= 0
        fallback_mask = (~residual_ok)[:, None] & (jp.arange(CANDIDATE_COUNT)[None, :] == reference_index[:, None])
        for name in ('safe', 'ik_ok', 'path_ok', 'status'):
            plan[name] = jp.where(fallback_mask, reference_plan[name], plan[name])
        observed_hazard = jp.any(reference_plan['unsafe'] | reference_plan['terrain_ok'], axis=1)
        proposal_safe = known | ~observed_hazard
        # Partial observations alone are not filled with GT or treated as flat.
        plan['selected_world'] = jp.where(known[:, None], plan['selected_world'],
                                           reference_plan['nominal'].at[:, 2].add(-FOOT_RADIUS))
        accepted_xy = (plan['selected_world'][:, :2]-ref[:, :2]) @ jp.linalg.inv(reference_plan['basis'])
        accepted = action.at[:12].set(jp.where(known[:, None], jp.clip(accepted_xy/RESIDUAL_RADIUS, -1., 1.), 0.).reshape(-1))
        effective_lift = plan['selected_clearance']-fw.SWING_HEIGHT-plan['selected_required']
        accepted = accepted.at[12:18].set(jp.where(known, jp.clip(effective_lift/.04, -1., 1.), 0.))
        accepted = accepted.at[18:].set(jp.where(global_known, action[18:], 0.))
        forward = ref[:, :2] @ reference_plan['basis'][0]
        centered = forward-jp.mean(forward)
        slope = jp.sum(centered*(ref[:, 2]-jp.mean(ref[:, 2])))/jp.maximum(jp.sum(centered**2), 1e-6)
        posture = posture.at[1].add(jp.clip(-jp.arctan(slope), -jp.deg2rad(12.), jp.deg2rad(12.)))
        posture = jp.where(global_known, posture, jp.zeros(3))
        projection = jp.where(known, jp.linalg.norm(plan['selected_world'][:, :2]-requested, axis=-1), 0.)
        plan.update(time=data.time, reference_index=reference_index, reference_world=ref, selected_index=selected,
                    requested_xy=requested, selected_known=known, proposal_safe=proposal_safe,
                    residual_rejected=(reference_index >= 0) & ~residual_ok,
                    projection=projection, accepted_action=accepted, posture=posture, stride=stride, period=period,
                    reference_safe=reference_plan['safe'], reference_status=reference_plan['status'])
        return plan

    def _prepare_controller_state(self, data, info, action):
        plan = self._landing_plan(data, info, action)
        info['foothold_plan'] = plan
        info['projection_m'] = jp.mean(plan['projection'])
        return info['controller_state']._replace(request=plan['accepted_action'],
            proposal_end=plan['selected_pre'], proposal_world=plan['selected_world'],
            proposal_known=plan['selected_known'], proposal_safe=plan['proposal_safe'],
            proposal_clearance=plan['selected_clearance'], proposal_posture=plan['posture'],
            proposal_stride=plan['stride'], proposal_period=plan['period'], proposal_index=plan['selected_index'],
            root_rotation=data.xmat[self._root_id], root_position=data.qpos[:3])

    def _controller_step(self, controller_state, **kwargs):
        return adaptive.step(controller_state, **kwargs)

    def _reward_posture_target(self, info, controller_state, pitch_ff):
        return controller_state.adapt_posture[:2]

    def _reward_height_command(self, info, controller_state):
        return controller_state.height_applied

    def _get_obs(self, data, info):
        cs, out = info['controller_state'], info['controller_output']
        proprio = jp.concatenate((info['command'],
            _quat_rotate_inverse(data.qpos[3:7], data.qvel[:3]), .2*data.qvel[3:6],
            _quat_rotate_inverse(data.qpos[3:7], jp.array((0., 0., -1.))),
            data.qpos[self._joint_qpos_ids]-self._home_qpos[self._joint_qpos_ids],
            .1*data.qvel[self._joint_qvel_ids], self._feet_controller_body(data).reshape(-1),
            info['contact_state'].astype(jp.float32), out.gait_progress,
            out.gait_state.astype(jp.float32)/2., out.applied_twist,
            out.ik_valid.astype(jp.float32), out.foot_limited.astype(jp.float32),
            cs.accepted_action, info['last_action'], cs.adapt_posture,
            jp.array((cs.stride_scale, cs.phase_duration, cs.plan_rejected.astype(jp.float32), cs.height_applied))))
        assert proprio.shape == (PROPRIO_SIZE,)
        candidates = self._candidates(data, info, privileged=self.perception == 'teacher')
        actor = jp.concatenate((proprio, candidates['features'].reshape(-1)))
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
            'stride_scale', 'phase_duration_s', 'pitch_target_rad', 'foothold_center_known_fraction',
            'foothold_coverage_fraction', 'foothold_terrain_fraction', 'foothold_ik_fraction',
            'foothold_safe_fraction', 'foothold_selected_fraction', 'foothold_residual_rejected_fraction')})
        return state

    def step(self, state, action):
        if action.shape != (adaptive.ACTION_SIZE,):
            raise ValueError('adaptive controller requires 23 actions; stage31 is incompatible')
        # Copy dictionaries because the legacy environment updates them in-place.
        state = state.replace(info=dict(state.info), metrics=dict(state.metrics))
        idle = jp.all(jp.abs(state.info['command'][:2]) < .001)
        state.info['no_progress_steps'] = jp.where(idle, 0, state.info['no_progress_steps'])
        state.info['progress_anchor_potential'] = jp.where(idle,
            state.data.qpos[0] + .5*state.info['support_height'], state.info['progress_anchor_potential'])
        previous_contacts = state.info['contact_state']
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
        penalty = self.dt*(.5*result.info['projection_m']/.04 + .2*slip +
                          .5*jp.mean(plan['residual_rejected'].astype(jp.float32))) + 2.*landing_error
        return result.replace(reward=result.reward-penalty, obs=self._get_obs(result.data, result.info))

    @property
    def action_size(self):
        return adaptive.ACTION_SIZE

    @property
    def observation_size(self):
        return {'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE}
