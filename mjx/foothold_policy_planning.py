"""CPU takeoff planning from LiDAR cells and the archived nominal gait.

No simulator terrain lookup is available here. MuJoCo is used only for foot
kinematics and checking paths against the current/predicted body pose.
"""
import time

import mujoco
import numpy as np

import firmware_mjx_controller as firmware
from foothold_planner import PlannerConfig, Plan, footprint_offsets, plan_foothold
from foothold_preview_runtime import SiteIK


class TakeoffPlanner:
    def __init__(self, env, require_observed=False):
        self.env = env
        self.ik = SiteIK(env.mj_model, np.asarray(env._home_qpos))
        self.cfg = PlannerConfig()
        self.require_observed = require_observed
        self.grid = None
        self.plans = {}

    def terrain_features(self, host, support_height):
        # Same 15 sample positions as training; only observed cells supply height.
        rotation = host.xmat[self.env.mj_model.body('hexapod').id].reshape(3, 3)
        forward = -rotation[:, 1]
        yaw = np.arctan2(forward[1], forward[0])
        c, s = np.cos(yaw), np.sin(yaw)
        xy = np.asarray(self.env._height_samples) @ np.array(((c, s), (-s, c))) + host.qpos[:2]
        if self.grid is None:
            return np.zeros(15, dtype=np.float32)
        z, valid = self.grid.sample(xy, time.monotonic())
        return np.where(valid, z-support_height, 0.).astype(np.float32)

    def proposals(self, host, control, command):
        paths = np.zeros((6, 41, 3), dtype=np.float32)
        modes = np.zeros(6, dtype=np.int32)
        # A scan may arrive mid-swing; decide only near the next phase boundary.
        if bool(control.gait_running) and float(control.phase_time) < float(firmware.GAIT_PHASE_TIME) - 0.021:
            return paths, modes
        now = time.monotonic()
        rotation = host.xmat[self.env.mj_model.body('hexapod').id].reshape(3, 3)
        origin = host.qpos[:3].copy()
        start_world = self.ik.positions(host)
        duration = float(firmware.GAIT_PHASE_TIME)
        vx, wz = command[:2]
        # Bound the body prediction to the explorer's command range.
        vx, wz = np.clip(vx, -.12, .12), np.clip(wz, -.3, .3)
        phase = np.linspace(0., 1., 41)
        predicted = []
        for p in phase:
            c, s = np.cos(p*duration*wz), np.sin(p*duration*wz)
            yaw_rotation = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
            pose = host.qpos.copy()
            pose[:3] += p * duration * vx * -rotation[:, 1]
            mujoco.mju_mat2Quat(pose[3:7], (yaw_rotation @ rotation).reshape(9))
            predicted.append((pose, yaw_rotation @ rotation))
        base = np.asarray(firmware.BASE_FEET)
        twist = np.asarray(control.gait_applied)
        displacement = duration * np.column_stack((
            -twist[0] + twist[3]*base[:, 1],
            -twist[1] - twist[3]*base[:, 0], np.full(6, -twist[2])))
        front = base - .5*displacement
        front[:, 2] -= np.clip(command[2], -.10, .10)
        posture = np.asarray(firmware._rotation_matrix(control.posture_command))
        front = front @ posture
        front_model = front[:, (1, 0, 2)].copy()
        front_model[:, 1] *= -1
        nominal = front_model @ predicted[-1][1].T + predicted[-1][0][:3] - (0, 0, .032)
        # Sample the firmware Bezier/radial nominal arc for known-hazard vetoes.
        starts_model = (start_world + (0, 0, .032) - origin) @ rotation
        starts = np.column_stack((-starts_model[:, 1], starts_model[:, 0], starts_model[:, 2]))
        scaled = phase**3 * (10. + phase*(-15. + 6.*phase))
        blend = scaled**2 * (3. - 2.*scaled)
        envelope = 4.*scaled*(1.-scaled)
        angles = np.asarray(firmware.LEG_ANGLES)
        arc = np.column_stack((float(firmware.SWING_RADIAL_OFFSET)*np.cos(angles),
                               float(firmware.SWING_RADIAL_OFFSET)*np.sin(angles),
                               np.full(6, float(firmware.SWING_HEIGHT))))
        nominal_path = starts[None] + blend[:, None, None]*(front-starts)[None] + envelope[:, None, None]*arc[None]
        for j, (pose, rot) in enumerate(predicted):
            model = nominal_path[j][:, (1, 0, 2)].copy()
            model[:, 1] *= -1
            nominal_path[j] = model @ rot.T + pose[:3] - (0, 0, .032)
        offsets = footprint_offsets(self.grid.cfg.resolution, self.cfg.support_radius)*self.grid.cfg.resolution if self.grid is not None else None
        self.plans = {}
        for leg in range(6):
            def reachable(point):
                self.ik.home = predicted[-1][0].copy()
                return self.ik.solve(leg, point) is not None

            def path_reachable(path):
                seed = None
                for j, point in enumerate(path):
                    self.ik.home = predicted[j][0].copy()
                    seed = self.ik.solve(leg, point, seed)
                    if seed is None:
                        return False
                return True

            if self.grid is None:
                plan = Plan(nominal[leg], np.empty((0, 3)), np.empty(0, dtype=bool), [],
                            nominal[leg], nominal_path[:, leg], 'nominal_unknown_terrain', 0, 'nominal')
            else:
                plan = plan_foothold(self.grid, start_world[leg], nominal[leg], now, reachable,
                                     self.cfg, allow_unknown_path=not self.require_observed,
                                     path_reachable=path_reachable)
                if plan.path is None:
                    path = nominal_path[:, leg]
                    heights, observed = self.grid.sample(path[:, None, :2] + offsets[None], now)
                    collision = np.any(observed & (heights > path[:, None, 2] + .015))
                    mismatch = np.any(observed[-1] & (np.abs(heights[-1]-path[-1, 2]) > .015))
                    if not observed[-1].all() and not collision and not mismatch and not self.require_observed:
                        plan.selected, plan.path = nominal[leg], path
                        plan.mode, plan.status = 'nominal', 'nominal_unknown_terrain'
                    else:
                        plan.mode, plan.status = 'hold', 'hold_observed_hazard_or_unreachable'
            if self.require_observed and self.grid is None:
                plan.mode, plan.status = 'hold', 'hold_requires_observation'
                plan.selected, plan.path = None, None
            self.plans[leg] = plan
            if plan.path is not None:
                paths[leg] = plan.path
                modes[leg] = 1 if plan.mode.startswith('geometric') else 0
            else:
                modes[leg] = 2
        return paths, modes
