"""Alternating tripod kinematic preview with odom stance anchors.

This is a visualization controller, not the STM32 contact/dynamics controller.
Swing targets are latched per phase; root and all six IK targets commit together.
"""
from __future__ import annotations

import time
import numpy as np
import mujoco

from foothold_planner import (
    PlannerConfig, footprint_offsets, plan_with_nominal_fallback,
)
from foothold_preview_runtime import SiteIK
from tripod_controller import LEG_PREFIXES


class ContinuousFootholdGait:
    GROUPS = ((0, 2, 4), (1, 3, 5))  # RF/RB/LM then RM/LF/LB

    def __init__(self, model, data, duration):
        self.model, self.data, self.duration = model, data, duration
        self.ik = SiteIK(model, data.qpos)
        self.cfg = PlannerConfig(max_ik_candidates=3)
        self.neutral_qpos = data.qpos.copy()
        root = data.xpos[model.body('hexapod').id]
        rotation = data.xmat[model.body('hexapod').id].reshape(3, 3)
        self.neutral_feet_body = (data.site_xpos[self.ik.site_ids] - root) @ rotation
        self.reset()

    def reset(self):
        mujoco.mj_forward(self.model, self.data)
        self.anchors = self.data.site_xpos[self.ik.site_ids].copy()
        self.group_index = 0
        self.plans = {}
        self.elapsed = 0.0
        self.last_attempt = -np.inf
        self.blocked = False
        self.rate_scale = 1.0
        self.status = 'Ready: arrows start alternating tripod swing'

    def root_pose(self, command, dt, target_height):
        pose = self.data.qpos.copy()
        w, x, y, z = pose[3:7]
        yaw = np.arctan2(2 * (w*z+x*y), 1 - 2 * (y*y+z*z)) - np.pi / 2
        yaw += command[2] * dt
        c, s = np.cos(yaw), np.sin(yaw)
        pose[:2] += dt * np.array((c*command[0]-s*command[1], s*command[0]+c*command[1]))
        pose[:2] = np.clip(pose[:2], -5.5, 5.5)
        pose[2] += np.clip(target_height-pose[2], -0.04*dt, 0.04*dt)
        angle = yaw + np.pi/2
        pose[3:7] = (np.cos(angle/2), 0, 0, np.sin(angle/2))
        return pose

    def start_phase(self, result, command, now, target_height):
        if result is None or now-result['stamp'] > 3.0:
            self.status = 'Waiting for a recent map; body held'
            return False
        self.last_attempt = now
        # Nominal positions are around the neutral stance, not accumulated offsets
        # from the previous touchdown. Predict the root at the end of this swing.
        future = self.root_pose(command, self.duration, target_height)
        a = 2 * np.arctan2(future[6], future[3])
        rotation = np.array(((np.cos(a), -np.sin(a), 0),
                             (np.sin(a), np.cos(a), 0), (0, 0, 1)))
        nominal = self.neutral_feet_body @ rotation.T + future[:3]
        nominal[:, 2] = self.anchors[:, 2]  # unknown terrain continues support height
        planned = {}
        self.ik.home = self.data.qpos.copy()
        for leg in self.GROUPS[self.group_index]:
            plan = plan_with_nominal_fallback(
                result['grid'], self.anchors[leg], nominal[leg], now,
                lambda point, leg=leg: self.ik.solve(leg, point) is not None,
                self.cfg, allow_unknown=self.allow_unknown)
            if plan.path is None:
                self.status = f'Body held: {LEG_PREFIXES[leg]} {plan.status}'
                return False
            planned[leg] = plan
        self.plans = planned
        self.elapsed, self.blocked = 0.0, False
        return True

    def tick(self, result, command, dt, target_height, in_place=False, allow_unknown=True):
        now = time.monotonic()
        motion_dt = dt*self.rate_scale
        self.allow_unknown = allow_unknown
        requested = in_place or np.max(np.abs(command)) > 1e-5
        if self.blocked:
            return
        if not self.plans and requested and now-self.last_attempt >= 0.5:
            self.start_phase(result, command, now, target_height)
        if not self.plans:
            # Allow height adjustment while stationary, with stance feet anchored.
            if abs(target_height-self.data.qpos[2]) < 1e-6:
                return
            next_elapsed = 0.0
            pose = self.root_pose(np.zeros(3), motion_dt, target_height)
        else:
            next_elapsed = min(self.duration, self.elapsed + motion_dt)
            pose = self.root_pose(command, motion_dt, target_height)
        targets = self.anchors.copy()
        if self.plans:
            if result is None:
                return
            grid = result['grid']
            offsets = footprint_offsets(grid.cfg.resolution, self.cfg.support_radius)*grid.cfg.resolution
            for leg, plan in self.plans.items():
                heights, seen = grid.sample(plan.path[10:31, None, :2]+offsets[None], now)
                ends, end_seen = grid.sample(plan.selected[:2]+offsets, now)
                conflict = np.any(seen & (heights > plan.path[10:31, None, 2]-0.005))
                mismatch = np.any(end_seen & (np.abs(ends-plan.selected[2]) > 0.025))
                expired = ((plan.mode == 'geometric' and not seen.all()) or
                           (plan.mode.startswith('geometric') and not end_seen.all()))
                if conflict or mismatch or expired:
                    self.blocked = True
                    self.status = f'Held mid-swing: {LEG_PREFIXES[leg]} terrain changed; R retries, H resets'
                    return
                position = next_elapsed/self.duration*(len(plan.path)-1)
                index = min(int(position), len(plan.path)-2)
                blend = position-index
                targets[leg] = (1-blend)*plan.path[index]+blend*plan.path[index+1]
        self.ik.home = pose.copy()
        desired_pose = pose.copy()
        for leg in range(6):
            qids = self.model.jnt_qposadr[self.ik.joints[leg]]
            desired = self.ik.solve(leg, targets[leg], self.data.qpos[qids])
            if desired is None:
                self.blocked = True
                self.status = f'Held: {LEG_PREFIXES[leg]} IK; body and phase unchanged; R retries, H resets'
                return
            # Slow the entire kinematic motion rather than moving the base while
            # lagging joints drag stance feet. The next tick retries a smaller step.
            if np.max(np.abs(desired-self.data.qpos[qids])) > np.deg2rad(315.8)*dt:
                self.status = 'Joint-rate limit: slowing body and swing together'
                self.rate_scale = max(0.05, getattr(self, 'rate_scale', 1.0)*0.5)
                return
            desired_pose[qids] = desired
        self.data.qpos[:] = desired_pose
        mujoco.mj_forward(self.model, self.data)
        self.elapsed = next_elapsed
        self.rate_scale = min(1.0, getattr(self, 'rate_scale', 1.0)*1.02)
        if self.plans:
            legs = '/'.join(LEG_PREFIXES[i] for i in self.plans)
            self.status = f'Continuous tripod {legs}: {self.elapsed/self.duration:.0%}'
            if self.elapsed >= self.duration:
                for leg, plan in self.plans.items():
                    self.anchors[leg] = plan.selected.copy()
                self.plans = {}
                self.group_index = 1-self.group_index
                self.status = 'Touchdown; preparing next tripod' if requested else 'Stopped after touchdown'

    def retry(self):
        self.blocked = False
        self.last_attempt = -np.inf
