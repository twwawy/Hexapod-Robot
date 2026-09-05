#!/usr/bin/env python3
"""Keyboard pose navigation, rolling LiDAR map and six-leg foothold inspection.

Kinematic preview only: no physics stepping, learned policy or hardware commands.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import queue
import time
import traceback

import glfw
import mujoco
import numpy as np

from foothold_preview_runtime import PerceptionWorker, add_sphere, add_line, add_height_cell
from foothold_preview_runtime import draw_mid360_fov, draw_robot_skeleton
from foothold_preview_scene import build_scene
from tripod_controller import LEG_PREFIXES
from continuous_foothold_gait import ContinuousFootholdGait


class Preview:
    def __init__(self, args):
        self.args = args
        scene, self.manifest = build_scene(args.output, args.revision, args.terrain,
                                          args.lidar_frame, args.lidar_tf_source)
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)
        self.initial_home = self.data.qpos.copy()
        self.rest = self.initial_home.copy()
        self.gait = ContinuousFootholdGait(self.model, self.data, args.gait_swing_duration)
        self.in_place = False
        self.target_height = float(self.rest[2])
        self.worker = PerceptionWorker(self.model, args, self.rest)
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='foothold-perception')
        self.future, self.result = None, None
        self.worker_failed = False
        self.last_submit = self.last_report = -np.inf
        self.save_requested = False
        self.generation = 0
        self.keys = queue.SimpleQueue()
        self.leg, self.yaw = 0, 0.0
        self.command = np.zeros(3)
        self.show_map, self.show_points, self.scanning = True, False, True
        self.show_fov = True
        self.follow, self.top_view = True, False
        self.status = 'Arrows start continuous tripod swing; Space finishes swing and stops'

    def stop_swing(self):
        """Explicit home reset only; velocity changes never reset joint poses."""
        self.in_place = False
        self.data.qpos[:] = self.rest
        mujoco.mj_forward(self.model, self.data)
        self.gait.reset()

    def handle_keys(self, viewer):
        while not self.keys.empty():
            key = self.keys.get()
            if key in (glfw.KEY_UP, ord('W'), glfw.KEY_DOWN, ord('S')):
                delta = 0.04 if key in (glfw.KEY_UP, ord('W')) else -0.04
                self.command[0] = np.clip(self.command[0] + delta, -0.16, 0.16)
            elif key in (glfw.KEY_LEFT, ord('Q'), glfw.KEY_RIGHT, ord('E')):
                delta = 0.15 if key in (glfw.KEY_LEFT, ord('Q')) else -0.15
                self.command[2] = np.clip(self.command[2] + delta, -0.6, 0.6)
            elif key in (ord('A'), ord('D')):
                self.command[1] = np.clip(self.command[1] + (0.04 if key == ord('A') else -0.04), -0.12, 0.12)
            elif key == glfw.KEY_SPACE:
                self.command[:] = 0
                self.in_place = False
                self.status = 'Stop requested: finish the current swing, then hold stance'
                self.last_submit = -np.inf
            elif key == glfw.KEY_ENTER:
                self.command[:] = 0
                self.in_place = not self.in_place
                self.gait.retry()
                self.status = 'In-place continuous swing' if self.in_place else 'Stop after current touchdown'
            elif ord('1') <= key <= ord('6'):
                self.leg = key - ord('1')
            elif key in (glfw.KEY_PAGE_UP, glfw.KEY_PAGE_DOWN):
                self.target_height = float(np.clip(
                    self.target_height + (0.02 if key == glfw.KEY_PAGE_UP else -0.02),
                    self.initial_home[2] - 0.12, self.initial_home[2] + 0.6))
                self.gait.retry()
                self.last_submit = -np.inf
            elif key == ord('H'):
                self.command[:] = 0
                self.yaw = 0
                self.rest = self.initial_home.copy()
                self.target_height = float(self.rest[2])
                self.stop_swing()
                self.generation += 1
                self.result = None
                self.last_submit = -np.inf
            elif key == ord('C'):
                self.command[:] = 0
                self.in_place = False
                self.generation += 1
                self.result = None
                self.last_submit = -np.inf
            elif key in (ord('R'), ord('P')):
                self.worker_failed = False
                self.gait.retry()
                self.last_submit = -np.inf
                self.save_requested |= key == ord('P')
            elif key == ord('M'):
                self.show_map = not self.show_map
            elif key == ord('L'):
                self.show_points = not self.show_points
            elif key == ord('G'):
                self.show_fov = not self.show_fov
            elif key == ord('K'):
                self.scanning = not self.scanning
                self.last_submit = -np.inf
            elif key == ord('T'):
                self.top_view = not self.top_view
                viewer.cam.elevation = -89 if self.top_view else -50
            elif key == ord('F'):
                self.follow = not self.follow
                if self.follow:
                    viewer.cam.distance = 3.2
            elif key == ord('V'):
                self.follow = False
                viewer.cam.lookat[:] = (0, 0, 0)
                viewer.cam.distance, viewer.cam.elevation = 12, -75

    def navigate(self, dt):
        if self.worker_failed:
            return
        self.gait.tick(self.result, self.command, dt, self.target_height,
                       in_place=self.in_place, allow_unknown=not self.args.require_observed)
        self.rest = self.data.qpos.copy()
        w, x, y, z = self.rest[3:7]
        self.yaw = np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))-np.pi/2
        self.status = self.gait.status

    def update_perception(self, now):
        if self.future is not None and self.future.done():
            try:
                result = self.future.result()
                if result['generation'] == self.generation:
                    self.result = result
                    if now - self.last_report > 2 or self.save_requested:
                        self.save_result(now, self.save_requested)
                        self.last_report, self.save_requested = now, False
            except Exception:
                self.command[:] = 0
                self.in_place = False
                self.worker_failed = True
                self.status = 'Perception error: see terminal traceback; R retries'
                traceback.print_exc()
            self.future = None
        if (self.future is None and not self.worker_failed
                and now - self.last_submit >= 1 / self.args.scan_hz):
            self.future = self.executor.submit(self.worker.compute, self.rest.copy(), self.command.copy(),
                                               self.generation, self.scanning)
            self.last_submit = now

    def save_result(self, now, save_map=False):
        result = self.result
        summary = dict(pose=result['pose'][:7].tolist(), stats=result['stats'],
                       scan_ms=result['scan_ms'], plan_ms=result['plan_ms'],
                       near_valid_fraction=result['near_valid_fraction'],
                       snapshot_age_s=now - result['stamp'], physics_stepped=False, rl_residual=0,
                       gait_status=self.gait.status, active_tripod=list(self.gait.plans), legs={})
        for index, (leg, draft) in enumerate(zip(LEG_PREFIXES, result['plans'])):
            plan = self.gait.plans.get(index, draft)
            summary['legs'][leg] = dict(mode=plan.mode, status=plan.status, nominal=plan.nominal.tolist(),
                                       selected=None if plan.selected is None else plan.selected.tolist(),
                                       geometric_candidate=None if plan.geometric_candidate is None else plan.geometric_candidate.tolist(),
                                       reasons=dict(Counter(plan.reasons)))
        (self.args.output / 'latest_plan.json').write_text(json.dumps(summary, indent=2) + '\n')
        if save_map:
            grid = result['grid']
            np.savez_compressed(self.args.output / 'latest_map.npz', height=grid.height,
                                valid=grid.valid(now), timestamp=grid.timestamp, center_odom=grid.center,
                                vertical_range=grid.vertical_range, points_odom=result['points'],
                                resolution=grid.cfg.resolution, half_extent=grid.cfg.half_extent,
                                snapshot_time=now)
            print(f'Saved six-leg plan and map: {self.args.output}', flush=True)

    def draw(self, viewer, now):
        scene = viewer.user_scn
        scene.ngeom = 0
        if self.args.robot_display == 'skeleton':
            draw_robot_skeleton(scene, self.data, self.gait.ik.joints, self.gait.ik.site_ids)
        sensor = self.model.site('lidar_origin').id
        origin = self.data.site_xpos[sensor]
        rotation = self.data.site_xmat[sensor].reshape(3, 3)
        if self.show_fov:
            draw_mid360_fov(scene, origin, rotation, self.args.fov_display_radius)
        result, rows = self.result, []
        # Targets get visual capacity first, before potentially thousands of map cells.
        if result is not None:
            for i, draft in enumerate(result['plans']):
                plan = self.gait.plans.get(i, draft)
                focus = i == self.leg
                nominal, selected = plan.nominal, plan.selected
                add_sphere(scene, nominal + (0, 0, 0.016), 0.010, (1, 0.8, 0.1, 1))
                if selected is not None:
                    color = ((1, 0.45, 0.06, 1) if plan.mode == 'nominal' else
                             (0.1, 0.85, 1, 1) if plan.mode == 'geometric_partial' else (0.2, 0.3, 1, 1))
                    marker = selected + (0, 0, 0.11)
                    add_sphere(scene, selected + (0, 0, 0.012), 0.021, color)
                    add_line(scene, selected, marker, color, 0.005)
                    add_sphere(scene, marker, 0.030 if focus else 0.025, color, LEG_PREFIXES[i])
                    add_line(scene, nominal + (0, 0, 0.018), selected + (0, 0, 0.018), color)
                else:
                    center = nominal + (0, 0, 0.11)
                    add_line(scene, center + (-0.025, -0.025, 0), center + (0.025, 0.025, 0), (1, 0.1, 0.1, 1))
                    add_line(scene, center + (-0.025, 0.025, 0), center + (0.025, -0.025, 0), (1, 0.1, 0.1, 1))
                candidate = plan.geometric_candidate
                if candidate is not None and (selected is None or np.linalg.norm(candidate - selected) > 0.005):
                    add_sphere(scene, candidate + (0, 0, 0.04), 0.014, (0.15, 0.4, 1, 0.8))
                if plan.path is not None:
                    color = (1, 0.45, 0.05, 1) if plan.mode == 'nominal' else (0.85, 0.15, 1, 1)
                    for a, b in zip(plan.path[:-1:2], plan.path[2::2]):
                        add_line(scene, a, b, color, 0.004 if i in self.gait.plans else 0.002)
                if focus:
                    for point, accepted in zip(plan.candidates, plan.accepted):
                        add_sphere(scene, point + (0, 0, 0.008), 0.005,
                                   (0.1, 1, 0.2, 0.35) if accepted else (1, 0.15, 0.15, 0.2))
                rows.append(f'{">" if focus else " "} {LEG_PREFIXES[i]}: {plan.mode} | {plan.status}')
            if self.show_map:
                grid = result['grid']
                valid = grid.valid(now)
                xy, z = grid.centers()[valid], grid.height[valid]
                distance = np.linalg.norm(xy - self.rest[:2], axis=-1)
                cell_ids = np.argwhere(valid)
                keep = (distance < 1) | ((cell_ids[:, 0] % 3 == 0) & (cell_ids[:, 1] % 3 == 0))
                ids = np.flatnonzero(keep)
                ids = ids[np.argsort(distance[ids])[:self.args.map_draw_budget]]
                for index in ids:
                    value = np.clip(z[index] / 0.30, 0, 1)
                    color = (0.15 + 0.85 * value, 0.8 - 0.4 * value, 0.7 * (1 - value), self.args.map_alpha)
                    add_height_cell(scene, np.r_[xy[index], z[index]], grid.cfg.resolution, color)
            if self.show_points:
                points = result['points']
                for point in points[::max(1, len(points) // 1500)]:
                    add_sphere(scene, point + (0, 0, 0.006), 0.003, (0.7, 0.9, 1, self.args.lidar_point_alpha))
        sensor = self.model.site('lidar_origin').id
        origin = self.data.site_xpos[sensor]
        rotation = self.data.site_xmat[sensor].reshape(3, 3)
        for i, color in enumerate(((1, 0.1, 0.1, 1), (0.1, 1, 0.1, 1), (0.1, 0.2, 1, 1))):
            add_line(scene, origin, origin + 0.15 * rotation[:, i], color, 0.003)
        forward = np.array((np.cos(self.yaw), np.sin(self.yaw), 0))
        add_line(scene, self.rest[:3], self.rest[:3] + forward * 0.28, (1, 0.8, 0.1, 1), 0.004)
        if self.follow:
            viewer.cam.lookat[:] = self.rest[:3] + 0.5 * forward
        diagnostics = 'Waiting for first LiDAR/map/planning result'
        if result is not None:
            stats = result['stats']
            diagnostics = (f'Ground hits {stats["ground_hits"]} / rays {stats["rays"]} | body hits {stats["body_hits"]}\n'
                           f'Near-foot coverage {result["near_valid_fraction"]:.0%} | snapshot age {now-result["stamp"]:.1f}s\n'
                           f'Scan {result["scan_ms"]:.0f} ms | six-leg plan {result["plan_ms"]:.0f} ms')
        text = (f'FOOTHOLD EXPLORER | {self.manifest["commit"][:7]} | KINEMATIC / RL=0 | {self.args.robot_display.upper()}\n'
                f'{self.status}\n'
                f'x={self.rest[0]:.2f} y={self.rest[1]:.2f} yaw={np.rad2deg(self.yaw):.0f}deg '
                f'| vx={self.command[0]:+.2f} vy={self.command[1]:+.2f} wz={self.command[2]:+.2f}\n'
                f'{diagnostics}\n' + '\n'.join(rows) + '\n'
                'Arrows: forward/back + turn | A/D strafe | SPACE STOP | PgUp/PgDn height\n'
                'Continuous tripod | 1..6 detail only | Enter in-place toggle | R retry | P save | H home | C clear map\n'
                'M height tiles | L raw points | G MID360 FOV | K scan | F follow | T top | V course\n'
                'MID360 FOV: H360 V[-7,+52]deg | wires: angular boundary, not returns\n'
                'Orange: unseen nominal | Blue: geometric | Cyan: endpoint known, path partly unknown\n'
                'Height: green low -> yellow/red high | Red X: no executable target')
        return [(mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT, text, '')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='origin/main')
    parser.add_argument('--terrain', choices=('flat', 'steps'), default='steps')
    parser.add_argument('--robot-display', choices=('skeleton', 'mesh'), default='skeleton',
                        help='robot rendering only; CAD geometry is retained for LiDAR occlusion')
    parser.add_argument('--lidar-frame', default='livox_mid_360_2')
    parser.add_argument('--lidar-tf-source', choices=('measured', 'urdf'), default='measured')
    parser.add_argument('--step-length', type=float, default=0.08)
    parser.add_argument('--map-resolution', type=float, default=0.04)
    parser.add_argument('--map-half-extent', type=float, default=4.0)
    parser.add_argument('--map-max-age', type=float, default=60.0)
    parser.add_argument('--map-draw-budget', type=int, default=2500)
    parser.add_argument('--require-observed', action='store_true')
    parser.add_argument('--scan-hz', type=float, default=3.0)
    parser.add_argument('--lidar-range', type=float, default=8.0)
    parser.add_argument('--fov-display-radius', type=float, default=1.2,
                        help='FOV wire radius [m], independent of raycast range; G toggles wires')
    parser.add_argument('--azimuth-samples', type=int, default=720)
    parser.add_argument('--elevation-samples', type=int, default=64)
    parser.add_argument('--swing-duration', '--gait-swing-duration', dest='gait_swing_duration',
                        type=float, default=0.8, help='duration of each alternating tripod swing [s]')
    parser.add_argument('--map-alpha', type=float, default=0.16)
    parser.add_argument('--lidar-point-alpha', type=float, default=0.22)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'generated/foothold_preview')
    args = parser.parse_args()
    if (min(args.scan_hz, args.azimuth_samples, args.gait_swing_duration, args.map_resolution,
            args.map_half_extent, args.map_max_age, args.lidar_range, args.map_draw_budget,
            args.fov_display_radius) <= 0
            or args.elevation_samples < 2):
        parser.error('rates, distances and budgets must be positive; elevation-samples must be >=2')
    if not (0 <= args.map_alpha <= 1 and 0 <= args.lidar_point_alpha <= 1):
        parser.error('map-alpha and lidar-point-alpha must be between 0 and 1')
    args.output = args.output.resolve()
    preview = Preview(args)
    import mujoco.viewer
    try:
        with mujoco.viewer.launch_passive(preview.model, preview.data,
                                          key_callback=preview.keys.put) as viewer:
            with viewer.lock():
                # Hide CAD in the renderer only. The worker keeps its own ray group mask,
                # so invisible chassis/legs still occlude LiDAR as before.
                viewer.opt.geomgroup[1] = int(args.robot_display == 'mesh')
                viewer.opt.geomgroup[5] = int(args.robot_display == 'mesh')
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.2, 135, -50
            print('Arrow keys start continuous tripod motion; SPACE finishes swing and stops. No physics / hardware control.', flush=True)
            print(f'Artifacts: {args.output}', flush=True)
            previous = time.monotonic()
            while viewer.is_running():
                current = time.monotonic()
                dt = min(current - previous, 0.05)
                preview.update_perception(current)
                with viewer.lock():
                    preview.handle_keys(viewer)
                    preview.navigate(dt)
                    texts = preview.draw(viewer, current)
                viewer.set_texts(texts)
                viewer.sync()
                previous = current
                time.sleep(0.01)
    finally:
        preview.executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
