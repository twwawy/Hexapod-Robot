#!/usr/bin/env python3
"""Verify the batched Torch controller against the frozen MJX tick trace."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def _load_controller(repo_root: Path):
    path = repo_root / (
        "isaaclab_hexapod/source/hexapod_isaaclab/hexapod_isaaclab/"
        "controllers/firmware_controller_torch.py"
    )
    spec = importlib.util.spec_from_file_location("hexapod_firmware_torch", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _quat_to_rpy(quaternion: np.ndarray) -> np.ndarray:
    quaternion = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-8)
    w, x, y, z = quaternion
    return np.asarray(
        (
            np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0)),
            np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)),
        ),
        dtype=np.float32,
    )


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dtype=np.float32,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--atol", type=float, default=2.5e-5)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    firmware = _load_controller(repo_root)
    golden_path = repo_root / "mjx/golden/isaac_contract_v1_flat_seed0.npz"
    trace = np.load(golden_path)
    steps = min(args.steps, trace["t"].shape[0])
    home_quaternion = trace["pre/root_pose"][0, 3:7]
    home_conjugate = home_quaternion * np.asarray((1.0, -1.0, -1.0, -1.0), np.float32)
    state = firmware.initial_state(1, "cpu")
    max_abs = {
        "model_joint_targets": 0.0,
        "foot_targets_body": 0.0,
        "applied_twist": 0.0,
        "gait_progress": 0.0,
        "residual_filter": 0.0,
    }

    for policy_step in range(steps):
        # The source uses float32 ``policy_steps * 0.02 >= 1.0``.  Its first
        # active capture is index 50; using the rounded exported ``t`` at
        # index 49 would activate one policy step too early.
        command = trace["command"][policy_step] if policy_step >= 50 else np.zeros(5, np.float32)
        root_pose = trace["pre/root_pose"][policy_step]
        attitude = _quat_to_rpy(_quat_multiply(root_pose[3:7], home_conjugate))
        common = dict(
            target_velocity=torch.from_numpy(command[:2]).view(1, 2),
            body_position_world=torch.from_numpy(root_pose[:3]).view(1, 3),
            attitude_rpy=torch.from_numpy(attitude).view(1, 3),
            contacts=torch.from_numpy(trace["pre/contact"][policy_step]).view(1, 6),
            policy_action=torch.from_numpy(trace["action_applied"][policy_step]).view(1, 18),
            pitch_ff=torch.zeros(1),
            roll_cmd=torch.tensor([command[4]]),
            pitch_cmd=torch.tensor([command[3]]),
            height_offset=torch.tensor([command[2]]),
            swing_boost=torch.zeros(1),
        )
        for tick in range(4):
            state, output = firmware.step(state, **common)
            comparisons = {
                "model_joint_targets": (output.model_joint_targets, "firmware_tick/output/model_joint_targets"),
                "foot_targets_body": (output.foot_targets_body, "firmware_tick/output/foot_targets_body"),
                "applied_twist": (output.applied_twist, "firmware_tick/output/applied_twist"),
                "gait_progress": (output.gait_progress, "firmware_tick/output/gait_progress"),
                "residual_filter": (state.residual_filter, "firmware_tick/state/residual_filter"),
            }
            for name, (actual, key) in comparisons.items():
                expected = torch.from_numpy(trace[key][policy_step, tick]).to(actual.dtype)
                error = float((actual[0] - expected).abs().max())
                max_abs[name] = max(max_abs[name], error)
            for actual, key in (
                (output.gait_state, "firmware_tick/output/gait_state"),
                (output.ik_valid, "firmware_tick/output/ik_valid"),
                (output.policy_valid, "firmware_tick/output/policy_valid"),
                (output.foot_limited, "firmware_tick/output/foot_limited"),
            ):
                expected = torch.from_numpy(trace[key][policy_step, tick])
                if not torch.equal(actual[0].cpu(), expected):
                    raise AssertionError(f"boolean/integer parity failed for {key} at {policy_step}:{tick}")

    report = {
        "schema_version": 1,
        "golden": str(golden_path),
        "policy_steps": steps,
        "firmware_ticks": steps * 4,
        "absolute_tolerance": args.atol,
        "max_absolute_error": max_abs,
        "passed": max(max_abs.values()) <= args.atol,
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit("Torch firmware contract differs from the frozen MJX trace")


if __name__ == "__main__":
    main()
