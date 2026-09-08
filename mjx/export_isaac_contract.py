#!/usr/bin/env python3
"""Export a deterministic 500-step MJX transition trace for Isaac replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jp
import mujoco
import numpy as np

import firmware_mjx_controller as firmware
from rough_terrain_env import (
    ACTION_CONTRACT_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    HexapodRoughTerrainEnv,
    _effective_posture_target,
)


SCHEMA = "hexapod_mjx_transition_v1"
CANONICAL_LEGS = ("RB", "RM", "RF", "LB", "LM", "LF")
POLICY_STEPS = 500
FIRMWARE_TICKS = 4


def _git(args: list[str], root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scripted_actions(steps: int) -> np.ndarray:
    """Low-amplitude bounded input that excites XYZ for every leg."""
    t = np.arange(steps, dtype=np.float32) * np.float32(0.02)
    actions = np.zeros((steps, 18), dtype=np.float32)
    active = t >= np.float32(1.0)
    frequencies = np.linspace(0.17, 0.43, 18, dtype=np.float32)
    phases = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False, dtype=np.float32)
    amplitudes = np.tile(np.array((0.18, 0.14, 0.20), dtype=np.float32), 6)
    values = amplitudes[None, :] * np.sin(
        2.0 * np.float32(np.pi) * t[:, None] * frequencies[None, :]
        + phases[None, :]
    )
    actions[active] = values[active]
    return np.clip(actions, -1.0, 1.0).astype(np.float32)


def _flatten(prefix: str, value: Any, output: dict[str, np.ndarray]) -> None:
    if hasattr(value, "_fields"):
        for field in value._fields:
            _flatten(f"{prefix}/{field}", getattr(value, field), output)
        return
    if isinstance(value, dict):
        for key in sorted(value):
            _flatten(f"{prefix}/{key}", value[key], output)
        return
    array = np.asarray(value)
    if array.dtype == np.float64:
        array = array.astype(np.float32)
    elif np.issubdtype(array.dtype, np.integer) and array.dtype != np.int32:
        array = array.astype(np.int32)
    output[prefix] = array


def _build_capture(env: HexapodRoughTerrainEnv):
    canonical_joint_qpos = jp.asarray(
        [
            env.mj_model.joint(f"{leg}_{joint}").qposadr[0]
            for leg in CANONICAL_LEGS
            for joint in (1, 2, 3)
        ],
        dtype=jp.int32,
    )
    canonical_joint_qvel = jp.asarray(
        [
            env.mj_model.joint(f"{leg}_{joint}").dofadr[0]
            for leg in CANONICAL_LEGS
            for joint in (1, 2, 3)
        ],
        dtype=jp.int32,
    )

    def capture(state, action_requested):
        clipped = jp.clip(action_requested, -1.0, 1.0)
        delay_buffer = jp.concatenate(
            (clipped[None, :], state.info["action_delay_buffer"][:-1]), axis=0
        )
        applied = delay_buffer[state.info["action_delay_ticks"]]
        contacts = env._foot_contacts(state.data)
        attitude = env._relative_attitude(state.data)
        pitch_ff = env._terrain_pitch_ff(
            state.data, state.info["support_height"], state.info["pitch_ff"]
        )
        swing_boost = env._terrain_swing_boost(
            state.data, state.info["support_height"]
        )
        active = state.info["policy_steps"] * env.dt >= env._config.command_delay
        firmware_command = jp.where(active, state.info["command"], jp.zeros(5))

        def tick(controller_state, _):
            next_controller_state, output = firmware.step(
                controller_state,
                target_velocity=firmware_command[:2],
                body_position_world=state.data.qpos[:3],
                attitude_rpy=attitude,
                contacts=contacts,
                policy_action=applied,
                pitch_ff=pitch_ff,
                roll_cmd=firmware_command[4],
                pitch_cmd=firmware_command[3],
                height_offset=firmware_command[2],
                swing_boost=swing_boost,
            )
            return next_controller_state, (next_controller_state, output)

        _, (state_history, output_history) = jax.lax.scan(
            tick, state.info["controller_state"], xs=None, length=FIRMWARE_TICKS
        )
        next_state = env.step(state, action_requested)
        reward_scaled = {
            name: next_state.metrics[f"reward/{name}"]
            for name in env._config.reward.keys()
        }
        reward_raw = {
            name: reward_scaled[name] / jp.asarray(env._config.reward[name])
            for name in env._config.reward.keys()
        }
        done_reason = {
            name.split("/", 1)[1]: value.astype(jp.bool_)
            for name, value in next_state.metrics.items()
            if name.startswith("termination/")
        }
        post_contacts = env._foot_contacts(next_state.data)
        record = {
            "t": state.info["policy_steps"].astype(jp.float32) * env.dt,
            "pre": {
                "root_pose": state.data.qpos[:7],
                "root_velocity": state.data.qvel[:6],
                "joint_q": state.data.qpos[canonical_joint_qpos],
                "joint_qd": state.data.qvel[canonical_joint_qvel],
                "contact": contacts,
            },
            "command": state.info["command"],
            "action_requested": clipped,
            "action_applied": applied,
            "firmware_tick": {
                "state": state_history,
                "output": output_history,
            },
            "q_des": output_history.model_joint_targets[-1].reshape(18),
            "post": {
                "root_pose": next_state.data.qpos[:7],
                "root_velocity": next_state.data.qvel[:6],
                "joint_q": next_state.data.qpos[canonical_joint_qpos],
                "joint_qd": next_state.data.qvel[canonical_joint_qvel],
                "contact": post_contacts,
            },
            "observation_pre": state.obs,
            "observation_post": next_state.obs,
            "reward_raw": reward_raw,
            "reward_scaled": reward_scaled,
            "reward_ascent": next_state.metrics["reward/ascent"],
            "reward_success": next_state.metrics["reward/success"],
            "reward_failure": next_state.metrics["reward/failure"],
            "reward_total": next_state.reward,
            "done": next_state.done.astype(jp.bool_),
            "done_reason": done_reason,
            "debug": {
                "support_height": next_state.info["support_height"],
                "terrain_15": env._terrain_features(
                    next_state.data, next_state.info["support_height"]
                ),
                "pitch_ff": next_state.info["pitch_ff"],
                "swing_boost": next_state.info["swing_boost"],
                "foot_world": next_state.data.site_xpos[env._foot_site_ids],
                "foot_controller_body": env._feet_controller_body(next_state.data),
                "torso_contact": env._body_contact(next_state.data),
                "self_collision": env._self_collision(next_state.data),
                "actuator_force": next_state.data.actuator_force,
                "joint_limit_margin": firmware.JOINT_LIMIT
                - jp.max(jp.abs(output_history.servo_joint_targets[-1])),
                "posture_target": _effective_posture_target(
                    state.info["command"], pitch_ff
                ),
            },
        }
        return next_state, record

    return capture


def export(output_npz: Path, output_json: Path, steps: int) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = HexapodRoughTerrainEnv(terrain_level=0)
    initial = env.reset(jax.random.PRNGKey(0))
    actions = jp.asarray(_scripted_actions(steps))
    capture = _build_capture(env)
    _, trace = jax.jit(lambda state, xs: jax.lax.scan(capture, state, xs))(
        initial, actions
    )
    jax.block_until_ready(trace["reward_total"])

    arrays: dict[str, np.ndarray] = {}
    _flatten("", trace, arrays)
    arrays = {key.lstrip("/"): value for key, value in arrays.items()}
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **arrays)

    xml_path = repo_root / "mjx/generated/hexapod_rl.xml"
    urdf_path = repo_root / "mjx/generated/hexapod.urdf"
    isaaclab_root = Path("/home/huro/IsaacLab")
    status = _git(["status", "--short"], repo_root)
    metadata = {
        "schema": SCHEMA,
        "seed": 0,
        "terrain_level": 0,
        "domain_randomization": False,
        "collision_mode": "lower_leg",
        "duration_seconds": steps * 0.02,
        "policy_steps": steps,
        "firmware_ticks_per_policy_step": FIRMWARE_TICKS,
        "action_source": "deterministic_scripted_v1_zero_first_1s",
        "action_contract": ACTION_CONTRACT_VERSION,
        "observation_contract": OBSERVATION_CONTRACT_VERSION,
        "canonical_leg_order": list(CANONICAL_LEGS),
        "source": {
            "git_commit": _git(["rev-parse", "HEAD"], repo_root),
            "git_dirty": bool(status),
            "git_status": status.splitlines(),
            "python": platform.python_version(),
            "jax": jax.__version__,
            "mujoco": mujoco.__version__,
            "numpy": np.__version__,
            "devices": [str(device) for device in jax.devices()],
            "urdf_sha256": _sha256(urdf_path),
            "rl_xml_sha256": _sha256(xml_path),
            "isaaclab_version": (isaaclab_root / "VERSION").read_text().strip(),
            "isaaclab_commit": _git(["rev-parse", "HEAD"], isaaclab_root),
            "isaaclab_status": _git(["status", "--short"], isaaclab_root).splitlines(),
        },
        "npz": {
            "path": output_npz.name,
            "sha256": _sha256(output_npz),
            "arrays": {
                key: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for key, value in sorted(arrays.items())
            },
        },
    }
    output_json.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_npz.resolve())
    print(output_json.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    default_dir = Path(__file__).resolve().parent / "golden"
    parser.add_argument("--steps", type=int, default=POLICY_STEPS)
    parser.add_argument(
        "--output-npz", type=Path, default=default_dir / "isaac_contract_v1_flat_seed0.npz"
    )
    parser.add_argument(
        "--output-json", type=Path, default=default_dir / "isaac_contract_v1_flat_seed0.json"
    )
    args = parser.parse_args()
    export(args.output_npz, args.output_json, args.steps)


if __name__ == "__main__":
    main()
