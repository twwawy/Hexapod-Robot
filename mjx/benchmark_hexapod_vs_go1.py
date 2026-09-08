#!/usr/bin/env python3
"""Reproducible Hexapod-vs-Go1 MuJoCo locomotion benchmark.

The Hexapod uses the highest-reward checkpoint available for the curriculum
level corresponding to each benchmark terrain.  Go1 uses MuJoCo Playground's
bundled, pretrained joystick ONNX policy.  Rewards are reported only as
Hexapod training provenance because the two reward contracts are not equal.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any
import xml.etree.ElementTree as ET

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jp
import matplotlib.pyplot as plt
import mujoco
import numpy as np
import onnxruntime as ort


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
RUN_ROOT = HERE / "runs" / "terrain"
COMMAND_SPEED_MPS = 0.12
GO1_NATIVE_WALK_SPEED_MPS = 0.30
MAX_DURATION_S = 25.0
CUSTOM_STAIR_LEVEL = 17

TERRAINS = {
    "flat": {
        "label": "Flat",
        "level": 0,
        "checkpoint_level": 0,
        "goal_x": 1.00,
    },
    "rough_hard": {
        "label": "Rough 5 cm",
        "level": 2,
        "checkpoint_level": 2,
        "goal_x": 2.31,
    },
    "ramp_steep": {
        "label": "Ramp 15 deg",
        "level": 4,
        "checkpoint_level": 4,
        "goal_x": 2.00,
    },
    "step_7p5cm": {
        "label": "Step 7.5 cm",
        "level": CUSTOM_STAIR_LEVEL,
        # The closest trained stair level is 7 x 8 cm (level 7).
        "checkpoint_level": 7,
        "goal_x": 1.05,
    },
}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def collect_stage_history() -> list[dict[str, Any]]:
    """Return the maximum evaluated reward in every stage/run."""
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(RUN_ROOT.iterdir()):
        metadata_path = run_dir / "run_metadata.json"
        history_path = run_dir / "monitor" / "metrics_history.jsonl"
        if not metadata_path.exists() or not history_path.exists():
            continue
        try:
            metadata = _read_json(metadata_path)
            evaluations = [
                json.loads(line)
                for line in history_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except (OSError, ValueError, TypeError):
            continue
        if not evaluations or not isinstance(metadata.get("terrain_level"), int):
            continue
        best = max(evaluations, key=lambda item: float(item.get("score", -np.inf)))
        step = int(best.get("step", 0))
        checkpoint = run_dir / "checkpoints" / f"{step:012d}"
        if not (checkpoint / "ppo_network_config.json").exists():
            checkpoint_text = ""
        else:
            checkpoint_text = str(checkpoint.resolve())
        metrics = best.get("metrics", {})
        rows.append(
            {
                "run": run_dir.name,
                "stage": metadata.get("curriculum_stage"),
                "level": int(metadata["terrain_level"]),
                "terrain": metadata.get("terrain_name", "unknown"),
                "best_reward": float(best["score"]),
                "reward_step": step,
                "terrain_success_rate": float(
                    metrics.get("eval/episode_terrain_success", np.nan)
                ),
                "checkpoint": checkpoint_text,
                "observation_contract": metadata.get(
                    "observation_contract_version", ""
                ),
                "reward_contract": metadata.get("reward_contract_version", ""),
                "action_contract": metadata.get("action_contract_version", ""),
                "collision_mode": metadata.get("collision_mode", "lower_leg"),
            }
        )
    return rows


def select_level_best(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Select highest reward with a materialized checkpoint for each level."""
    compatible = [
        row
        for row in rows
        if row["checkpoint"]
        and row["observation_contract"]
        == "firmware_state_collision_terrain_command5_pitch_v3"
        and row["action_contract"]
        == "stm32_firmware_adaptive_swing_residual_v3"
    ]
    selected: dict[int, dict[str, Any]] = {}
    for row in compatible:
        old = selected.get(row["level"])
        if old is None or row["best_reward"] > old["best_reward"]:
            selected[row["level"]] = row
    missing = sorted(
        {int(spec["checkpoint_level"]) for spec in TERRAINS.values()} - selected.keys()
    )
    if missing:
        raise RuntimeError(f"no compatible checkpoint for terrain levels: {missing}")
    return selected


def select_stage_level_best(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated experiments to one maximum-reward row per stage/level."""
    selected: dict[tuple[int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row["stage"], int):
            continue
        key = (int(row["stage"]), int(row["level"]))
        old = selected.get(key)
        if old is None or row["best_reward"] > old["best_reward"]:
            selected[key] = row
    return [selected[key] for key in sorted(selected)]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _quat_conjugate(quat: np.ndarray) -> np.ndarray:
    return quat * np.array([1.0, -1.0, -1.0, -1.0])


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]
    )


def _quat_to_euler(quat: np.ndarray) -> np.ndarray:
    quat = quat / max(np.linalg.norm(quat), 1.0e-12)
    w, x, y, z = quat
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def _relative_euler(quat: np.ndarray, initial_quat: np.ndarray) -> np.ndarray:
    return _quat_to_euler(_quat_multiply(quat, _quat_conjugate(initial_quat)))


def _summarize_trace(
    *,
    robot: str,
    terrain: str,
    times: list[float],
    positions: list[np.ndarray],
    attitudes: list[np.ndarray],
    goal_x: float,
    command_speed: float,
    reward: float | None = None,
    checkpoint: str = "",
) -> dict[str, Any]:
    time = np.asarray(times)
    pos = np.asarray(positions)
    rpy = np.unwrap(np.asarray(attitudes), axis=0)
    duration = float(time[-1]) if len(time) else 0.0
    distance = float(max(pos[-1, 0] - pos[0, 0], 0.0))
    velocities = np.gradient(pos[:, 0], time) if len(time) > 1 else np.zeros(1)
    warm = time >= min(1.0, 0.25 * duration)
    if not np.any(warm):
        warm = np.ones(len(time), dtype=bool)
    rpy_deg = np.rad2deg(rpy[warm])
    rms = np.sqrt(np.mean(np.square(rpy_deg), axis=0))
    peak = np.max(np.abs(rpy_deg), axis=0)
    mean_speed = float(np.mean(velocities[warm]))
    speed_rmse = float(
        np.sqrt(np.mean(np.square(velocities[warm] - command_speed)))
    )
    completion = min(distance / goal_x, 1.0) if goal_x > 0 else 0.0
    fell = bool(
        pos[-1, 2] < 0.12
        or np.max(np.abs(rpy[-1, :2])) > math.radians(60.0)
    )
    return {
        "robot": robot,
        "terrain": terrain,
        "terrain_label": TERRAINS[terrain]["label"],
        "command_speed_mps": command_speed,
        "duration_s": duration,
        "distance_m": distance,
        "completion_pct": 100.0 * completion,
        "mean_forward_speed_mps": mean_speed,
        "speed_tracking_rmse_mps": speed_rmse,
        "roll_rmse_deg": float(rms[0]),
        "pitch_rmse_deg": float(rms[1]),
        "yaw_rmse_deg": float(rms[2]),
        "roll_peak_deg": float(peak[0]),
        "pitch_peak_deg": float(peak[1]),
        "yaw_peak_deg": float(peak[2]),
        "fell": fell,
        "hexapod_training_reward": reward if reward is not None else "",
        "checkpoint": checkpoint,
    }


def _network_policy(checkpoint_path: Path):
    """Load current Brax PPO checkpoints while tolerating JSON null initializers."""
    from brax.training import networks as brax_networks
    from brax.training import types as brax_types
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks

    config = _read_json(checkpoint_path / "ppo_network_config.json")
    kwargs = dict(config.get("network_factory_kwargs", {}))
    activation = kwargs.get("activation")
    if isinstance(activation, str):
        kwargs["activation"] = brax_networks.ACTIVATION[activation]
    initializer_keys = (
        "policy_network_kernel_init_fn",
        "value_network_kernel_init_fn",
        "mean_kernel_init_fn",
    )
    for key in initializer_keys:
        name = kwargs.get(key)
        if name is None:
            kwargs.pop(key, None)
            continue
        if isinstance(name, str):
            kwargs[key] = brax_networks.KERNEL_INITIALIZER[name]
    preprocess = brax_types.identity_observation_preprocessor
    if bool(config.get("normalize_observations", False)):
        preprocess = running_statistics.normalize
    obs_shape = config["observation_size"]["shape"]
    network = ppo_networks.make_ppo_networks(
        observation_size=int(obs_shape[0]),
        action_size=int(config["action_size"]),
        preprocess_observations_fn=preprocess,
        **kwargs,
    )
    params = ppo_checkpoint.load(checkpoint_path)
    return ppo_networks.make_inference_fn(network)(params, deterministic=True)


def _install_custom_stair() -> None:
    import terrain_curriculum

    if len(terrain_curriculum.TERRAIN_LEVELS) > CUSTOM_STAIR_LEVEL:
        return
    custom = terrain_curriculum.TerrainLevel(
        CUSTOM_STAIR_LEVEL,
        "benchmark_step_1x7p5cm",
        "stairs",
        stair_count=1,
        stair_riser=0.075,
    )
    terrain_curriculum.TERRAIN_LEVELS = (*terrain_curriculum.TERRAIN_LEVELS, custom)


def run_hexapod(
    terrain: str, selected: dict[int, dict[str, Any]], seed: int
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sys.path.insert(0, str(HERE))
    _install_custom_stair()
    from rough_terrain_env import HexapodRoughTerrainEnv, default_config

    spec = TERRAINS[terrain]
    chosen = selected[int(spec["checkpoint_level"])]
    config = default_config()
    config.episode_length = int(MAX_DURATION_S / config.ctrl_dt)
    config.command_min_speed = COMMAND_SPEED_MPS
    config.command_max_speed = COMMAND_SPEED_MPS
    config.command_max_yaw_rate = 0.0
    config.dr_enabled = False
    config.collision_mode = chosen["collision_mode"] or "lower_leg"
    env = HexapodRoughTerrainEnv(config=config, terrain_level=int(spec["level"]))
    policy = jax.jit(_network_policy(Path(chosen["checkpoint"])))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    rng = jax.random.PRNGKey(seed)
    state = reset(rng)
    state.obs.block_until_ready()
    state.info["command"] = jp.asarray([COMMAND_SPEED_MPS, 0.0, 0.0, 0.0, 0.0])
    initial_quat = np.asarray(state.data.qpos[3:7], dtype=float)
    times = [0.0]
    positions = [np.asarray(state.data.qpos[:3], dtype=float)]
    attitudes = [np.zeros(3)]
    trace = []
    max_steps = int(MAX_DURATION_S / env.dt)
    for index in range(max_steps):
        rng, action_key = jax.random.split(rng)
        action, _ = policy(state.obs, action_key)
        state = step(state, action)
        state.obs.block_until_ready()
        position = np.asarray(state.data.qpos[:3], dtype=float)
        attitude = _relative_euler(
            np.asarray(state.data.qpos[3:7], dtype=float), initial_quat
        )
        current_time = (index + 1) * env.dt
        times.append(current_time)
        positions.append(position)
        attitudes.append(attitude)
        trace.append(
            {
                "robot": "Hexapod",
                "terrain": terrain,
                "time_s": current_time,
                "x_m": position[0],
                "y_m": position[1],
                "z_m": position[2],
                "roll_deg": math.degrees(attitude[0]),
                "pitch_deg": math.degrees(attitude[1]),
                "yaw_deg": math.degrees(attitude[2]),
                "reward": float(np.asarray(state.reward)),
            }
        )
        if bool(np.asarray(state.done)):
            break
    summary = _summarize_trace(
        robot="Hexapod",
        terrain=terrain,
        times=times,
        positions=positions,
        attitudes=attitudes,
        goal_x=float(spec["goal_x"]),
        command_speed=COMMAND_SPEED_MPS,
        reward=float(chosen["best_reward"]),
        checkpoint=chosen["checkpoint"],
    )
    return summary, trace


def _go1_scene_xml(terrain: str) -> tuple[str, dict[str, bytes]]:
    from mujoco_playground._src.locomotion.go1 import go1_constants
    from mujoco_playground._src.locomotion.go1.base import get_assets
    from terrain_curriculum import (
        PLATEAU_DEPTH,
        RAMP_LENGTH,
        ROUGH_HFIELD_NCOL,
        ROUGH_HFIELD_NROW,
        ROUGH_LENGTH,
        TERRAIN_HALF_WIDTH,
        TERRAIN_START_X,
        rough_heightfield_grid,
    )

    root = ET.fromstring(
        Path(go1_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix()).read_text(
            encoding="utf-8"
        )
    )
    assets = root.find("asset")
    worldbody = root.find("worldbody")
    assert assets is not None and worldbody is not None
    if terrain == "rough_hard":
        grid = np.asarray(rough_heightfield_grid(0.05)) / 0.05
        ET.SubElement(
            assets,
            "hfield",
            name="benchmark_rough",
            nrow=str(ROUGH_HFIELD_NROW),
            ncol=str(ROUGH_HFIELD_NCOL),
            size=f"{ROUGH_LENGTH / 2} {TERRAIN_HALF_WIDTH} 0.05 0.001",
            elevation=" ".join(f"{value:.9g}" for value in grid.reshape(-1)),
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="benchmark_rough_geom",
            type="hfield",
            hfield="benchmark_rough",
            pos=f"{TERRAIN_START_X + ROUGH_LENGTH / 2} 0 0",
            friction="1.1 0.01 0.001",
            contype="1",
            conaffinity="0",
        )
    elif terrain == "ramp_steep":
        angle = math.radians(15.0)
        rise = RAMP_LENGTH * math.tan(angle)
        half_thickness = 0.025
        ET.SubElement(
            worldbody,
            "geom",
            name="benchmark_ramp",
            type="box",
            pos=(
                f"{TERRAIN_START_X + RAMP_LENGTH / 2} 0 "
                f"{rise / 2 - half_thickness * math.cos(angle)}"
            ),
            size=(
                f"{RAMP_LENGTH / (2 * math.cos(angle))} "
                f"{TERRAIN_HALF_WIDTH} {half_thickness}"
            ),
            quat=f"{math.cos(angle / 2)} 0 {-math.sin(angle / 2)} 0",
            friction="1.1 0.01 0.001",
            contype="1",
            conaffinity="0",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="benchmark_plateau",
            type="box",
            pos=(
                f"{TERRAIN_START_X + RAMP_LENGTH + PLATEAU_DEPTH / 2} "
                f"0 {rise / 2}"
            ),
            size=f"{PLATEAU_DEPTH / 2} {TERRAIN_HALF_WIDTH} {rise / 2}",
            friction="1.1 0.01 0.001",
            contype="1",
            conaffinity="0",
        )
    elif terrain == "step_7p5cm":
        height = 0.075
        ET.SubElement(
            worldbody,
            "geom",
            name="benchmark_step",
            type="box",
            pos=f"{TERRAIN_START_X + 0.125} 0 {height / 2}",
            size=f"0.125 {TERRAIN_HALF_WIDTH} {height / 2}",
            friction="1.1 0.01 0.001",
            contype="1",
            conaffinity="0",
        )
        ET.SubElement(
            worldbody,
            "geom",
            name="benchmark_step_plateau",
            type="box",
            pos=f"{TERRAIN_START_X + 0.25 + PLATEAU_DEPTH / 2} 0 {height / 2}",
            size=f"{PLATEAU_DEPTH / 2} {TERRAIN_HALF_WIDTH} {height / 2}",
            friction="1.1 0.01 0.001",
            contype="1",
            conaffinity="0",
        )
    model_assets = get_assets()
    # PyPI Playground ships the benchmark XML and pretrained policy but may not
    # materialize the optional Menagerie STL cache.  The collision model and
    # explicit inertials are complete, so strip visual-only meshes when those
    # files are absent.  This does not alter contacts, mass, joints, or control.
    included_name = "go1_mjx_feetonly.xml"
    included = ET.fromstring(model_assets[included_name])
    asset_node = included.find("asset")
    if asset_node is not None:
        for mesh in list(asset_node.findall("mesh")):
            asset_node.remove(mesh)
    for body in included.iter("body"):
        for geom in list(body.findall("geom")):
            if geom.get("class") == "visual" or geom.get("mesh"):
                body.remove(geom)
    model_assets[included_name] = ET.tostring(included, encoding="utf-8")
    return ET.tostring(root, encoding="unicode"), model_assets


def run_go1(
    terrain: str, command_speed: float = COMMAND_SPEED_MPS
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import mujoco_playground

    xml, assets = _go1_scene_xml(terrain)
    model = mujoco.MjModel.from_xml_string(xml, assets=assets)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    model.opt.timestep = 0.004
    model.dof_damping[6:] = 0.5
    model.actuator_gainprm[:, 0] = 35.0
    model.actuator_biasprm[:, 1] = -35.0
    default_angles = model.keyframe("home").qpos[7:].copy()
    installed_policy = (
        Path(mujoco_playground.__file__).resolve().parent
        / "experimental"
        / "sim2sim"
        / "onnx"
        / "go1_policy.onnx"
    )
    source_policy = (
        REPO.parent
        / "mujoco"
        / "mujoco_playground"
        / "mujoco_playground"
        / "mujoco_playground"
        / "experimental"
        / "sim2sim"
        / "onnx"
        / "go1_policy.onnx"
    )
    policy_path = installed_policy if installed_policy.exists() else source_policy
    if not policy_path.exists():
        raise FileNotFoundError("MuJoCo Playground Go1 ONNX policy was not found")
    session = ort.InferenceSession(
        str(policy_path), providers=["CPUExecutionProvider"]
    )
    last_action = np.zeros(12, dtype=np.float32)
    initial_quat = data.qpos[3:7].copy()
    times = [0.0]
    positions = [data.qpos[:3].copy()]
    attitudes = [np.zeros(3)]
    trace = []
    control_steps = int(MAX_DURATION_S / 0.02)
    for index in range(control_steps):
        linvel = data.sensor("local_linvel").data.copy()
        gyro = data.sensor("gyro").data.copy()
        imu_xmat = data.site_xmat[model.site("imu").id].reshape(3, 3)
        gravity = imu_xmat.T @ np.array([0.0, 0.0, -1.0])
        obs = np.hstack(
            [
                linvel,
                gyro,
                gravity,
                data.qpos[7:] - default_angles,
                data.qvel[6:],
                last_action,
                np.array([command_speed, 0.0, 0.0]),
            ]
        ).astype(np.float32)
        last_action = session.run(
            ["continuous_actions"], {"obs": obs.reshape(1, -1)}
        )[0][0]
        data.ctrl[:] = last_action * 0.5 + default_angles
        for _ in range(5):
            mujoco.mj_step(model, data)
        position = data.qpos[:3].copy()
        attitude = _relative_euler(data.qpos[3:7].copy(), initial_quat)
        current_time = (index + 1) * 0.02
        times.append(current_time)
        positions.append(position)
        attitudes.append(attitude)
        trace.append(
            {
                "robot": "Go1",
                "terrain": terrain,
                "time_s": current_time,
                "x_m": position[0],
                "y_m": position[1],
                "z_m": position[2],
                "roll_deg": math.degrees(attitude[0]),
                "pitch_deg": math.degrees(attitude[1]),
                "yaw_deg": math.degrees(attitude[2]),
                "reward": "",
            }
        )
        if position[2] < 0.12 or np.max(np.abs(attitude[:2])) > math.radians(60.0):
            break
        if position[0] >= float(TERRAINS[terrain]["goal_x"]):
            break
    summary = _summarize_trace(
        robot="Go1",
        terrain=terrain,
        times=times,
        positions=positions,
        attitudes=attitudes,
        goal_x=float(TERRAINS[terrain]["goal_x"]),
        command_speed=command_speed,
    )
    return summary, trace


def plot_stage_rewards(rows: list[dict[str, Any]], output: Path) -> None:
    usable = [row for row in rows if isinstance(row["stage"], int)]
    fig, ax = plt.subplots(figsize=(12, 6.5))
    levels = sorted({row["level"] for row in usable})
    cmap = plt.get_cmap("viridis", max(len(levels), 1))
    for color_index, level in enumerate(levels):
        level_rows = sorted(
            (row for row in usable if row["level"] == level),
            key=lambda row: row["stage"],
        )
        try:
            from terrain_curriculum import terrain_level

            terrain_name = terrain_level(level).name
        except ValueError:
            terrain_name = level_rows[0]["terrain"]
        ax.scatter(
            [row["stage"] for row in level_rows],
            [row["best_reward"] for row in level_rows],
            s=38,
            color=cmap(color_index),
            label=f"Level {level}: {terrain_name}",
            alpha=0.85,
        )
    ax.set(title="Hexapod maximum evaluated reward by curriculum stage", xlabel="Stage", ylabel="Maximum episode reward")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = [str(TERRAINS[key]["label"]) for key in TERRAINS]
    x = np.arange(len(labels))
    width = 0.36
    robots = ("Hexapod", "Go1")
    colors = {"Hexapod": "#2474b5", "Go1": "#e08032"}

    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for axis, metric, title in zip(
        axes,
        ("roll_rmse_deg", "pitch_rmse_deg", "yaw_rmse_deg"),
        ("Roll RMS error", "Pitch RMS error", "Yaw RMS error"),
    ):
        for robot_index, robot in enumerate(robots):
            vals = [
                next(row[metric] for row in rows if row["robot"] == robot and row["terrain"] == terrain)
                for terrain in TERRAINS
            ]
            axis.bar(x + (robot_index - 0.5) * width, vals, width, label=robot, color=colors[robot])
        axis.set_ylabel("degrees")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    axes[-1].set_xticks(x, labels)
    fig.suptitle("Body attitude retention (lower is better)", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_orientation.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for robot_index, robot in enumerate(robots):
        completion = [
            next(row["completion_pct"] for row in rows if row["robot"] == robot and row["terrain"] == terrain)
            for terrain in TERRAINS
        ]
        speed_error = [
            next(row["speed_tracking_rmse_mps"] for row in rows if row["robot"] == robot and row["terrain"] == terrain)
            for terrain in TERRAINS
        ]
        axes[0].bar(x + (robot_index - 0.5) * width, completion, width, label=robot, color=colors[robot])
        axes[1].bar(x + (robot_index - 0.5) * width, speed_error, width, label=robot, color=colors[robot])
    axes[0].set(title="Course completion (higher is better)", ylabel="percent")
    axes[0].set_ylim(0, 110)
    axes[0].legend()
    axes[1].set(title="Forward-speed tracking RMS error (lower is better)", ylabel="m/s")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels)
    fig.tight_layout()
    fig.savefig(output_dir / "comparison_walking.png", dpi=180)
    plt.close(fig)


def plot_native_walking(rows: list[dict[str, Any]], output: Path) -> None:
    labels = [str(TERRAINS[key]["label"]) for key in TERRAINS]
    x = np.arange(len(labels))
    width = 0.36
    robots = ("Hexapod", "Go1")
    colors = {"Hexapod": "#2474b5", "Go1": "#e08032"}
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    for robot_index, robot in enumerate(robots):
        completion = [
            next(row["completion_pct"] for row in rows if row["robot"] == robot and row["terrain"] == terrain)
            for terrain in TERRAINS
        ]
        attitude = [
            math.sqrt(
                sum(
                    next(row[key] for row in rows if row["robot"] == robot and row["terrain"] == terrain) ** 2
                    for key in ("roll_rmse_deg", "pitch_rmse_deg", "yaw_rmse_deg")
                )
            )
            for terrain in TERRAINS
        ]
        axes[0].bar(x + (robot_index - 0.5) * width, completion, width, label=robot, color=colors[robot])
        axes[1].bar(x + (robot_index - 0.5) * width, attitude, width, label=robot, color=colors[robot])
    axes[0].set(title="Course completion at each controller's walking speed", ylabel="percent", ylim=(0, 110))
    axes[0].legend(title="Hexapod 0.12 m/s; Go1 0.30 m/s")
    axes[1].set(title="Combined attitude RMS during walking (lower is better)", ylabel="degrees")
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    axes[-1].set_xticks(x, labels)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

def write_report(
    output: Path,
    summaries: list[dict[str, Any]],
    selected: dict[int, dict[str, Any]],
    native_walking: list[dict[str, Any]],
) -> None:
    lines = [
        "# Hexapod vs Unitree Go1 MuJoCo 비교평가",
        "",
        f"- 공통 명령: 전진 {COMMAND_SPEED_MPS:.2f} m/s, yaw 0 rad/s",
        f"- 최대 rollout: {MAX_DURATION_S:.0f}초",
        "- 지형: 평지, 결정론적 5 cm 요철, 15도 경사, 단일 7.5 cm 턱",
        "- Hexapod: level 0, 2, 4와 가장 가까운 계단 level 7의 최고 reward 호환 체크포인트",
        "- Go1: MuJoCo Playground에 포함된 사전학습 joystick ONNX 정책",
        "- 두 로봇의 reward 정의가 다르므로 학습 reward를 로봇 간 직접 비교하지 않음",
        "",
        "## 공통 0.12 m/s rollout 지표",
        "",
        "| 로봇 | 지형 | Roll RMSE (deg) | Pitch RMSE (deg) | Yaw RMSE (deg) | 통과율 (%) | 속도 RMSE (m/s) | 낙상 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['robot']} | {row['terrain_label']} | {row['roll_rmse_deg']:.2f} | "
            f"{row['pitch_rmse_deg']:.2f} | {row['yaw_rmse_deg']:.2f} | "
            f"{row['completion_pct']:.1f} | {row['speed_tracking_rmse_mps']:.3f} | "
            f"{'yes' if row['fell'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## 로봇별 운용 보행속도 보조평가",
            "",
            f"포함된 Go1 정책은 {COMMAND_SPEED_MPS:.2f} m/s에서 제자리 유지에 가깝게 동작했다. 실제 지형 통과를 보기 위해 아래 표에서는 Hexapod을 학습 최대 속도({COMMAND_SPEED_MPS:.2f} m/s), Go1을 보행이 시작되는 {GO1_NATIVE_WALK_SPEED_MPS:.2f} m/s로 평가했다. 따라서 이 보조 결과는 동일 속도 비교가 아니다.",
            "",
            "| 로봇 | 지형 | 명령 (m/s) | 통과율 (%) | 자세 RMS norm (deg) | 낙상 |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for row in native_walking:
        attitude_norm = math.sqrt(
            row["roll_rmse_deg"] ** 2
            + row["pitch_rmse_deg"] ** 2
            + row["yaw_rmse_deg"] ** 2
        )
        lines.append(
            f"| {row['robot']} | {row['terrain_label']} | {row['command_speed_mps']:.2f} | "
            f"{row['completion_pct']:.1f} | {attitude_norm:.2f} | "
            f"{'yes' if row['fell'] else 'no'} |"
        )
    lines.extend(["", "## 선택된 Hexapod 체크포인트", ""])
    for level in sorted({int(spec["checkpoint_level"]) for spec in TERRAINS.values()}):
        row = selected[level]
        lines.append(
            f"- Level {level} ({row['terrain']}), stage {row['stage']}: "
            f"reward={row['best_reward']:.3f}, step={row['reward_step']}, `{row['checkpoint']}`"
        )
    lines.extend(
        [
            "",
            "## 해석 제한",
            "",
            "Hexapod 체크포인트는 자체 학습 reward로 선정했지만 공통 비교에는 측정된 운동학 지표만 사용했다. 7.5 cm 시험은 학습에 없던 단일 턱이며, Hexapod에는 가장 가까운 8 cm curriculum 정책을 사용했다. 로봇·지형별 결정론적 rollout 1회 결과이므로 신뢰구간을 포함한 통계적 결론이 아니라 재현 가능한 엔지니어링 비교로 해석해야 한다.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path, default=REPO / "outputs" / "mujoco_benchmark"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    history = collect_stage_history()
    stage_level_best = select_stage_level_best(history)
    selected = select_level_best(history)
    _write_csv(output_dir / "hexapod_all_run_best_history.csv", history)
    _write_csv(output_dir / "hexapod_stage_level_best.csv", stage_level_best)
    level_rows = [selected[level] for level in sorted(selected)]
    _write_csv(output_dir / "hexapod_terrain_level_best.csv", level_rows)
    plot_stage_rewards(
        stage_level_best, output_dir / "hexapod_stage_reward_history.png"
    )

    summaries: list[dict[str, Any]] = []
    hexapod_summaries: list[dict[str, Any]] = []
    traces: list[dict[str, Any]] = []
    for terrain in TERRAINS:
        print(f"[benchmark] Hexapod terrain={terrain}", flush=True)
        summary, trace = run_hexapod(terrain, selected, args.seed)
        summaries.append(summary)
        hexapod_summaries.append(summary)
        traces.extend(trace)

    native_walking = list(hexapod_summaries)
    for terrain in TERRAINS:
        print(f"[benchmark] Go1 native-walk terrain={terrain}", flush=True)
        summary, trace = run_go1(terrain, GO1_NATIVE_WALK_SPEED_MPS)
        native_walking.append(summary)
        for row in trace:
            row["robot"] = "Go1_native_0p30mps"
        traces.extend(trace)
        print(f"[benchmark] Go1 terrain={terrain}", flush=True)
        summary, trace = run_go1(terrain)
        summaries.append(summary)
        traces.extend(trace)

    _write_csv(output_dir / "comparison_summary.csv", summaries)
    _write_csv(output_dir / "rollout_timeseries.csv", traces)
    _write_csv(output_dir / "native_walking_summary.csv", native_walking)
    (output_dir / "comparison_summary.json").write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    plot_comparison(summaries, output_dir)
    plot_native_walking(native_walking, output_dir / "comparison_native_walking.png")
    write_report(output_dir / "REPORT.md", summaries, selected, native_walking)
    print(f"BENCHMARK_OUTPUT={output_dir}")


if __name__ == "__main__":
    main()
