#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LOCAL_EXT = _REPO_ROOT / "source" / "spider_rl"
if _LOCAL_EXT.exists():
    sys.path.insert(0, str(_LOCAL_EXT))

ENV_ID = "Hexapedal-MuJoCo-Direct-v0"
COMMAND_SCHEMA = ["vx", "vy", "wz"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SB3 PPO checkpoints on the standalone MuJoCo hexapedal direct environment.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path, e.g. latest.zip or best.zip.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "source" / "spider_rl" / "spider_mujoco" / "hexapedal_direct" / "agents" / "sb3_ppo_cfg.yaml",
        help="Path to the SB3 PPO YAML config.",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Deterministic episode count.")
    parser.add_argument("--seed", type=int, default=None, help="Optional eval seed override.")
    parser.add_argument("--device", type=str, default="auto", help="SB3 device, e.g. auto/cpu/cuda.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON scorecard output path.")
    parser.add_argument("--task", type=str, default=ENV_ID, help="Gymnasium environment id to evaluate.")
    return parser.parse_args()



def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in config: {path}")
    return data



def bootstrap_spider_mujoco() -> None:
    import spider_mujoco  # noqa: F401



def make_env(task_id: str, seed: int | None) -> gym.Env[Any, Any]:
    import gymnasium as gym
    bootstrap_spider_mujoco()
    env = gym.make(task_id)
    if seed is not None:
        env.reset(seed=seed)
    return env



def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def unwrap_attr(env: gym.Env[Any, Any], name: str) -> Any:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "env", None)
    return None



def infer_forward_x(env: gym.Env[Any, Any], info: dict[str, Any]) -> float:
    for key in ("x_position", "forward_distance", "base_x"):
        if key in info:
            return as_float(info[key], 0.0)
    data = unwrap_attr(env, "data")
    qpos = getattr(data, "qpos", None)
    if qpos is not None and len(qpos) > 0:
        return float(qpos[0])
    return 0.0



def infer_termination_reason(info: dict[str, Any], terminated: bool, truncated: bool) -> str:
    for key in ("termination_reason", "terminated_reason", "done_reason"):
        value = info.get(key)
        if value:
            return str(value)
    if truncated:
        return "timeout"
    if terminated:
        return "terminated"
    return "running"



def episode_metrics(episode_info: dict[str, Any], termination_reason: str) -> dict[str, Any]:
    mean_vx_error = as_float(episode_info.get("tracking_lin_vel_error"), 0.0)
    mean_wz_error = as_float(episode_info.get("tracking_ang_vel_error"), 0.0)
    tracking_error_scalar = 0.5 * abs(mean_vx_error) / 0.10 + 0.5 * abs(mean_wz_error) / 0.25
    fall_count = int(as_float(episode_info.get("fall_count"), 0.0))
    return {
        "return": as_float(episode_info.get("r"), 0.0),
        "length": int(as_float(episode_info.get("l"), 0.0)),
        "mean_vx_tracking_error": mean_vx_error,
        "mean_wz_tracking_error": mean_wz_error,
        "tracking_error_scalar": tracking_error_scalar,
        "undesired_contact_count": int(as_float(episode_info.get("undesired_contact_count"), 0.0)),
        "desired_contact_count": int(as_float(episode_info.get("desired_contact_count"), 0.0)),
        "fall_count": fall_count,
        "termination_reason": termination_reason,
        "command_vx": as_float(episode_info.get("command_vx"), 0.0),
        "command_vy": as_float(episode_info.get("command_vy"), 0.0),
        "command_wz": as_float(episode_info.get("command_wz"), 0.0),
        "fall_like_termination": fall_count > 0 or termination_reason in {"fall", "height_violation"},
    }



def evaluate_model(model: Any, env: gym.Env[Any, Any], episodes: int) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for episode_idx in range(episodes):
        obs, info = env.reset()
        start_x = infer_forward_x(env, info)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episode_info = info.get("episode")
                if not isinstance(episode_info, dict):
                    raise KeyError("Expected info['episode'] metrics from the MuJoCo env at episode end.")
                metrics = episode_metrics(episode_info, infer_termination_reason(info, terminated, truncated))
                metrics["episode_index"] = episode_idx
                metrics["forward_distance"] = infer_forward_x(env, info) - start_x
                if not np.isclose(metrics["command_vy"], 0.0):
                    raise AssertionError(f"Approved v1 contract requires vy == 0.0, got {metrics['command_vy']}")
                results.append(metrics)
                break
    aggregates = {
        "return": float(np.mean([item["return"] for item in results])),
        "forward_distance": float(np.mean([item["forward_distance"] for item in results])),
        "mean_vx_tracking_error": float(np.mean([item["mean_vx_tracking_error"] for item in results])),
        "mean_wz_tracking_error": float(np.mean([item["mean_wz_tracking_error"] for item in results])),
        "tracking_error_scalar": float(np.mean([item["tracking_error_scalar"] for item in results])),
        "fall_rate": float(np.mean([1.0 if item["fall_like_termination"] else 0.0 for item in results])),
        "undesired_contact_count": float(np.mean([item["undesired_contact_count"] for item in results])),
    }
    termination_reasons: dict[str, int] = {}
    for item in results:
        termination_reasons[item["termination_reason"]] = termination_reasons.get(item["termination_reason"], 0) + 1
    return {
        "aggregate": aggregates,
        "per_episode": results,
        "termination_reason_counts": termination_reasons,
    }



def infer_checkpoint_role(path: Path) -> str:
    stem = path.stem.lower()
    if stem == "best":
        return "best"
    if stem == "latest":
        return "latest"
    return "checkpoint"



def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))

    from stable_baselines3 import PPO

    env = make_env(args.task, seed=seed)
    model = PPO.load(str(args.checkpoint), env=env, device=args.device)
    scorecard = evaluate_model(model, env, episodes=args.episodes)
    output = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_role": infer_checkpoint_role(args.checkpoint),
        "episodes": args.episodes,
        "seed": seed,
        "package_import": "spider_mujoco",
        "gym_id": args.task,
        "command_schema": COMMAND_SCHEMA,
        "command_vy_expected": 0.0,
        **scorecard,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
