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
    parser = argparse.ArgumentParser(description="Train SB3 PPO on the standalone MuJoCo hexapedal direct environment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "source" / "spider_rl" / "spider_mujoco" / "hexapedal_direct" / "agents" / "sb3_ppo_cfg.yaml",
        help="Path to the SB3 PPO YAML config.",
    )
    parser.add_argument("--run-dir", type=Path, default=None, help="Output directory. Defaults to <logging.root_dir>/<protocol>.")
    parser.add_argument(
        "--protocol",
        choices=("smoke", "baseline", "extension"),
        default="baseline",
        help="Training protocol surface: 10k smoke, 500k baseline, or 1M extension.",
    )
    parser.add_argument("--timesteps", type=int, default=None, help="Optional explicit total timesteps override.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--device", type=str, default="auto", help="SB3 device, e.g. auto/cpu/cuda.")
    parser.add_argument("--task", type=str, default=ENV_ID, help="Gymnasium environment id to train.")
    return parser.parse_args()


def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in config: {path}")
    return data


def bootstrap_spider_mujoco() -> None:
    import spider_mujoco  # noqa: F401



def make_env(task_id: str) -> gym.Env[Any, Any]:
    import gymnasium as gym
    bootstrap_spider_mujoco()
    return gym.make(task_id)



def get_protocol_timesteps(cfg: dict[str, Any], protocol: str, override: int | None) -> int:
    if override is not None:
        return override
    protocol_cfg = cfg["protocol"]
    key = {
        "smoke": "smoke_timesteps",
        "baseline": "baseline_timesteps",
        "extension": "extension_timesteps",
    }[protocol]
    return int(protocol_cfg[key])



def activation_fn(name: str):
    import torch.nn as nn

    table = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    try:
        return table[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported activation_fn: {name}") from exc



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



def evaluate_model(model: Any, env: gym.Env[Any, Any], episodes: int, deterministic: bool) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for episode_idx in range(episodes):
        obs, info = env.reset()
        start_x = infer_forward_x(env, info)
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
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
    mean_return = float(np.mean([item["return"] for item in results]))
    mean_tracking_error_scalar = float(np.mean([item["tracking_error_scalar"] for item in results]))
    fall_rate = float(np.mean([1.0 if item["fall_like_termination"] else 0.0 for item in results]))
    mean_vx_error = float(np.mean([item["mean_vx_tracking_error"] for item in results]))
    mean_wz_error = float(np.mean([item["mean_wz_tracking_error"] for item in results]))
    scorecard = {
        "episodes": results,
        "aggregate": {
            "return": mean_return,
            "forward_distance": float(np.mean([item["forward_distance"] for item in results])),
            "mean_vx_tracking_error": mean_vx_error,
            "mean_wz_tracking_error": mean_wz_error,
            "tracking_error_scalar": mean_tracking_error_scalar,
            "fall_rate": fall_rate,
            "undesired_contact_count": float(np.mean([item["undesired_contact_count"] for item in results])),
        },
    }
    return scorecard



def is_better(candidate: dict[str, Any], incumbent: dict[str, Any] | None) -> tuple[bool, str]:
    if incumbent is None:
        return True, "initial_best"
    candidate_agg = candidate["aggregate"]
    incumbent_agg = incumbent["aggregate"]
    if candidate_agg["return"] > incumbent_agg["return"]:
        return True, "higher_mean_return"
    if candidate_agg["return"] < incumbent_agg["return"]:
        return False, "kept_existing_best"
    if candidate_agg["tracking_error_scalar"] < incumbent_agg["tracking_error_scalar"]:
        return True, "tie_breaker_lower_tracking_error_scalar"
    if candidate_agg["tracking_error_scalar"] > incumbent_agg["tracking_error_scalar"]:
        return False, "kept_existing_best"
    if candidate_agg["fall_rate"] < incumbent_agg["fall_rate"]:
        return True, "tie_breaker_lower_fall_rate"
    return False, "kept_existing_best"



def json_dump(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")



def jsonl_append(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")



def main() -> None:
    args = parse_args()
    cfg = load_cfg(args.config)

    total_timesteps = get_protocol_timesteps(cfg, args.protocol, args.timesteps)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    run_dir = args.run_dir or Path(cfg["logging"]["root_dir"]) / args.protocol
    run_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env

    policy_cfg = cfg["policy"]
    ppo_cfg = cfg["ppo"]
    protocol_cfg = cfg["protocol"]
    logging_cfg = cfg["logging"]

    train_env = make_vec_env(lambda: make_env(args.task), n_envs=int(cfg["env"]["num_envs"]), seed=seed)
    eval_env = make_env(args.task)

    model = PPO(
        policy=policy_cfg["type"],
        env=train_env,
        learning_rate=float(ppo_cfg["learning_rate"]),
        n_steps=int(ppo_cfg["n_steps"]),
        batch_size=int(ppo_cfg["batch_size"]),
        n_epochs=int(ppo_cfg["n_epochs"]),
        gamma=float(ppo_cfg["gamma"]),
        gae_lambda=float(ppo_cfg["gae_lambda"]),
        clip_range=float(ppo_cfg["clip_range"]),
        ent_coef=float(ppo_cfg["ent_coef"]),
        vf_coef=float(ppo_cfg["vf_coef"]),
        max_grad_norm=float(ppo_cfg["max_grad_norm"]),
        tensorboard_log=logging_cfg["tensorboard_log"],
        seed=seed,
        device=args.device,
        verbose=1,
        policy_kwargs={
            "net_arch": list(policy_cfg["net_arch"]),
            "activation_fn": activation_fn(str(policy_cfg["activation_fn"])),
        },
    )

    latest_path = run_dir / protocol_cfg["checkpoint_latest_name"]
    best_path = run_dir / protocol_cfg["checkpoint_best_name"]
    eval_metrics_path = run_dir / logging_cfg["eval_metrics_file"]
    final_scorecard_path = run_dir / logging_cfg["final_scorecard_file"]
    manifest = {
        "config": str(args.config),
        "protocol": args.protocol,
        "total_timesteps": total_timesteps,
        "seed": seed,
        "package_import": "spider_mujoco",
        "gym_id": args.task,
        "command_schema": COMMAND_SCHEMA,
        "command_vy_expected": 0.0,
    }
    json_dump(run_dir / "train_manifest.json", manifest)

    eval_every = int(protocol_cfg["eval_freq"])
    eval_episodes = int(protocol_cfg["eval_episodes"])
    deterministic_eval = bool(protocol_cfg["deterministic_eval"])
    best_scorecard: dict[str, Any] | None = None
    last_eval_step = 0

    while model.num_timesteps < total_timesteps:
        next_target = min(total_timesteps, last_eval_step + eval_every)
        step_budget = next_target - model.num_timesteps
        if step_budget <= 0:
            break
        model.learn(total_timesteps=step_budget, reset_num_timesteps=False, tb_log_name=args.protocol)
        last_eval_step = next_target
        model.save(str(latest_path))
        scorecard = evaluate_model(model, eval_env, episodes=eval_episodes, deterministic=deterministic_eval)
        better, reason = is_better(scorecard, best_scorecard)
        row = {
            "checkpoint_role": "latest",
            "eval_mean_return": scorecard["aggregate"]["return"],
            "tracking_error_scalar": scorecard["aggregate"]["tracking_error_scalar"],
            "fall_rate": scorecard["aggregate"]["fall_rate"],
            "mean_vx_tracking_error": scorecard["aggregate"]["mean_vx_tracking_error"],
            "mean_wz_tracking_error": scorecard["aggregate"]["mean_wz_tracking_error"],
            "best_checkpoint_selection_reason": reason,
            "timesteps": model.num_timesteps,
        }
        jsonl_append(eval_metrics_path, row)
        model.logger.record("eval/mean_return", row["eval_mean_return"])
        model.logger.record("eval/tracking_error_scalar", row["tracking_error_scalar"])
        model.logger.record("eval/fall_rate", row["fall_rate"])
        model.logger.record("eval/mean_vx_tracking_error", row["mean_vx_tracking_error"])
        model.logger.record("eval/mean_wz_tracking_error", row["mean_wz_tracking_error"])
        model.logger.dump(model.num_timesteps)
        if better:
            model.save(str(best_path))
            best_scorecard = scorecard
            json_dump(final_scorecard_path, {"checkpoint_role": "best", "timesteps": model.num_timesteps, **scorecard})

    model.save(str(latest_path))
    final_latest = evaluate_model(model, eval_env, episodes=eval_episodes, deterministic=deterministic_eval)
    better, reason = is_better(final_latest, best_scorecard)
    row = {
        "checkpoint_role": "latest",
        "eval_mean_return": final_latest["aggregate"]["return"],
        "tracking_error_scalar": final_latest["aggregate"]["tracking_error_scalar"],
        "fall_rate": final_latest["aggregate"]["fall_rate"],
        "mean_vx_tracking_error": final_latest["aggregate"]["mean_vx_tracking_error"],
        "mean_wz_tracking_error": final_latest["aggregate"]["mean_wz_tracking_error"],
        "best_checkpoint_selection_reason": reason,
        "timesteps": model.num_timesteps,
    }
    jsonl_append(eval_metrics_path, row)
    if better:
        model.save(str(best_path))
        best_scorecard = final_latest
        json_dump(final_scorecard_path, {"checkpoint_role": "best", "timesteps": model.num_timesteps, **final_latest})
    elif best_scorecard is not None:
        json_dump(final_scorecard_path, {"checkpoint_role": "best", "timesteps": model.num_timesteps, **best_scorecard})
    else:
        model.save(str(best_path))
        json_dump(final_scorecard_path, {"checkpoint_role": "best", "timesteps": model.num_timesteps, **final_latest})

    eval_env.close()
    train_env.close()


if __name__ == "__main__":
    main()
