#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
EXPECTED_CONTACT_SITES = [
    "LF_motor_horn_3_1_contact_site",
    "LM_motor_horn_3_1_contact_site",
    "LB_motor_horn_3_1_contact_site",
    "RF_motor_horn_3_1_contact_site",
    "RM_motor_horn_3_1_contact_site",
    "RB_motor_horn_3_1_contact_site",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load the standalone MuJoCo hexapedal model/env and print the approved runtime contract.")
    parser.add_argument(
        "--config",
        type=Path,
        default=_REPO_ROOT / "source" / "spider_rl" / "spider_mujoco" / "hexapedal_direct" / "agents" / "sb3_ppo_cfg.yaml",
        help="Path to the SB3 PPO YAML config for contract cross-checking.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=ENV_ID,
        help="Gymnasium environment id to load.",
    )
    parser.add_argument(
        "--regenerate-check",
        action="store_true",
        help="Regenerate the packaged MJCF/source-map assets before loading the environment.",
    )
    return parser.parse_args()



def load_cfg(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in config: {path}")
    return data



def bootstrap_spider_mujoco() -> None:
    import spider_mujoco  # noqa: F401



def unwrap_attr(env: gym.Env[Any, Any], name: str) -> Any:
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        if hasattr(current, name):
            return getattr(current, name)
        current = getattr(current, "env", None)
    return None



def collect_site_names(model: Any) -> list[str]:
    if model is None:
        return []
    import mujoco

    names: list[str] = []
    for idx in range(int(getattr(model, "nsite", 0))):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, idx)
        if name is not None:
            names.append(name)
    return names



def main() -> None:
    args = parse_args()
    import gymnasium as gym

    cfg = load_cfg(args.config)
    regeneration_status = "not_requested"
    if args.regenerate_check:
        from spider_mujoco.hexapedal_direct.model_builder import write_hexapedal_assets

        try:
            write_hexapedal_assets()
        except FileNotFoundError as exc:
            regeneration_status = f"skipped:{exc}"
        else:
            regeneration_status = "regenerated"
    bootstrap_spider_mujoco()
    env = gym.make(args.task)
    obs, info = env.reset(seed=int(cfg.get("seed", 42)))
    zero_action = np.zeros(env.action_space.shape, dtype=env.action_space.dtype)
    _, _, _, _, step_info = env.step(zero_action)

    model = unwrap_attr(env, "model")
    data = unwrap_attr(env, "data")
    site_names = collect_site_names(model)
    missing_sites = [name for name in EXPECTED_CONTACT_SITES if name not in site_names]

    contract = {
        "package_import": "spider_mujoco",
        "gym_id": args.task,
        "command_schema": COMMAND_SCHEMA,
        "command_vy_expected": 0.0,
        "regeneration_status": regeneration_status,
        "observation_dim": int(np.asarray(obs).shape[-1]),
        "action_dim": int(np.prod(env.action_space.shape, dtype=np.int64)),
        "action_space": {
            "low_shape": list(np.asarray(env.action_space.low).shape),
            "high_shape": list(np.asarray(env.action_space.high).shape),
        },
        "mujoco_model": {
            "compiled": model is not None and data is not None,
            "nq": int(getattr(model, "nq", 0)) if model is not None else 0,
            "nv": int(getattr(model, "nv", 0)) if model is not None else 0,
            "nu": int(getattr(model, "nu", 0)) if model is not None else 0,
            "nbody": int(getattr(model, "nbody", 0)) if model is not None else 0,
            "nsite": int(getattr(model, "nsite", 0)) if model is not None else 0,
        },
        "desired_contact_site_names": EXPECTED_CONTACT_SITES,
        "desired_contact_site_names_present": sorted(set(EXPECTED_CONTACT_SITES) - set(missing_sites)),
        "desired_contact_site_names_missing": missing_sites,
        "undesired_contact_body_patterns": ["base_link", ".*_motor_horn_1_1", ".*_motor_horn_2_1"],
        "episode_metric_keys_after_step": sorted(step_info.get("episode", {}).keys()) if isinstance(step_info.get("episode"), dict) else [],
        "config_contract": cfg.get("contract", {}),
    }
    print(json.dumps(contract, indent=2, sort_keys=True))
    env.close()


if __name__ == "__main__":
    main()
