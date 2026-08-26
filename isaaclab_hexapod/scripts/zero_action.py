#!/usr/bin/env python3
"""Run a bounded headless smoke of the Hexapod DirectRLEnv."""

import argparse
import traceback

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import hexapod_isaaclab  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def main() -> None:
    print("[hexapod] creating environment", flush=True)
    cfg = parse_env_cfg(
        "Hexapod-Firmware-Flat-Direct-v0", device=args.device, num_envs=1
    )
    env = gym.make("Hexapod-Firmware-Flat-Direct-v0", cfg=cfg)
    print("[hexapod] environment created", flush=True)
    obs, _ = env.reset()
    print("[hexapod] environment reset", flush=True)
    reward = torch.zeros(1, device=env.unwrapped.device)
    for _ in range(args.steps):
        action = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
        obs, reward, terminated, truncated, _ = env.step(action)
        if torch.any(terminated | truncated):
            break
    print(
        "HEXAPOD_SMOKE_OK",
        f"steps={int(env.unwrapped.episode_length_buf.max().item())}",
        f"obs={tuple(obs['policy'].shape)}",
        f"reward={float(reward.mean().item()):.6f}",
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
