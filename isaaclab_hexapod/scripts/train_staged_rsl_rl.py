#!/usr/bin/env python3
"""Train levels 0..9 in one Isaac Lab run and publish each stage's best video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


ISAACLAB_ROOT = Path("/home/huro/IsaacLab")
UPSTREAM_RSL_DIR = ISAACLAB_ROOT / "scripts/reinforcement_learning/rsl_rl"
sys.path.insert(0, str(UPSTREAM_RSL_DIR))
import cli_args  # noqa: E402, isort: skip


parser = argparse.ArgumentParser(
    description="Staged RSL-RL training with level-best W&B videos."
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default=None)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--stage_iterations", type=int, default=500)
parser.add_argument("--first_stage", type=int, default=0)
parser.add_argument("--last_stage", type=int, default=9)
parser.add_argument("--stage_video_length", type=int, default=300)
parser.add_argument("--promote_threshold", type=float, default=0.50)
parser.add_argument("--easy_max_attempts", type=int, default=2)
parser.add_argument("--hard_max_attempts", type=int, default=4)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import json  # noqa: E402
import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab.envs import DirectRLEnvCfg  # noqa: E402
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
import hexapod_isaaclab  # noqa: E402, F401
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


class StageBestRunner(OnPolicyRunner):
    """RSL-RL runner that retains the strongest policy within one terrain stage."""

    stage_level: int | None = None
    stage_index: int | None = None
    stage_best_rank: tuple[float, ...]
    stage_best_path: Path | None
    stage_best_metrics: dict[str, float]

    def begin_stage(self, level: int, stage_index: int) -> None:
        self.stage_level = level
        self.stage_index = stage_index
        self.stage_best_rank = (-float("inf"),) * 4
        self.stage_best_path = None
        self.stage_best_metrics = {}
        with torch.inference_mode():
            self.env.unwrapped.set_training_level(level)
        print(
            f"[STAGE] stage={stage_index} level={level} "
            f"started at iteration={self.current_learning_iteration}"
        )

    @staticmethod
    def _mean_info(ep_infos: list[dict], key: str) -> float | None:
        values: list[float] = []
        for info in ep_infos:
            if key not in info:
                continue
            value = info[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().item()
            values.append(float(value))
        return statistics.mean(values) if values else None

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        success = self._mean_info(locs["ep_infos"], "Episode/success_rate")
        progress = self._mean_info(locs["ep_infos"], "Episode/progress_ratio")
        failure = self._mean_info(locs["ep_infos"], "Episode/failure_rate")
        mean_reward = (
            statistics.mean(locs["rewbuffer"]) if len(locs["rewbuffer"]) else None
        )
        super().log(locs, width=width, pad=pad)

        if self.stage_level is None or None in (success, progress, failure, mean_reward):
            return
        rank = (success, progress, -failure, mean_reward)
        if rank <= self.stage_best_rank:
            return

        self.stage_best_rank = rank
        self.stage_best_metrics = {
            "success_rate": success,
            "progress_ratio": progress,
            "failure_rate": failure,
            "mean_reward": mean_reward,
            "iteration": float(locs["it"]),
        }
        self.stage_best_path = Path(self.log_dir) / (
            f"stage_{self.stage_index:02d}_level_{self.stage_level:02d}_best.pt"
        )
        self._save_best_local(self.stage_best_path)
        metadata = {
            "stage": self.stage_level,
            "stage_index": self.stage_index,
            "rank": list(rank),
            **self.stage_best_metrics,
            "checkpoint": str(self.stage_best_path),
        }
        self.stage_best_path.with_suffix(".json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        for key, value in self.stage_best_metrics.items():
            self.writer.add_scalar(f"Stage/best_{key}", value, locs["it"])
        print(
            f"[STAGE BEST] stage={self.stage_index} level={self.stage_level} "
            f"iteration={locs['it']} "
            f"success={success:.3f} progress={progress:.3f} "
            f"failure={failure:.3f} reward={mean_reward:.3f}"
        )

    def _save_best_local(self, path: Path) -> None:
        payload = {
            "model_state_dict": self.alg.policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": {
                "stage": self.stage_level,
                "metrics": self.stage_best_metrics,
            },
        }
        if getattr(self.alg, "rnd", None):
            payload["rnd_state_dict"] = self.alg.rnd.state_dict()
            payload["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
        torch.save(payload, path)

    def ensure_stage_best(self) -> Path:
        if self.stage_best_path is None:
            self.stage_best_path = Path(self.log_dir) / (
                f"stage_{self.stage_index:02d}_level_{self.stage_level:02d}_best.pt"
            )
            self.stage_best_metrics = {
                "success_rate": 0.0,
                "progress_ratio": 0.0,
                "failure_rate": 1.0,
                "mean_reward": float("nan"),
                "iteration": float(self.current_learning_iteration),
            }
            self._save_best_local(self.stage_best_path)
        return self.stage_best_path


def _record_and_publish_stage_best(
    runner: StageBestRunner,
    record_env: gym.wrappers.RecordVideo,
    stage_index: int,
    level: int,
    video_length: int,
) -> Path:
    checkpoint = runner.ensure_stage_best()
    stage_end_iteration = runner.current_learning_iteration
    # RSL-RL updates observation-normalizer buffers while collecting rollouts
    # under inference mode.  PyTorch therefore requires checkpoint copies into
    # those buffers to happen under the same mode as well.
    with torch.inference_mode():
        runner.load(str(checkpoint))
        runner.env.unwrapped.set_training_level(level)
    runner.current_learning_iteration = stage_end_iteration

    video_name = f"stage-{stage_index:02d}-level-{level:02d}-best"
    video_path = Path(record_env.video_folder) / f"{video_name}.mp4"
    record_env.start_recording(video_name)
    policy = runner.get_inference_policy(device=runner.env.unwrapped.device)
    obs = runner.env.get_observations()
    with torch.inference_mode():
        for _ in range(video_length + 1):
            actions = policy(obs)
            obs, _, dones, _ = runner.env.step(actions)
            runner.alg.policy.reset(dones)
            if not record_env.recording:
                break
    if record_env.recording:
        record_env.stop_recording()
    runner.train_mode()

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"stage video was not created: {video_path}")

    if runner.logger_type == "wandb":
        import wandb

        metrics = runner.stage_best_metrics
        caption = (
            f"Isaac Lab stage best | stage={stage_index} | level={level} | "
            f"success={metrics['success_rate']:.3f} | "
            f"progress={metrics['progress_ratio']:.3f}"
        )
        payload = {
            "Stage/level": level,
            "Stage/index": stage_index,
            "Stage/best_success_rate": metrics["success_rate"],
            "Stage/best_progress_ratio": metrics["progress_ratio"],
            "Stage/best_video": wandb.Video(str(video_path), format="mp4", caption=caption),
            f"Stage/best_video_stage{stage_index}": wandb.Video(
                str(video_path), format="mp4", caption=caption
            ),
            f"Level/best_video_level{level}": wandb.Video(
                str(video_path), format="mp4", caption=caption
            ),
        }
        wandb.log(payload, step=stage_end_iteration, commit=True)
        artifact = wandb.Artifact(
            name=f"{wandb.run.id}-stage-{stage_index:02d}-level-{level:02d}-best",
            type="policy-stage-best",
            metadata={"stage": stage_index, "terrain_level": level, **metrics},
        )
        artifact.add_file(str(checkpoint), name=checkpoint.name)
        artifact.add_file(str(video_path), name=video_path.name)
        wandb.log_artifact(artifact, aliases=["latest", f"level-{level}"])
        wandb.run.summary[f"stage_{stage_index:02d}/terrain_level"] = level
        wandb.run.summary[f"stage_{stage_index:02d}/best_checkpoint"] = str(checkpoint)
        wandb.run.summary[f"stage_{stage_index:02d}/best_video"] = str(video_path)
        wandb.run.summary[f"stage_{stage_index:02d}/best_success_rate"] = metrics["success_rate"]
        runner.writer.save_model(str(checkpoint), stage_end_iteration)

    print(f"[STAGE VIDEO] stage={stage_index} level={level} path={video_path}")
    return video_path


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    if not 0 <= args_cli.first_stage <= args_cli.last_stage <= 9:
        raise ValueError("stages must satisfy 0 <= first_stage <= last_stage <= 9")
    if args_cli.stage_iterations < 1 or args_cli.stage_video_length < 1:
        raise ValueError("stage_iterations and stage_video_length must be positive")
    if not 0.0 <= args_cli.promote_threshold <= 1.0:
        raise ValueError("promote_threshold must be in [0, 1]")
    if args_cli.easy_max_attempts < 1 or args_cli.hard_max_attempts < 1:
        raise ValueError("stage attempt limits must be positive")

    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or env_cfg.sim.device

    log_root = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    log_root = log_root.resolve()
    run_folder = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_folder += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_folder
    env_cfg.log_dir = str(log_dir)
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    record_env = gym.wrappers.RecordVideo(
        env,
        video_folder=str(log_dir / "videos" / "stage_best"),
        step_trigger=lambda _step: False,
        video_length=args_cli.stage_video_length,
        name_prefix="stage-best",
        disable_logger=True,
    )
    wrapped_env = RslRlVecEnvWrapper(record_env, clip_actions=agent_cfg.clip_actions)
    runner = StageBestRunner(
        wrapped_env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device
    )
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.resume:
        resume_path = Path(agent_cfg.load_checkpoint).expanduser().resolve()
        if not resume_path.is_file():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        with torch.inference_mode():
            runner.load(str(resume_path))
        print(f"[INFO] Resumed curriculum policy from: {resume_path}")

    dump_yaml(str(log_dir / "params" / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "params" / "agent.yaml"), agent_cfg)

    start_time = time.time()
    stage_index = 0
    first_attempt = True
    for level in range(args_cli.first_stage, args_cli.last_stage + 1):
        max_attempts = (
            args_cli.easy_max_attempts if level <= 3 else args_cli.hard_max_attempts
        )
        for attempt in range(1, max_attempts + 1):
            runner.begin_stage(level, stage_index)
            runner.learn(
                num_learning_iterations=args_cli.stage_iterations,
                init_at_random_ep_len=(first_attempt and not agent_cfg.resume),
            )
            first_attempt = False
            _record_and_publish_stage_best(
                runner,
                record_env,
                stage_index,
                level,
                args_cli.stage_video_length,
            )
            success = runner.stage_best_metrics["success_rate"]
            promoted = success >= args_cli.promote_threshold
            print(
                f"[COMPETENCE] stage={stage_index} level={level} "
                f"attempt={attempt}/{max_attempts} success={success:.3f} "
                f"threshold={args_cli.promote_threshold:.3f} promoted={promoted}"
            )
            if runner.logger_type == "wandb":
                import wandb

                wandb.log(
                    {
                        "Curriculum/stage": stage_index,
                        "Curriculum/terrain_level": level,
                        "Curriculum/attempt": attempt,
                        "Curriculum/promoted": float(promoted),
                    },
                    step=runner.current_learning_iteration,
                )
            stage_index += 1
            if promoted or attempt == max_attempts:
                break
            runner.current_learning_iteration += 1
        if level < args_cli.last_stage:
            runner.current_learning_iteration += 1

    print(f"Training time: {time.time() - start_time:.1f} seconds")
    if runner.writer is not None and hasattr(runner.writer, "stop"):
        runner.writer.stop()
    wrapped_env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
