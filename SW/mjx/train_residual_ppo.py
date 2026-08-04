from __future__ import annotations

"""Train the residual-RL locomotion policy with a small PPO loop.

This trainer now persists progress incrementally so long runs are recoverable:
- metrics JSON is rewritten after every PPO update,
- the current optimizer state is checkpointed to a "latest" file,
- the best-so-far policy is checkpointed immediately when reward improves,
- SIGINT/SIGTERM request a clean stop after the current update,
- optional W&B logging mirrors scalar metrics and training artifacts.
"""

import argparse
import hashlib
import json
import signal
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hexapod_mjx.model import STAND_POSE, load_hexapod_model, repo_root_from
from hexapod_mjx.residual_controller import ACTION_DIM, ResidualControllerConfig, build_residual_controller

from hexapod_mjx.residual_env import (
    CommandCurriculumConfig,
    ResidualEnvConfig,
    command_sampling_config,
    joint_group_index,
    reset_env,
    step_env,
)
from hexapod_mjx.residual_rl import (
    PPOConfig,
    RolloutBatch,
    TrainState,
    compute_gae,
    flatten_rollout,
    init_train_state,
    load_checkpoint,
    minibatches,
    ppo_update,
    sample_action,
    save_checkpoint,
    value_predict,
    write_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a residual-RL hexapod locomotion policy with PPO.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--num-envs", "--num-env", dest="num_envs", type=int, default=24)
    parser.add_argument("--rollout-steps", type=int, default=48)
    parser.add_argument("--num-updates", "--num-update", dest="num_updates", type=int, default=4)
    parser.add_argument("--ppo-epochs", type=int, default=3)
    parser.add_argument("--minibatch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-path", type=str, default="SW/mjx/artifacts/residual_rl_policy.pkl")
    parser.add_argument("--latest-output-path", type=str, default=None)
    parser.add_argument("--metrics-path", type=str, default="SW/mjx/artifacts/residual_rl_metrics.json")
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--save-every-updates", type=int, default=1)
    parser.add_argument("--termination-contact-z", type=float, default=0.0)
    parser.add_argument("--contact-model", choices=("mesh", "hybrid"), default="hybrid", help="Collision model for training. `hybrid` is the fast proxy model; `mesh` keeps original mesh collisions but is much heavier.")
    parser.add_argument("--joint-limit-margin-1", type=float, default=0.0, help="Tighten the joint-1 range by this many radians on each side.")
    parser.add_argument("--joint-limit-margin-2", type=float, default=0.0, help="Tighten the joint-2 range by this many radians on each side.")
    parser.add_argument("--joint-limit-margin-3", type=float, default=0.0, help="Tighten the joint-3 range by this many radians on each side.")
    parser.add_argument("--command-curriculum", choices=("staged", "none"), default="staged")
    parser.add_argument("--forward-only-updates", type=int, default=120)
    parser.add_argument("--yaw-stage-updates", type=int, default=120)
    parser.add_argument("--forward-only-scale", type=float, default=0.60)
    parser.add_argument("--yaw-stage-scale", type=float, default=0.35)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="hexapod-residual-rl")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-job-type", type=str, default="mjx-train")
    parser.add_argument("--wandb-mode", type=str, default="online")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-run-id", type=str, default=None)
    return parser.parse_args()



def _resolve_repo_path(repo_root: Path, path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    path = Path(path_str)
    return path if path.is_absolute() else (repo_root / path).resolve()



def _derive_latest_output_path(output_path: Path, latest_output_path: str | None) -> Path:
    if latest_output_path is not None:
        path = Path(latest_output_path)
        return path if path.is_absolute() else path
    return output_path.with_name(f"{output_path.stem}_latest{output_path.suffix}")



def _load_history(metrics_path: Path) -> list[dict[str, float]]:
    if not metrics_path.exists():
        return []
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    history = payload.get("history", [])
    return history if isinstance(history, list) else []



def _init_wandb(repo_root: Path, args: argparse.Namespace, output_path: Path, latest_output_path: Path, metrics_path: Path):
    if not args.wandb:
        return None, None
    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in ~/.venvs/hexapod-mjx. Install it with `~/.venvs/hexapod-mjx/bin/python -m pip install wandb`."
        ) from exc

    run_name = args.wandb_name or output_path.stem
    run_id = args.wandb_run_id or output_path.stem
    tags = [tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()]
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        group=args.wandb_group,
        job_type=args.wandb_job_type,
        mode=args.wandb_mode,
        name=run_name,
        id=run_id,
        resume="allow",
        tags=tags or None,
        config={
            "repo_root": str(repo_root),
            "num_envs": args.num_envs,
            "rollout_steps": args.rollout_steps,
            "num_updates": args.num_updates,
            "ppo_epochs": args.ppo_epochs,
            "minibatch_size": args.minibatch_size,
            "learning_rate": args.learning_rate,
            "hidden_size": args.hidden_size,
            "seed": args.seed,
            "joint_limit_margin_1": args.joint_limit_margin_1,
            "joint_limit_margin_2": args.joint_limit_margin_2,
            "joint_limit_margin_3": args.joint_limit_margin_3,
            "output_path": str(output_path),
            "latest_output_path": str(latest_output_path),
            "metrics_path": str(metrics_path),
            "simulator": "mjx",
        },
    )
    wandb.define_metric("update")
    wandb.define_metric("*", step_metric="update")
    wandb_metadata = {
        "enabled": True,
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "group": args.wandb_group,
        "job_type": args.wandb_job_type,
        "mode": args.wandb_mode,
        "name": run_name,
        "run_id": run.id,
        "tags": tags,
    }
    return run, wandb_metadata



def _stand_pose_hash() -> str:
    payload = json.dumps(STAND_POSE, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _code_signature(repo_root: Path) -> dict[str, Any]:
    rel_paths = [
        Path("SW/mjx/hexapod_mjx/model.py"),
        Path("SW/mjx/hexapod_mjx/residual_controller.py"),
        Path("SW/mjx/hexapod_mjx/residual_env.py"),
        Path("SW/mjx/hexapod_mjx/residual_rl.py"),
        Path("SW/mjx/train_residual_ppo.py"),
    ]
    digest = hashlib.sha256()
    files: list[str] = []
    for rel_path in rel_paths:
        abs_path = (repo_root / rel_path).resolve()
        files.append(str(rel_path))
        digest.update(str(rel_path).encode("utf-8"))
        digest.update(b"\0")
        digest.update(abs_path.read_bytes())
        digest.update(b"\0")
    return {
        "hash": digest.hexdigest(),
        "files": files,
    }


def _checkpoint_metadata(
    repo_root: Path,
    args: argparse.Namespace,
    controller_reset_root_height: float,
    obs_dim: int,
    *,
    updates_seen: int,
    best_update: int,
    best_mean_reward: float,
    checkpoint_kind: str,
    interrupted: bool,
    completed: bool,
    wandb_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "updates_seen": updates_seen,
        "best_update": best_update,
        "best_mean_reward": best_mean_reward,
        "obs_dim": obs_dim,
        "action_dim": ACTION_DIM,
        "controller_reset_root_height": controller_reset_root_height,
        "checkpoint_kind": checkpoint_kind,
        "interrupted": interrupted,
        "completed": completed,
        "wandb": wandb_metadata,
        "stand_pose": STAND_POSE,
        "stand_pose_hash": _stand_pose_hash(),
        "code_signature": _code_signature(repo_root),
        "ppo_config": vars(args),
    }



def _write_progress_metrics(
    metrics_path: Path,
    ppo_config: PPOConfig,
    history: list[dict[str, float]],
    *,
    best_mean_reward: float,
    best_update: int,
    interrupted: bool,
    completed: bool,
) -> None:
    history_for_write = [dict(item) for item in history]
    if history_for_write:
        history_for_write[-1]["best_mean_reward"] = best_mean_reward
        history_for_write[-1]["best_update"] = float(best_update)
        history_for_write[-1]["interrupted"] = float(interrupted)
        history_for_write[-1]["completed"] = float(completed)
    write_metrics(metrics_path, ppo_config, history_for_write)



def _log_wandb_artifacts(run, output_path: Path, latest_output_path: Path, metrics_path: Path, interrupted: bool) -> None:
    if run is None:
        return
    import wandb  # type: ignore

    artifact = wandb.Artifact(f"{run.name}-train-artifacts", type="mjx-train")
    for path in [output_path, latest_output_path, metrics_path]:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.summary["interrupted"] = interrupted
    run.summary["best_checkpoint"] = str(output_path)
    run.summary["latest_checkpoint"] = str(latest_output_path)
    run.summary["metrics_path"] = str(metrics_path)



def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    output_path = _resolve_repo_path(repo_root, args.output_path)
    assert output_path is not None
    latest_output_path = _derive_latest_output_path(output_path, args.latest_output_path)
    if not latest_output_path.is_absolute():
        latest_output_path = (repo_root / latest_output_path).resolve()
    metrics_path = _resolve_repo_path(repo_root, args.metrics_path)
    assert metrics_path is not None

    bundle = load_hexapod_model(repo_root, contact_mode=args.contact_model)
    controller_config = ResidualControllerConfig(
        joint_limit_margin=(
            args.joint_limit_margin_1,
            args.joint_limit_margin_2,
            args.joint_limit_margin_3,
        )
    )
    controller_bundle = build_residual_controller(bundle, controller_config)
    env_config = ResidualEnvConfig(
        episode_steps=args.rollout_steps,
        termination_contact_z=args.termination_contact_z,
    )
    curriculum_config = CommandCurriculumConfig(
        mode=args.command_curriculum,
        forward_only_updates=args.forward_only_updates,
        yaw_stage_updates=args.yaw_stage_updates,
        forward_only_scale=args.forward_only_scale,
        yaw_stage_scale=args.yaw_stage_scale,
    )
    ppo_config = PPOConfig(
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        num_updates=args.num_updates,
        minibatch_size=args.minibatch_size,
        ppo_epochs=args.ppo_epochs,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        seed=args.seed,
        output_path=args.output_path,
        metrics_path=args.metrics_path,
    )

    stop_state = {"requested": False, "signal": None}

    def request_stop(signum: int, _frame) -> None:
        stop_state["requested"] = True
        stop_state["signal"] = signum
        print(f"stop_requested: signal={signum} (will save progress after the current update)")

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    key = jax.random.key(args.seed)
    _, warmup_obs = reset_env(
        bundle,
        controller_bundle,
        controller_config,
        key,
        1,
        command_sampling_config(controller_config, curriculum_config, 1),
    )
    obs_dim = int(warmup_obs.shape[-1])

    completed_updates = 0
    history = _load_history(metrics_path)
    best_mean_reward = float("-inf")
    best_update = 0

    if args.resume_path:
        resume_path = _resolve_repo_path(repo_root, args.resume_path)
        assert resume_path is not None
        train_state, resume_metadata = load_checkpoint(resume_path)
        completed_updates = int(resume_metadata.get("updates_seen", 0))
        print(f"resume_path: {resume_path}")
        print(f"resume_updates_seen: {completed_updates}")
    else:
        train_state = init_train_state(obs_dim, ACTION_DIM, ppo_config)
        resume_metadata = {}

    if output_path.exists():
        best_train_state, best_metadata = load_checkpoint(output_path)
        best_mean_reward = float(best_metadata.get("best_mean_reward", float("-inf")))
        best_update = int(best_metadata.get("best_update", 0))
    else:
        best_train_state = train_state
        best_mean_reward = float(resume_metadata.get("best_mean_reward", float("-inf")))
        best_update = int(resume_metadata.get("best_update", 0))

    if history:
        last_history = history[-1]
        best_mean_reward = max(best_mean_reward, float(last_history.get("best_mean_reward", best_mean_reward)))
        best_update = max(best_update, int(last_history.get("best_update", best_update)))

    wandb_run, wandb_metadata = _init_wandb(repo_root, args, output_path, latest_output_path, metrics_path)

    group_index = joint_group_index(bundle)
    key, _ = jax.random.split(key)
    step_fn = jax.jit(
        lambda state, action: step_env(
            bundle,
            controller_bundle,
            controller_config,
            env_config,
            group_index,
            state,
            action,
        )
    )
    sample_fn = jax.jit(sample_action)
    value_fn = jax.jit(value_predict)

    def sampling_for_update(update_index: int):
        return command_sampling_config(controller_config, curriculum_config, update_index)

    try:
        for update_idx in range(args.num_updates):
            absolute_update = completed_updates + update_idx + 1
            key, reset_key = jax.random.split(key)
            state, obs = reset_env(
                bundle,
                controller_bundle,
                controller_config,
                reset_key,
                args.num_envs,
                sampling_for_update(absolute_update),
            )
            obs_list = []
            action_list = []
            log_prob_list = []
            reward_list = []
            done_list = []
            value_list = []
            metrics_list = []

            for _rollout_step in range(args.rollout_steps):
                key, action_key = jax.random.split(key)
                action, log_prob, value = sample_fn(train_state.params, obs, action_key)
                next_state, next_obs, reward, done, metrics = step_fn(state, action)
                obs_list.append(obs)
                action_list.append(action)
                log_prob_list.append(log_prob)
                reward_list.append(reward)
                done_list.append(done)
                value_list.append(value)
                metrics_list.append(metrics)
                state = next_state
                obs = next_obs

            last_values = value_fn(train_state.params, obs)
            rollout = RolloutBatch(
                obs=jnp.stack(obs_list, axis=0),
                actions=jnp.stack(action_list, axis=0),
                log_prob=jnp.stack(log_prob_list, axis=0),
                rewards=jnp.stack(reward_list, axis=0),
                dones=jnp.stack(done_list, axis=0),
                values=jnp.stack(value_list, axis=0),
                metrics=jnp.stack(metrics_list, axis=0),
                last_obs=obs,
            )
            advantages, returns = compute_gae(rollout.rewards, rollout.dones, rollout.values, last_values, ppo_config)
            policy_batch = flatten_rollout(rollout, advantages, returns)

            update_metrics: dict[str, float] = {}
            for epoch_idx in range(args.ppo_epochs):
                for batch in minibatches(policy_batch, ppo_config, seed=args.seed + absolute_update * 101 + epoch_idx):
                    train_state, update_metrics = ppo_update(train_state, batch, ppo_config)

            mean_reward = float(jnp.mean(rollout.rewards))
            mean_done = float(jnp.mean(rollout.dones))
            metric_means = jnp.mean(rollout.metrics.reshape(-1, rollout.metrics.shape[-1]), axis=0)
            summary = {
                "update": float(absolute_update),
                "mean_reward": mean_reward,
                "mean_done": mean_done,
                "velocity_reward": float(metric_means[0]),
                "yaw_reward": float(metric_means[1]),
                "attitude_reward": float(metric_means[2]),
                "height_reward": float(metric_means[3]),
                "slip_cost": float(metric_means[4]),
                "control_cost": float(metric_means[5]),
                "action_cost": float(metric_means[6]),
                "forward_velocity": float(metric_means[7]),
                "lateral_velocity": float(metric_means[8]),
                "yaw_rate": float(metric_means[9]),
                "body_contact": float(metric_means[10]),
            }
            summary.update(update_metrics)
            history.append(summary)

            best_improved = mean_reward > best_mean_reward
            if best_improved:
                best_mean_reward = mean_reward
                best_update = absolute_update
                best_train_state = train_state

            if best_improved or not output_path.exists():
                save_checkpoint(
                    output_path,
                    best_train_state,
                    _checkpoint_metadata(
                        repo_root,
                        args,
                        float(controller_bundle.reset_root_height),
                        obs_dim,

                        updates_seen=absolute_update,
                        best_update=best_update,
                        best_mean_reward=best_mean_reward,
                        checkpoint_kind="best",
                        interrupted=False,
                        completed=False,
                        wandb_metadata=wandb_metadata,
                    ),
                )

            if args.save_every_updates > 0 and (absolute_update % args.save_every_updates == 0 or stop_state["requested"]):
                save_checkpoint(
                    latest_output_path,
                    train_state,
                    _checkpoint_metadata(
                        repo_root,
                        args,
                        float(controller_bundle.reset_root_height),
                        obs_dim,

                        updates_seen=absolute_update,
                        best_update=best_update,
                        best_mean_reward=best_mean_reward,
                        checkpoint_kind="latest",
                        interrupted=stop_state["requested"],
                        completed=False,
                        wandb_metadata=wandb_metadata,
                    ),
                )
                _write_progress_metrics(
                    metrics_path,
                    ppo_config,
                    history,
                    best_mean_reward=best_mean_reward,
                    best_update=best_update,
                    interrupted=stop_state["requested"],
                    completed=False,
                )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        **summary,
                        "best_mean_reward": best_mean_reward,
                        "best_update": best_update,
                    },
                    step=absolute_update,
                )

            print(
                f"update={absolute_update} mean_reward={summary['mean_reward']:.4f} "
                f"done_rate={summary['mean_done']:.4f} actor_loss={summary.get('actor_loss', 0.0):.4f} "
                f"value_loss={summary.get('value_loss', 0.0):.4f}"
            )

            if stop_state["requested"]:
                print(f"stopping_early_after_update: {absolute_update}")
                break

        total_updates_seen = completed_updates + args.num_updates if not stop_state["requested"] else int(history[-1]["update"]) if history else completed_updates
        completed = not stop_state["requested"]
        interrupted = stop_state["requested"]

        save_checkpoint(
            latest_output_path,
            train_state,
            _checkpoint_metadata(
                repo_root,
                args,
                float(controller_bundle.reset_root_height),
                obs_dim,

                updates_seen=total_updates_seen,
                best_update=best_update,
                best_mean_reward=best_mean_reward,
                checkpoint_kind="latest",
                interrupted=interrupted,
                completed=completed,
                wandb_metadata=wandb_metadata,
            ),
        )
        if history:
            save_checkpoint(
                output_path,
                best_train_state,
                _checkpoint_metadata(
                    repo_root,
                    args,
                    float(controller_bundle.reset_root_height),
                    obs_dim,
                    updates_seen=total_updates_seen,
                    best_update=best_update,
                    best_mean_reward=best_mean_reward,
                    checkpoint_kind="best",
                    interrupted=interrupted,
                    completed=completed,
                    wandb_metadata=wandb_metadata,
                ),
            )
        _write_progress_metrics(
            metrics_path,
            ppo_config,
            history,
            best_mean_reward=best_mean_reward,
            best_update=best_update,
            interrupted=interrupted,
            completed=completed,
        )

        if wandb_run is not None:
            wandb_run.summary["best_mean_reward"] = best_mean_reward
            wandb_run.summary["best_update"] = best_update
            if history:
                wandb_run.summary["final_mean_reward"] = history[-1]["mean_reward"]
            _log_wandb_artifacts(wandb_run, output_path, latest_output_path, metrics_path, interrupted)

        print(f"repo_root: {repo_root}")
        print(f"policy_checkpoint: {output_path}")
        print(f"latest_checkpoint: {latest_output_path}")
        print(f"metrics_path: {metrics_path}")
        print(f"best_policy_checkpoint: {output_path}")
        if history:
            print(f"best_mean_reward: {best_mean_reward:.4f} (update={best_update})")
            print(f"final_mean_reward: {history[-1]['mean_reward']:.4f}")
        print(f"completed: {completed}")
        print(f"interrupted: {interrupted}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
