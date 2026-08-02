from __future__ import annotations

"""Cross-Entropy Method search for a minimal open-loop tripod gait.

This module does *not* train a neural network. Instead it searches a tiny,
interpretable parameter vector that drives a hand-written sinusoidal gait. That
makes it a good first stage because failures stay explainable:

- if the robot falls immediately, look at the model/reset/contact assumptions,
- if it shuffles but does not move, look at gait timing/amplitudes,
- only later do we pay the complexity cost of full RL policy learning.
"""

from dataclasses import dataclass, asdict
import json

import jax
import jax.numpy as jnp
from mujoco import mjx
import numpy as np

from .model import HexapodModelBundle, estimate_standing_root_height

# The optimizer always treats parameters as an ordered vector. Keeping the names
# beside that vector is important because the saved JSON should still be human-
# readable after the search is done.
PARAMETER_NAMES = (
    "frequency_hz",
    "hip1_amplitude",
    "hip2_amplitude",
    "knee_amplitude",
    "hip2_bias_delta",
    "knee_bias_delta",
    "knee_phase_offset",
)

# Search box for each parameter. These are conservative on purpose: a smaller,
# physically sane search space is usually better than giving CEM room to find
# violent but unstable motions.
LOWER_BOUNDS = jnp.array([0.5, 0.0, 0.1, 0.1, -0.5, -0.8, -jnp.pi], dtype=jnp.float32)
UPPER_BOUNDS = jnp.array([4.0, 0.7, 1.0, 1.2, 0.5, 0.8, jnp.pi], dtype=jnp.float32)

# Initial distribution for CEM. The mean is our best hand-written guess; the
# standard deviation encodes how far we are initially willing to explore away
# from that guess.
INITIAL_MEAN = jnp.array([1.8, 0.10, 0.32, 0.44, 0.00, 0.00, 1.10], dtype=jnp.float32)
INITIAL_STD = jnp.array([0.45, 0.08, 0.12, 0.16, 0.08, 0.12, 0.35], dtype=jnp.float32)

# Per-joint-group PD gains and torque limits. There are only three controller
# channels because every leg shares the same per-suffix controller structure.
PD_KP = jnp.array([10.0, 34.0, 34.0], dtype=jnp.float32)
PD_KD = jnp.array([0.6, 1.6, 1.6], dtype=jnp.float32)
TORQUE_LIMIT = jnp.array([8.0, 24.0, 24.0], dtype=jnp.float32)


@dataclass(frozen=True)
class CEMConfig:
    """User-visible configuration for one search run."""

    population_size: int = 256
    elite_count: int = 32
    num_iterations: int = 20
    rollout_steps: int = 600
    seed: int = 0
    action_repeat: int = 2
    base_height: float | None = None
    output_path: str = "SW/mjx/artifacts/hexapod_cem_result.json"
    resume_result_path: str | None = None
    resume_std_scale: float = 0.5


@dataclass(frozen=True)
class SearchResult:
    """Compact summary returned to the CLI after the JSON artifact is written."""

    best_score: float
    best_params: dict[str, float]
    mean_params: dict[str, float]
    score_history: list[float]
    output_path: str


def _quat_up_z(quat: jnp.ndarray) -> jnp.ndarray:
    """Return the world-space Z component of the body-up axis.

    1.0 means perfectly upright, 0.0 means rolled/pitched by 90 degrees, and a
    negative value means upside down.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return 1.0 - 2.0 * (x * x + y * y)



def _quat_yaw(quat: jnp.ndarray) -> jnp.ndarray:
    """Extract yaw so the score can penalize spinning in place."""
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return jnp.arctan2(siny_cosp, cosy_cosp)



def _make_batched_data(bundle: HexapodModelBundle, batch_size: int, *, base_height: float) -> mjx.Data:
    """Create one MJX ``Data`` struct per candidate in the population.

    Every candidate starts from the exact same floating-base pose. Only the gait
    parameters differ; the initial robot state does not.
    """
    qpos0 = np.zeros((batch_size, bundle.model.nq), dtype=np.float32)
    qvel0 = np.zeros((batch_size, bundle.model.nv), dtype=np.float32)
    qpos0[:, 0:3] = np.array([0.0, 0.0, base_height], dtype=np.float32)
    qpos0[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    qpos0[:, bundle.joint_qpos_adr] = np.asarray(bundle.default_joint_pose)

    # ``mjx.make_data`` makes one blank simulation state. ``vmap`` duplicates it
    # efficiently so the whole population can be rolled out in parallel.
    batch_data = jax.vmap(lambda _: mjx.make_data(bundle.mjx_model))(jnp.arange(batch_size, dtype=jnp.int32))
    batch_data = batch_data.replace(
        qpos=jnp.asarray(qpos0),
        qvel=jnp.asarray(qvel0),
    )
    batch_data = jax.vmap(mjx.forward, in_axes=(None, 0))(bundle.mjx_model, batch_data)
    return batch_data



def _joint_targets(bundle: HexapodModelBundle, params: jnp.ndarray, time_s: jnp.ndarray) -> jnp.ndarray:
    """Compute the desired joint angles for every candidate at one control time.

    ``params`` has shape ``[batch, num_params]`` and the returned target array
    has shape ``[batch, num_joints]``. Each leg shares the same waveform shape;
    tripod phase offsets decide which legs are in the stance vs swing half-cycle.
    """
    phase = 2.0 * jnp.pi * params[:, 0:1] * time_s + bundle.tripod_phase_offset[None, :]
    knee_phase = phase + params[:, 6:7]
    group = bundle.joint_group_index[None, :]

    hip1 = params[:, 1:2] * jnp.sin(phase)
    hip2 = params[:, 4:5] + params[:, 2:3] * jnp.sin(phase)
    knee = params[:, 5:6] + params[:, 3:4] * jnp.sin(knee_phase)

    return (
        bundle.default_joint_pose[None, :]
        + jnp.where(group == 0, hip1, 0.0)
        + jnp.where(group == 1, hip2, 0.0)
        + jnp.where(group == 2, knee, 0.0)
    )



def _rollout_scores(
    bundle: HexapodModelBundle,
    params: jnp.ndarray,
    rollout_steps: int,
    action_repeat: int,
    *,
    base_height: float,
) -> jnp.ndarray:
    """Roll out the full candidate batch and return one scalar score per row.

    The outer loop runs at the controller rate. Inside that loop we hold the
    same torque command for ``action_repeat`` physics steps, matching a typical
    low-level controller that updates more slowly than the physics integrator.
    """
    batch_size = params.shape[0]
    qpos_adr = jnp.asarray(bundle.joint_qpos_adr)
    dof_adr = jnp.asarray(bundle.joint_dof_adr)
    group = bundle.joint_group_index.astype(jnp.int32)
    data0 = _make_batched_data(bundle, batch_size, base_height=base_height)
    sim_dt = float(bundle.model.opt.timestep)

    def outer_step(carry, outer_idx):
        data, torque_sum, up_sum = carry
        time_s = outer_idx.astype(jnp.float32) * (sim_dt * action_repeat)
        desired = _joint_targets(bundle, params, time_s)
        qj = data.qpos[:, qpos_adr]
        qv = data.qvel[:, dof_adr]
        kp = PD_KP[group][None, :]
        kd = PD_KD[group][None, :]
        tau_limit = TORQUE_LIMIT[group][None, :]

        # The search space is joint-angle trajectories, not torques. The PD
        # controller is the thin execution layer that turns those targets into a
        # physically meaningful command.
        tau = kp * (desired - qj) - kd * qv
        tau = jnp.clip(tau, -tau_limit, tau_limit)

        def repeat_step(inner_data, _):
            qfrc = inner_data.qfrc_applied.at[:, :].set(0.0)
            qfrc = qfrc.at[:, dof_adr].set(tau)
            inner_data = inner_data.replace(qfrc_applied=qfrc)
            next_data = jax.vmap(mjx.step, in_axes=(None, 0))(bundle.mjx_model, inner_data)
            return next_data, None

        data, _ = jax.lax.scan(repeat_step, data, xs=None, length=action_repeat)
        up_z = _quat_up_z(data.qpos[:, 3:7])
        torque_sum = torque_sum + jnp.mean(tau * tau, axis=1)
        up_sum = up_sum + up_z
        return (data, torque_sum, up_sum), None

    outer_steps = max(1, rollout_steps // action_repeat)
    init_torque = jnp.zeros((batch_size,), dtype=jnp.float32)
    init_up = jnp.zeros((batch_size,), dtype=jnp.float32)
    (data_f, torque_sum, up_sum), _ = jax.lax.scan(
        outer_step,
        (data0, init_torque, init_up),
        xs=jnp.arange(outer_steps, dtype=jnp.int32),
    )

    progress = data_f.qpos[:, 0]
    lateral_drift = jnp.abs(data_f.qpos[:, 1])
    base_height = data_f.qpos[:, 2]
    yaw = jnp.abs(_quat_yaw(data_f.qpos[:, 3:7]))
    avg_up = up_sum / outer_steps
    control_cost = torque_sum / outer_steps
    fell = jnp.logical_or(base_height < 0.08, avg_up < 0.4).astype(jnp.float32)

    # The score deliberately stays simple and interpretable. The weights are not
    # meant to be universal reward coefficients; they are just enough to prefer
    # forward, upright, not-too-violent motion in this first search stage.
    score = (
        4.0 * progress
        + 0.6 * avg_up
        - 0.8 * lateral_drift
        - 0.3 * yaw
        - 0.0015 * control_cost
        - 3.0 * fell
    )
    return score



def _clip_params(params: jnp.ndarray) -> jnp.ndarray:
    """Project samples back into the allowed search box."""
    return jnp.clip(params, LOWER_BOUNDS, UPPER_BOUNDS)



def run_cem_search(
    bundle: HexapodModelBundle,
    config: CEMConfig,
    *,
    initial_mean: jnp.ndarray | None = None,
    initial_std: jnp.ndarray | None = None,
) -> SearchResult:
    """Run CEM, save a JSON artifact, and return the best run summary.

    ``initial_mean`` / ``initial_std`` let callers continue searching from a
    prior saved best instead of always restarting from the hand-written default.
    """
    if config.elite_count <= 0 or config.elite_count >= config.population_size:
        raise ValueError("elite_count must be > 0 and < population_size")

    if initial_mean is None:
        start_mean = INITIAL_MEAN
    else:
        start_mean = _clip_params(jnp.asarray(initial_mean, dtype=jnp.float32))
        if start_mean.shape != INITIAL_MEAN.shape:
            raise ValueError(f"initial_mean must have shape {INITIAL_MEAN.shape}, got {start_mean.shape}")

    if initial_std is None:
        start_std = INITIAL_STD
    else:
        start_std = jnp.maximum(jnp.asarray(initial_std, dtype=jnp.float32), 0.02)
        if start_std.shape != INITIAL_STD.shape:
            raise ValueError(f"initial_std must have shape {INITIAL_STD.shape}, got {start_std.shape}")

    resolved_base_height = float(config.base_height) if config.base_height is not None else estimate_standing_root_height(bundle)
    key = jax.random.key(config.seed)
    mean = start_mean
    std = start_std
    best_score = -jnp.inf
    best_params = mean
    score_history: list[float] = []

    # JIT the expensive rollout once, then feed new candidate batches through it
    # each CEM iteration. This is where MJX earns its keep.
    score_fn = jax.jit(
        lambda candidate_params: _rollout_scores(
            bundle,
            candidate_params,
            config.rollout_steps,
            config.action_repeat,
            base_height=resolved_base_height,
        )
    )

    for _ in range(config.num_iterations):
        key, sample_key = jax.random.split(key)
        noise = jax.random.normal(sample_key, shape=(config.population_size, INITIAL_MEAN.shape[0]))
        population = _clip_params(mean[None, :] + noise * std[None, :])
        scores = score_fn(population)

        # CEM keeps only the top-performing fraction, then recenters the search
        # distribution around them. No gradients are required.
        elite_idx = jnp.argsort(scores)[-config.elite_count :]
        elite = population[elite_idx]

        mean = jnp.mean(elite, axis=0)
        std = jnp.maximum(jnp.std(elite, axis=0), 0.02)

        iter_best_idx = int(jnp.argmax(scores))
        iter_best_score = float(scores[iter_best_idx])
        score_history.append(iter_best_score)
        if iter_best_score > float(best_score):
            best_score = scores[iter_best_idx]
            best_params = population[iter_best_idx]

    output_path = (bundle.repo_root / config.output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {**asdict(config), "base_height": resolved_base_height},
        "parameter_names": list(PARAMETER_NAMES),
        "resume_result_path": config.resume_result_path,
        "initial_mean_params": {name: float(value) for name, value in zip(PARAMETER_NAMES, start_mean)},
        "initial_std_params": {name: float(value) for name, value in zip(PARAMETER_NAMES, start_std)},
        "best_score": float(best_score),
        "best_params": {name: float(value) for name, value in zip(PARAMETER_NAMES, best_params)},
        "mean_params": {name: float(value) for name, value in zip(PARAMETER_NAMES, mean)},
        "score_history": score_history,
        "generated_mjcf": str(bundle.generated_mjcf_path),
        "joint_order": list(bundle.joint_names),
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return SearchResult(
        best_score=float(best_score),
        best_params=payload["best_params"],
        mean_params=payload["mean_params"],
        score_history=score_history,
        output_path=str(output_path),
    )