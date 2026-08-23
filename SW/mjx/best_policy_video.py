"""Render a deterministic MJX residual-policy rollout to an animated GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import jax.numpy as jp
import numpy as np


STAGE_LABELS = {
    0: "Stage 0 - Forward Only",
    1: "Stage 1 - Limited Yaw",
    2: "Stage 2 - Full Command",
}


def _camera(camera: Any, data: Any, *, terrain: str) -> None:
    """Follow the base while keeping the next staircase visible."""
    base = np.asarray(data.qpos[:3], dtype=float)
    lookahead = 0.55 if terrain in {"stairs", "mixed"} else 0.35
    camera.lookat[:] = (base[0] + lookahead, base[1], max(base[2] - 0.16, 0.12))
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -24


def _quat_rotate_inverse_numpy(quat: np.ndarray, vector: np.ndarray) -> np.ndarray:
    conjugate_xyz = -quat[1:]
    temporary = 2.0 * np.cross(conjugate_xyz, vector)
    return vector + quat[0] * temporary + np.cross(conjugate_xyz, temporary)


def _overlay_frame(
    frame: Any,
    *,
    data: Any,
    command: np.ndarray,
    stage: int,
    elapsed: float,
    transition: str | None,
) -> Any:
    """Draw the command/tracking contract directly into a rendered frame."""
    from PIL import ImageDraw, ImageFont

    image = frame.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        banner_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 26)
    except OSError:
        font = title_font = banner_font = ImageFont.load_default()

    local_velocity = _quat_rotate_inverse_numpy(
        np.asarray(data.qpos[3:7], dtype=float),
        np.asarray(data.qvel[:3], dtype=float),
    )
    # MODEL_FORWARD is [0, -1, 0].
    forward_velocity = -float(local_velocity[1])
    yaw_rate = float(data.qvel[5])
    lines = (
        STAGE_LABELS.get(stage, f"Stage {stage}"),
        f"t          {elapsed:6.2f} s",
        f"v_cmd / v  {command[0]:+6.3f} / {forward_velocity:+6.3f} m/s",
        f"yaw_cmd/yaw {command[1]:+6.3f} / {yaw_rate:+6.3f} rad/s",
    )
    panel_width = min(image.width - 20, 410)
    draw.rounded_rectangle(
        (10, 10, panel_width, 126), radius=8, fill=(0, 0, 0, 175)
    )
    draw.text((22, 18), lines[0], font=title_font, fill=(255, 255, 255, 255))
    for index, line in enumerate(lines[1:]):
        draw.text(
            (22, 48 + index * 23), line, font=font, fill=(235, 242, 248, 255)
        )

    if transition:
        left, top, right, bottom = draw.textbbox((0, 0), transition, font=banner_font)
        text_width = right - left
        text_height = bottom - top
        x = (image.width - text_width) // 2
        y = max(145, (image.height - text_height) // 2)
        draw.rounded_rectangle(
            (x - 20, y - 14, x + text_width + 20, y + text_height + 14),
            radius=10,
            fill=(17, 28, 42, 220),
            outline=(255, 190, 65, 255),
            width=3,
        )
        draw.text((x, y), transition, font=banner_font, fill=(255, 222, 128, 255))
    return image


def make_policy_evaluator(
    *,
    env: Any,
    make_policy: Callable[..., Any],
    duration: float,
    num_envs: int,
    seed: int,
) -> Callable[[Any], dict[str, float]]:
    """Build one compiled, batched deterministic stage evaluator."""
    if duration <= 0 or num_envs <= 0:
        raise ValueError("stage evaluation duration and num_envs must be positive")
    control_steps = max(1, int(np.ceil(duration / float(env.dt))))
    reset_batch = jax.vmap(env.reset)
    step_batch = jax.vmap(env.step)
    reset_keys = jax.random.split(jax.random.PRNGKey(seed), num_envs)

    @jax.jit
    def rollout(params: Any) -> tuple[jax.Array, ...]:
        policy = make_policy(params, deterministic=True)
        state = reset_batch(reset_keys)
        alive = jp.ones((num_envs,), dtype=jp.bool_)
        key = jax.random.PRNGKey(seed + 1)

        def body(carry, unused):
            del unused
            current, current_alive, action_key = carry
            action_key, policy_key = jax.random.split(action_key)
            action, _ = policy(current.obs, policy_key)
            next_state = step_batch(current, action)
            valid = current_alive
            reward = jp.sum(jp.where(valid, next_state.reward, 0.0))
            velocity_error = jp.sum(
                jp.where(valid, next_state.metrics["velocity_error_mps"], 0.0)
            )
            yaw_error = jp.sum(
                jp.where(valid, next_state.metrics["yaw_error_rps"], 0.0)
            )
            torque_squared = jp.sum(
                jp.where(valid, jp.square(next_state.metrics["torque_rms_nm"]), 0.0)
            )
            torque_saturation = jp.sum(
                jp.where(valid, next_state.metrics["torque_saturation"], 0.0)
            )
            self_collision = jp.sum(
                jp.where(valid, next_state.metrics["self_collision"], 0.0)
            )
            effective_stride = jp.sum(
                jp.where(valid, next_state.metrics["effective_stride_m"], 0.0)
            )
            max_effective_stride = jp.max(
                jp.where(valid, next_state.metrics["effective_stride_m"], 0.0)
            )
            step_scale = jp.sum(
                jp.where(valid, next_state.metrics["applied_step_scale"], 0.0)
            )
            frequency_scale = jp.sum(
                jp.where(
                    valid, next_state.metrics["applied_frequency_scale"], 0.0
                )
            )
            swing_height = jp.sum(
                jp.where(
                    valid, next_state.metrics["applied_swing_height_m"], 0.0
                )
            )
            radial_offset = jp.sum(
                jp.where(
                    valid, next_state.metrics["applied_radial_offset_m"], 0.0
                )
            )
            samples = jp.sum(valid.astype(jp.float32))
            next_alive = valid & (~next_state.done.astype(jp.bool_))
            return (next_state, next_alive, action_key), (
                reward,
                velocity_error,
                yaw_error,
                samples,
                torque_squared,
                torque_saturation,
                self_collision,
                effective_stride,
                max_effective_stride,
                step_scale,
                frequency_scale,
                swing_height,
                radial_offset,
            )

        (_, _, _), totals = jax.lax.scan(
            body, (state, alive, key), xs=None, length=control_steps
        )
        alive_count = jp.maximum(jp.sum(totals[3]), 1.0)
        scheduled_count = float(control_steps * num_envs)
        return (
            jp.sum(totals[0]) / scheduled_count,
            jp.sum(totals[1]) / alive_count,
            jp.sum(totals[2]) / alive_count,
            alive_count / scheduled_count,
            jp.sqrt(jp.sum(totals[4]) / alive_count),
            jp.sum(totals[5]) / alive_count,
            jp.sum(totals[6]) / alive_count,
            jp.sum(totals[7]) / alive_count,
            jp.max(totals[8]),
            jp.sum(totals[9]) / alive_count,
            jp.sum(totals[10]) / alive_count,
            jp.sum(totals[11]) / alive_count,
            jp.sum(totals[12]) / alive_count,
        )

    def evaluate(params: Any) -> dict[str, float]:
        (
            reward,
            velocity_error,
            yaw_error,
            survival,
            torque_rms,
            torque_saturation,
            self_collision_rate,
            effective_stride_mean,
            effective_stride_max,
            step_scale_mean,
            frequency_scale_mean,
            swing_height_mean,
            radial_offset_mean,
        ) = rollout(params)
        reward.block_until_ready()
        return {
            "reward_mean": float(reward),
            "velocity_error_mps": float(velocity_error),
            "yaw_error_rps": float(yaw_error),
            "survival_fraction": float(survival),
            "torque_rms_nm": float(torque_rms),
            "torque_saturation_mean": float(torque_saturation),
            "self_collision_rate": float(self_collision_rate),
            "effective_stride_mean_m": float(effective_stride_mean),
            "effective_stride_max_m": float(effective_stride_max),
            "gait_step_scale_mean": float(step_scale_mean),
            "gait_frequency_scale_mean": float(frequency_scale_mean),
            "gait_swing_height_mean_m": float(swing_height_mean),
            "gait_radial_offset_mean_m": float(radial_offset_mean),
        }

    return evaluate


def render_policy_video(
    *,
    env: Any,
    make_policy: Callable[..., Any],
    params: Any,
    output: Path,
    seed: int,
    duration: float,
    fps: int,
    width: int,
    height: int,
    terrain: str,
    overlay: bool = True,
) -> Path:
    """Render deterministic policy inference using the exact MJX task env.

    The trainer calls this only after a new evaluation best.  Dynamics and
    observations stay in MJX; only sparse frames are copied to ``MjData`` for
    MuJoCo's renderer.  GIF uses Pillow in-process, avoiding an ffmpeg fork
    after JAX has created worker threads.
    """
    if duration <= 0:
        raise ValueError("video duration must be greater than zero")
    if fps <= 0:
        raise ValueError("video fps must be greater than zero")
    if width <= 0 or height <= 0:
        raise ValueError("video width and height must be greater than zero")

    from PIL import Image
    import mujoco
    from mujoco import mjx

    output = output.resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("best-policy video output must end with .gif")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp{output.suffix}")
    temporary.unlink(missing_ok=True)

    policy = jax.jit(make_policy(params, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    control_steps = max(1, int(np.ceil(duration / float(env.dt))))
    # ``env.dt`` is 20 ms (50 Hz), so a 20 fps video cannot use one fixed
    # control-step interval.  Sample alternating 2/3-step gaps instead.  This
    # keeps both the requested frame count and the GIF's displayed duration
    # correct rather than silently turning 10 seconds into 12.5 seconds.
    frame_count = max(1, int(np.ceil(duration * fps)))
    frame_steps = np.floor(
        np.arange(frame_count, dtype=float) / (fps * float(env.dt))
    ).astype(int)
    frame_steps = np.clip(frame_steps, 0, control_steps - 1)
    next_frame = 0
    frames: list[Image.Image] = []
    previous_stage: int | None = None
    transition_text: str | None = None
    transition_until = -1.0

    try:
        key = jax.random.PRNGKey(seed + 1)
        for control_step in range(control_steps):
            displayed_command = np.asarray(state.info["command"], dtype=float)
            displayed_stage = int(np.asarray(state.info["curriculum_stage"]))
            elapsed = (control_step + 1) * float(env.dt)
            if previous_stage is not None and displayed_stage != previous_stage:
                transition_text = f"STAGE {previous_stage} -> STAGE {displayed_stage}"
                transition_until = elapsed + 0.8
            previous_stage = displayed_stage
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            state = step(state, action)
            if next_frame >= frame_count or control_step < frame_steps[next_frame]:
                continue
            host_data = mjx.get_data(env.mj_model, state.data)
            _camera(camera, host_data, terrain=terrain)
            renderer.update_scene(host_data, camera=camera)
            frame = Image.fromarray(renderer.render())
            if overlay:
                frame = _overlay_frame(
                    frame,
                    data=host_data,
                    command=displayed_command,
                    stage=displayed_stage,
                    elapsed=elapsed,
                    transition=(
                        transition_text if elapsed <= transition_until else None
                    ),
                )
            frames.append(frame.convert("P", palette=Image.ADAPTIVE))
            next_frame += 1
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("No frames were rendered for the best-policy video")
    frames[0].save(
        temporary,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=False,
    )
    temporary.replace(output)
    return output
