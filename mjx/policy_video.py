"""Render clean deterministic firmware-residual MJX rollout GIFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np


def _camera(camera: Any, data: Any) -> None:
    base = np.asarray(data.qpos[:3], dtype=float)
    camera.lookat[:] = (base[0] + 0.55, base[1], max(base[2] - 0.15, 0.12))
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -24


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
) -> Path:
    """Render one deterministic rollout using the exact training environment."""
    if duration <= 0 or fps <= 0 or width <= 0 or height <= 0:
        raise ValueError("duration, fps, width and height must all be positive")
    output = output.expanduser().resolve()
    if output.suffix.lower() != ".gif":
        raise ValueError("policy video output must end with .gif")

    from PIL import Image
    import mujoco
    from mujoco import mjx

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.gif")
    temporary.unlink(missing_ok=True)
    policy = jax.jit(make_policy(params, deterministic=True))
    reset = jax.jit(env.reset)
    step = jax.jit(env.step)
    state = reset(jax.random.PRNGKey(seed))
    state.obs.block_until_ready()

    control_steps = max(1, int(np.ceil(duration / float(env.dt))))
    frame_count = max(1, int(np.ceil(duration * fps)))
    frame_steps = np.floor(
        np.arange(frame_count, dtype=float) / (fps * float(env.dt))
    ).astype(int)
    frame_steps = np.clip(frame_steps, 0, control_steps - 1)
    renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    frames: list[Image.Image] = []
    next_frame = 0
    key = jax.random.PRNGKey(seed + 1)

    try:
        for control_step in range(control_steps):
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            state = step(state, action)
            should_render = (
                next_frame < frame_count and control_step >= frame_steps[next_frame]
            )
            done = bool(np.asarray(state.done))
            # Always preserve the unsafe/success terminal state even when it
            # falls between the requested FPS sample instants.
            if should_render or done:
                host_data = mjx.get_data(env.mj_model, state.data)
                _camera(camera, host_data)
                renderer.update_scene(host_data, camera=camera)
                frame = Image.fromarray(renderer.render())
                frames.append(frame.convert("P", palette=Image.ADAPTIVE))
                next_frame += 1
            if done:
                break
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        renderer.close()

    if not frames:
        raise RuntimeError("no video frames were rendered")
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
