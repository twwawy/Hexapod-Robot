"""Render a deterministic MJX residual-policy rollout to an animated GIF."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np


def _camera(camera: Any, data: Any, *, terrain: str) -> None:
    """Follow the base while keeping the next staircase visible."""
    base = np.asarray(data.qpos[:3], dtype=float)
    lookahead = 0.55 if terrain == "stairs" else 0.35
    camera.lookat[:] = (base[0] + lookahead, base[1], max(base[2] - 0.16, 0.12))
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
    terrain: str,
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

    try:
        key = jax.random.PRNGKey(seed + 1)
        for control_step in range(control_steps):
            key, action_key = jax.random.split(key)
            action, _ = policy(state.obs, action_key)
            state = step(state, action)
            if next_frame >= frame_count or control_step < frame_steps[next_frame]:
                continue
            host_data = mjx.get_data(env.mj_model, state.data)
            _camera(camera, host_data, terrain=terrain)
            renderer.update_scene(host_data, camera=camera)
            frames.append(
                Image.fromarray(renderer.render()).convert(
                    "P", palette=Image.ADAPTIVE
                )
            )
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
