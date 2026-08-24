"""Render deterministic firmware-residual MJX rollouts as annotated GIFs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np


FOOT_LABELS = ("RF", "RM", "RB", "LF", "LM", "LB")


def _camera(camera: Any, data: Any) -> None:
    base = np.asarray(data.qpos[:3], dtype=float)
    camera.lookat[:] = (base[0] + 0.55, base[1], max(base[2] - 0.15, 0.12))
    camera.distance = 1.75
    camera.azimuth = 135
    camera.elevation = -24


def _overlay_frame(
    frame: Any,
    *,
    state: Any,
    host_data: Any,
    elapsed: float,
    terrain_level: int,
    step_height: float,
    title: str | None,
) -> Any:
    from PIL import ImageDraw, ImageFont

    image = frame.convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 16)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 19)
    except OSError:
        font = title_font = ImageFont.load_default()

    command = np.asarray(state.info["command"], dtype=float)
    contacts = np.asarray(state.info["contact_state"], dtype=bool)
    controller = state.info["controller_output"]
    gait_progress = np.asarray(controller.gait_progress, dtype=float)
    gait_state = np.asarray(controller.gait_state, dtype=int)
    termination_names = (
        "controller_invalid",
        "joint_limit",
        "dynamics",
        "tilt",
        "clearance",
        "body_contact",
        "nonfinite",
    )
    active_termination = [
        name
        for name in termination_names
        if float(np.asarray(state.metrics[f"termination/{name}"])) > 0.5
    ]
    contact_text = " ".join(
        f"{label}{int(active)}" for label, active in zip(FOOT_LABELS, contacts)
    )
    gait_items = tuple(
        f"{label}:{phase:.2f}/{leg_state}"
        for label, phase, leg_state in zip(FOOT_LABELS, gait_progress, gait_state)
    )
    lines = (
        title or f"Firmware residual PPO | level {terrain_level}",
        f"t={elapsed:6.2f}s  x={float(host_data.qpos[0]):+6.3f}m  z={float(host_data.qpos[2]):+6.3f}m",
        f"command speed={command[0]:.3f}m/s yaw={command[1]:+.3f}rad/s",
        f"stairs level={terrain_level} riser={100.0 * step_height:.1f}cm",
        f"joint margin={float(np.asarray(state.metrics['joint_limit_margin_rad'])):.3f}rad",
        f"contacts {contact_text}",
        "gait R " + " ".join(gait_items[:3]),
        "gait L " + " ".join(gait_items[3:]),
        "termination " + (", ".join(active_termination) if active_termination else "none"),
    )
    panel_width = min(image.width - 20, 760)
    panel_bottom = 42 + 22 * len(lines)
    draw.rounded_rectangle(
        (10, 10, panel_width, panel_bottom), radius=8, fill=(0, 0, 0, 180)
    )
    draw.text((22, 18), lines[0], font=title_font, fill=(255, 255, 255, 255))
    for index, line in enumerate(lines[1:]):
        color = (255, 150, 130, 255) if index == len(lines) - 2 and active_termination else (235, 242, 248, 255)
        draw.text((22, 46 + index * 22), line, font=font, fill=color)
    return image


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
    overlay_title: str | None = None,
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
                frame = _overlay_frame(
                    frame,
                    state=state,
                    host_data=host_data,
                    elapsed=(control_step + 1) * float(env.dt),
                    terrain_level=env.curriculum_level,
                    step_height=env.terrain_step_height,
                    title=overlay_title,
                )
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
