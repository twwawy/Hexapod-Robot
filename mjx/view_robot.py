#!/usr/bin/env python3
"""Load and display the hexapod with MuJoCo."""

from __future__ import annotations

import argparse
from pathlib import Path

from prepare_scene import SCENE_OUTPUT, prepare_scene
from prepare_urdf import SOURCE_URDF


def ensure_model() -> Path:
    scene_sources = (
        SOURCE_URDF,
        Path(__file__).resolve().parent / "prepare_urdf.py",
        Path(__file__).resolve().parent / "prepare_scene.py",
    )
    if not SCENE_OUTPUT.exists() or any(
        SCENE_OUTPUT.stat().st_mtime < source.stat().st_mtime
        for source in scene_sources
    ):
        prepare_scene()
    return SCENE_OUTPUT


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="render one PNG instead of opening the interactive viewer",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "hexapod.png",
        help="output path used with --headless",
    )
    args = parser.parse_args()

    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Run: python3 -m pip install -r mjx/requirements.txt"
        ) from exc

    model_path = ensure_model()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)

    print(
        f"Loaded {model_path.name}: {model.nbody} bodies, "
        f"{model.njnt} joints, {model.ngeom} geoms"
    )

    if args.headless:
        # Standalone URDF models use MuJoCo's default 640x480 offscreen buffer.
        renderer = mujoco.Renderer(model, height=480, width=640)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.lookat[:] = (0.0, 0.0, 0.0)
        camera.distance = 1.45
        camera.azimuth = 135
        camera.elevation = -25
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()

        from PIL import Image

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels).save(args.output)
        print(f"Saved render to {args.output.resolve()}")
        return

    import mujoco.viewer

    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
