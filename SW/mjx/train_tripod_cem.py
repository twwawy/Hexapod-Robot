from __future__ import annotations

"""CLI entrypoint for the first MJX gait-search experiment.

This file is intentionally thin. The heavy lifting lives in:
- ``hexapod_mjx.model`` for URDF -> simplified MJCF -> MJX conversion,
- ``hexapod_mjx.cem`` for batched rollout and Cross-Entropy Method search.

Keeping this file small makes it easier to treat as the reproducible shell/API
boundary: flags in, result JSON out.
"""

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from hexapod_mjx.cem import CEMConfig, INITIAL_STD, PARAMETER_NAMES, run_cem_search
from hexapod_mjx.model import load_hexapod_model, repo_root_from


def parse_args() -> argparse.Namespace:
    """Parse CLI flags.

    The flags map almost 1:1 to ``CEMConfig`` so that shell history is enough
    to reconstruct an experiment later.
    """
    parser = argparse.ArgumentParser(
        description="Search a basic tripod gait for the Hexapod using MuJoCo MJX batched CEM."
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=None,
        help="Path inside or at the Hexapod-Robot repo. Defaults to this script location.",
    )
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--elite-count", type=int, default=32)
    parser.add_argument("--num-iterations", type=int, default=20)
    parser.add_argument("--rollout-steps", type=int, default=600)
    parser.add_argument("--action-repeat", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--resume-result-path",
        type=str,
        default=None,
        help="Optional previous result JSON. If set, restart CEM around that run's best_params.",
    )
    parser.add_argument(
        "--resume-std-scale",
        type=float,
        default=0.5,
        help="Scale factor applied to the default search std when resuming from a prior best.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="SW/mjx/artifacts/hexapod_cem_result.json",
        help="Repo-relative output JSON path.",
    )
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, value: str) -> Path:
    """Resolve either repo-relative or absolute paths against the repo root."""
    path = Path(value)
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _load_resume_best_params(result_path: Path) -> jnp.ndarray:
    """Load ``best_params`` from a prior run and convert them into vector order."""
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    best_params = payload.get("best_params", {})
    missing = [name for name in PARAMETER_NAMES if name not in best_params]
    if missing:
        raise KeyError(f"Resume result JSON is missing best_params entries: {missing}")
    return jnp.asarray([float(best_params[name]) for name in PARAMETER_NAMES], dtype=jnp.float32)


def main() -> None:
    """Resolve the repo, optionally resume from a prior best, then run CEM."""
    args = parse_args()
    if args.resume_std_scale <= 0.0:
        raise ValueError("resume_std_scale must be > 0")

    # When this file is executed directly, ``__file__`` already lives under
    # ``<repo>/SW/mjx``. Walking up two parents gives us a robust default root
    # without forcing the user to ``cd`` first.
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)

    # This one call performs all one-time model preparation needed for training:
    # cleaned URDF -> simplified floating-base MJCF -> compiled MuJoCo/MJX model.
    bundle = load_hexapod_model(repo_root)

    resume_result_path: Path | None = None
    initial_mean = None
    initial_std = None
    if args.resume_result_path:
        resume_result_path = _resolve_repo_path(repo_root, args.resume_result_path)
        initial_mean = _load_resume_best_params(resume_result_path)
        initial_std = INITIAL_STD * args.resume_std_scale

    # Keep the CLI arguments mirrored explicitly into the config object. That is
    # more verbose than splatting ``vars(args)``, but much easier to audit when
    # fields are added, renamed, or intentionally kept out of the public CLI.
    result = run_cem_search(
        bundle,
        CEMConfig(
            population_size=args.population_size,
            elite_count=args.elite_count,
            num_iterations=args.num_iterations,
            rollout_steps=args.rollout_steps,
            action_repeat=args.action_repeat,
            seed=args.seed,
            output_path=args.output_path,
            resume_result_path=str(resume_result_path) if resume_result_path is not None else None,
            resume_std_scale=args.resume_std_scale,
        ),
        initial_mean=initial_mean,
        initial_std=initial_std,
    )

    print("MJX tripod gait search complete")
    print(f"repo_root: {repo_root}")
    if resume_result_path is not None:
        print(f"resume_result_path: {resume_result_path}")
        print(f"resume_std_scale: {args.resume_std_scale:.3f}")
    print(f"generated_mjcf: {bundle.generated_mjcf_path}")
    print(f"best_score: {result.best_score:.4f}")
    print(f"output_path: {result.output_path}")
    print("best_params:")
    for key, value in result.best_params.items():
        print(f"  - {key}: {value:.6f}")


if __name__ == "__main__":
    main()