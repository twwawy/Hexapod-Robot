#!/usr/bin/env python3
"""Unified commands. Help, history and dry runs require only Python and Git."""
import argparse
import ast
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/integration-sources.json"
RL_TESTS = (
    "test_adaptive_foot_retry", "test_adaptive_grid", "test_adaptive_hybrid",
    "test_adaptive_v4", "test_operator_commands", "test_path_bottleneck",
    "test_path_migration",
)


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def sources():
    return json.loads(MANIFEST.read_text())["sources"]


def verify_history():
    if git("rev-parse", "--is-shallow-repository") == "true":
        raise ValueError("Full history required. Run: git fetch --unshallow origin")
    for source in sources():
        subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                        source["sha"], "HEAD"], check=True)
        print(f'PRESERVED {source["id"]}: {source["sha"]}', flush=True)


def run(command, *, dry_run=False, cwd=None, env=None):
    print("+ " + shlex.join(map(str, command)), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=cwd, env=env, check=True)


def check(rl=False):
    # Parse tracked Python, including optional runtimes, without importing them.
    for name in git("ls-files", "*.py").splitlines():
        path = ROOT / name
        if path.is_file():
            ast.parse(path.read_text(encoding="utf-8"), filename=name)
    for name in git("ls-files", "*.sh", "hexapod").splitlines():
        run(["bash", "-n", str(ROOT / name)])
    verify_history()
    groups = [
        ("launcher", [sys.executable, "-m", "unittest", "discover", "-s",
                      str(ROOT / "scripts/tests"), "-v"], {}),
        ("STM32 host", [sys.executable, str(ROOT / "scripts/test_stm32_host.py")], {}),
    ]
    if rl:
        groups.append(("adaptive RL", [sys.executable, "-m", "unittest", "-v", *RL_TESTS],
                       {"cwd": ROOT / "mjx/tests", "env": {**os.environ, "JAX_PLATFORMS": "cpu"}}))
    failed = []
    for name, command, options in groups:
        try:
            run(command, **options)
        except subprocess.CalledProcessError:
            failed.append(name)
    if failed:
        raise ValueError("Checks failed: " + ", ".join(failed) + ". See docs/INTEGRATION_VALIDATION.md.")


def bundle(destination):
    verify_history()
    if git("status", "--porcelain", "--untracked-files=normal"):
        raise ValueError("Commit source changes first; bundles contain committed Git history only.")
    destination = Path(destination).expanduser().absolute()
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # HEAD reaches all pinned sources, including history-only legacy merges.
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        output = Path(temp) / "source.bundle"
        run(["git", "-C", str(ROOT), "bundle", "create", str(output), "HEAD"])
        run(["git", "-C", str(ROOT), "bundle", "verify", str(output)])
        # Hard-link is atomic and refuses to overwrite a concurrent destination.
        os.link(output, destination)
    print(f"Saved source + full development history: {destination}")
    print("Not included: ignored local checkpoints, runs, environments or uncommitted files.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print a launch command without executing it")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("doctor", help="Show interpreter, dependencies and hardware transport defaults")
    checks = sub.add_parser("check", help="Source, history, launcher and native STM32 checks")
    checks.add_argument("--rl", action="store_true", help="Also run CPU adaptive contract tests (requires RL dependencies)")
    history = sub.add_parser("history", help="List or restore a pinned original branch in a separate worktree")
    history.add_argument("source", nargs="?", choices=[x["id"] for x in sources()])
    history.add_argument("destination", nargs="?", type=Path)
    sub.add_parser("verify-history", help="Verify every original tip is reachable from HEAD")
    pack = sub.add_parser("bundle", help="Collect this branch and all ancestral sources into one Git bundle")
    pack.add_argument("destination", nargs="?", default=str(ROOT / ".artifacts/hexapod-source.bundle"))
    # parse_known_args preserves existing subcommand --help and all flags verbatim.
    for name, help_text in (
        ("train", "Current hybrid terrain curriculum; arguments pass through"),
        ("train-rc", "Flat RC command curriculum; arguments pass through"),
        ("train-stage", "Single-stage adaptive PPO"),
        ("view", "Adaptive foothold viewer"),
        ("replay", "Legacy 18-D trained-policy viewer"),
        ("resume-v5", "Explicit reviewed v5 to current v6 warm start"),
        ("resume-v6", "Explicit reviewed v6 clearance warm start"),
        ("build", "Build STM32 with ARM GCC (never flash)"),
    ):
        sub.add_parser(name, add_help=False, help=help_text)
    args, extra = parser.parse_known_args(argv)
    launches = {
        "train": ("train_adaptive_curriculum.sh", ["--profile", "hybrid", "--command-mode", "terrain", "--perception", "teacher"]),
        "train-rc": ("train_adaptive_curriculum.sh", ["--profile", "rc", "--command-mode", "rc", "--perception", "teacher"]),
        "train-stage": ("train_adaptive_gait.sh", []),
        "view": ("view_foothold_planner.sh", ["--controller", "adaptive"]),
        "replay": ("view_trained_policy.sh", []),
        "resume-v5": ("resume_stair5_path_v6.sh", []),
        "resume-v6": ("resume_stair5_v6_clearance.sh", []),
    }
    if args.command in launches:
        script, defaults = launches[args.command]
        run(["bash", str(ROOT / "scripts" / script), *defaults, *extra], dry_run=args.dry_run,
            env={**os.environ, "HEXAPOD_PYTHON": sys.executable})
    elif args.command == "build":
        run([sys.executable, str(ROOT / "scripts/build_stm32.py"), *extra], dry_run=args.dry_run)
    else:
        if extra:
            parser.error("unrecognized arguments: " + " ".join(extra))
        if args.dry_run:
            parser.error("--dry-run is supported only for launch commands (train/view/replay/resume/build)")
        if args.command == "doctor":
            print(f"Repository: {ROOT}\nPython: {sys.executable}\nRevision: {git('rev-parse', 'HEAD')}")
            for module in ("numpy", "jax", "mujoco", "mujoco_playground", "brax", "onnxruntime", "wandb"):
                print(f"{module}: {'available' if importlib.util.find_spec(module) else 'missing'}")
            print(f"ARM GCC: {shutil.which('arm-none-eabi-gcc') or 'not found'}")
            print("STM32 default: 64-byte sensor/GPS. Adaptive 128-byte transport requires explicit firmware selection.")
        elif args.command == "check":
            check(args.rl)
        elif args.command == "verify-history":
            verify_history()
        elif args.command == "history":
            if not args.source:
                for source in sources():
                    print(f'{source["id"]:12} {source["sha"]}  {source["branch"]}\n  {source["disposition"]}')
            else:
                if args.destination is None:
                    parser.error("history SOURCE requires a destination for the isolated worktree")
                destination = args.destination.expanduser().absolute()
                if destination.exists():
                    raise ValueError(f"Destination already exists: {destination}")
                verify_history()
                source = next(x for x in sources() if x["id"] == args.source)
                run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(destination), source["sha"]])
        elif args.command == "bundle":
            bundle(args.destination)
        else:
            parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f"hexapod: {error}", file=sys.stderr)
        sys.exit(1)
