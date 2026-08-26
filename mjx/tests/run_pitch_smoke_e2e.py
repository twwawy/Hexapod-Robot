#!/usr/bin/env python3
"""Run and assert the four bounded command-adaptive CPU PPO scenarios."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[2]
TRAINER = ROOT / "mjx" / "train_rough_terrain.py"
REQUIRED_METRICS = (
    "eval/episode_reward/upright",
    "eval/episode_reward/height",
    "eval/episode_reward/foot_clearance_terrain",
    "eval/episode_reward/edge_margin",
    "eval/episode_reward/touchdown_impact",
)


def _run_logged(command: list[str], log_path: Path) -> tuple[int, str]:
    environment = os.environ.copy()
    environment["JAX_PLATFORMS"] = "cpu"
    environment["PYTHONUNBUFFERED"] = "1"
    lines: list[str] = []
    with log_path.open("w", encoding="utf-8") as log:
        started = datetime.now(timezone.utc).isoformat()
        marker = f"STARTED_AT_UTC={started}\nCOMMAND={' '.join(command)}\n"
        print(marker, end="", flush=True)
        log.write(marker)
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            lines.append(line)
        returncode = process.wait()
        finished = datetime.now(timezone.utc).isoformat()
        marker = f"FINISHED_AT_UTC={finished}\nEXIT_CODE={returncode}\n"
        print(marker, end="", flush=True)
        log.write(marker)
        return returncode, "".join(lines)


def _run_directory(stdout: str) -> Path:
    matches = re.findall(r"^RUN_DIR=(.+)$", stdout, flags=re.MULTILINE)
    if len(matches) != 1:
        raise AssertionError(f"expected one RUN_DIR line, found {len(matches)}")
    return Path(matches[0]).resolve()


def _assert_success(stdout: str) -> None:
    run_dir = _run_directory(stdout)
    latest = run_dir / "monitor" / "latest_metrics.json"
    best = run_dir / "monitor" / "best_checkpoint.json"
    if not latest.exists():
        raise AssertionError(f"missing monitor JSON: {latest}")
    if not best.exists():
        raise AssertionError(f"missing best checkpoint pointer: {best}")
    metrics = json.loads(latest.read_text(encoding="utf-8"))["metrics"]
    missing = [key for key in REQUIRED_METRICS if key not in metrics]
    if missing:
        raise AssertionError(f"missing reward metrics: {missing}")
    tail = "\n".join(stdout.splitlines()[-100:])
    if re.search(r"\bnan\b", tail, flags=re.IGNORECASE):
        raise AssertionError("stdout tail contains NaN")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, choices=(32, 64), default=64)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        default=(
            ROOT
            / ".omo"
            / "evidence"
            / "stair-climb-posture-servo"
            / "task-9"
        ),
    )
    args = parser.parse_args()
    args.evidence_dir.mkdir(parents=True, exist_ok=True)
    run_root = args.evidence_dir / "runs"

    for level in (0, 9):
        for bank_size in (1, 16):
            token = f"l{level}-bank{bank_size}"
            command = [
                sys.executable,
                str(TRAINER),
                "--terrain-level",
                str(level),
                "--dr-bank-size",
                str(bank_size),
                "--timesteps",
                "65536",
                "--num-evals",
                "2",
                "--num-envs",
                str(args.num_envs),
                # Keep the required 65,536 sample budget while bounding the
                # CPU-only PPO compile graph used by this integration smoke.
                "--batch-size",
                "32",
                "--num-minibatches",
                "1",
                "--num-updates-per-batch",
                "1",
                "--network-layers",
                "32",
                "32",
                "--run-name",
                f"smoke-cmd-{token}",
                "--run-root",
                str(run_root),
                "--allow-cpu",
                "--no-best-video",
                "--no-stage-video",
                "--no-progress-video",
            ]
            returncode, stdout = _run_logged(
                command, args.evidence_dir / f"{token}.log"
            )
            if returncode != 0:
                raise AssertionError(f"{token} exited with {returncode}")
            _assert_success(stdout)
            print(f"ASSERTED:{token}", flush=True)

    failure_command = [
        sys.executable,
        str(TRAINER),
        "--dr-bank-size",
        "999",
    ]
    returncode, stdout = _run_logged(
        failure_command, args.evidence_dir / "invalid-bank.log"
    )
    if returncode == 0 or "invalid choice: 999" not in stdout:
        raise AssertionError("invalid --dr-bank-size did not fail fast")
    print("ASSERTED:invalid-bank", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
