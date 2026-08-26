#!/usr/bin/env python3
"""Download the selected MJX reference run metadata/history from W&B."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import wandb


DEFAULT_RUN = "hurolilys-inha-university/hexapod-firmware-terrain/g568d0hq"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    raw_dict = getattr(value, "_dict", None)
    if isinstance(raw_dict, dict):
        return _jsonable(raw_dict)
    tolist = getattr(type(value), "tolist", None)
    if tolist is not None:
        return _jsonable(tolist(value))
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default=DEFAULT_RUN)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/wandb/reference_run.json",
    )
    args = parser.parse_args()

    api = wandb.Api()
    run = api.run(args.run)
    history = run.history(samples=args.samples, pandas=False)
    payload = {
        "source": args.run,
        "url": run.url,
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "config": {key: _jsonable(value) for key, value in run.config.items()},
        "summary": {key: _jsonable(value) for key, value in run.summary.items()},
        "history_sampled": [
            {key: _jsonable(value) for key, value in row.items()} for row in history
        ],
        "checkpoint_note": (
            "W&B stores only the local best/checkpoint path for this project; "
            "no model/checkpoint artifact collection was logged."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
