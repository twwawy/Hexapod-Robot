from __future__ import annotations

import argparse
from pathlib import Path

from hexapod_mjx.model import contact_override_path, export_contact_override_template, repo_root_from


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the current inferred MJX foot contact points into an editable JSON override file.")
    parser.add_argument("--repo-root", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None, help="Override the output path. Default: SW/mjx/contact_points_override.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_root = Path(__file__).resolve().parents[2]
    repo_root = repo_root_from(args.repo_root or default_root)
    target = contact_override_path(repo_root) if args.output_path is None else args.output_path
    output_path = export_contact_override_template(repo_root, target)
    print(f"contact_override_template: {output_path}")


if __name__ == "__main__":
    main()
