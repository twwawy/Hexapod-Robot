from __future__ import annotations

"""Upload post-training visualization artifacts to the same W&B run.

The trainer owns scalar logging and checkpoint metadata. This helper reopens the
same run after MP4 rendering so the final video and run metadata land beside the
training curves instead of living only on disk.
"""

import argparse
import pickle
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload MJX run artifacts to an existing W&B run.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--latest-checkpoint-path", required=True)
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--run-metadata-path", required=True)
    parser.add_argument("--video-path", required=True)
    parser.add_argument("--stage-video-path", action="append", default=[], help="Upload a curriculum-stage best video; may be repeated.")
    parser.add_argument("--stage-checkpoint-path", action="append", default=[], help="Upload a curriculum-stage best checkpoint; may be repeated.")
    parser.add_argument("--neutral-pose-image-path", default=None)
    parser.add_argument("--neutral-pose-metadata-path", default=None)
    return parser.parse_args()



def _read_checkpoint_metadata(path: Path) -> dict:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError(f"Checkpoint metadata is not a dict: {path}")
    return metadata



def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).resolve()
    latest_checkpoint_path = Path(args.latest_checkpoint_path).resolve()
    metrics_path = Path(args.metrics_path).resolve()
    run_metadata_path = Path(args.run_metadata_path).resolve()
    video_path = Path(args.video_path).resolve()
    stage_video_paths = [Path(path).resolve() for path in args.stage_video_path]
    stage_checkpoint_paths = [Path(path).resolve() for path in args.stage_checkpoint_path]
    neutral_pose_image_path = Path(args.neutral_pose_image_path).resolve() if args.neutral_pose_image_path else None
    neutral_pose_metadata_path = Path(args.neutral_pose_metadata_path).resolve() if args.neutral_pose_metadata_path else None
    required_paths = [checkpoint_path, latest_checkpoint_path, metrics_path, run_metadata_path, video_path, *stage_video_paths, *stage_checkpoint_paths]
    missing_paths = [path for path in required_paths if not path.exists()]
    if missing_paths:
        missing = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"wandb upload aborted; required artifact(s) missing: {missing}")

    metadata = _read_checkpoint_metadata(checkpoint_path)
    wandb_meta = metadata.get("wandb")
    if not wandb_meta or not wandb_meta.get("enabled"):
        print("wandb_upload_skipped: checkpoint metadata does not contain an active wandb run")
        return

    try:
        import wandb  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is not installed in ~/.venvs/hexapod-mjx. Install it with `~/.venvs/hexapod-mjx/bin/python -m pip install wandb`."
        ) from exc

    run = wandb.init(
        project=wandb_meta.get("project"),
        entity=wandb_meta.get("entity"),
        group=wandb_meta.get("group"),
        job_type="mjx-render",
        mode=wandb_meta.get("mode", "online"),
        name=wandb_meta.get("name"),
        id=wandb_meta.get("run_id"),
        resume="allow",
    )

    artifact = wandb.Artifact(f"{run.name}-outputs", type="mjx-run")
    artifact_paths = [checkpoint_path, latest_checkpoint_path, metrics_path, run_metadata_path, video_path]
    artifact_paths.extend(stage_video_paths)
    artifact_paths.extend(stage_checkpoint_paths)
    if neutral_pose_image_path is not None:
        artifact_paths.append(neutral_pose_image_path)
    if neutral_pose_metadata_path is not None:
        artifact_paths.append(neutral_pose_metadata_path)
    for path in artifact_paths:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    run.summary["render_video"] = str(video_path)
    run.summary["stage_best_videos"] = [str(path) for path in stage_video_paths if path.exists()]
    run.summary["stage_best_checkpoints"] = [str(path) for path in stage_checkpoint_paths if path.exists()]
    run.summary["run_metadata_path"] = str(run_metadata_path)
    if neutral_pose_image_path is not None and neutral_pose_image_path.exists():
        run.summary["neutral_pose_image"] = str(neutral_pose_image_path)
    if neutral_pose_metadata_path is not None and neutral_pose_metadata_path.exists():
        run.summary["neutral_pose_metadata"] = str(neutral_pose_metadata_path)
    run.finish()
    print(f"wandb_upload_complete: run_id={wandb_meta.get('run_id')} video={video_path}")


if __name__ == "__main__":
    main()
