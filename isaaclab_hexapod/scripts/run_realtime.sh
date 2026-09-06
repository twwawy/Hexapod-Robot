#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# One full-CAD robot and one terrain scene.  Keep this GUI path independent
# from the batched RSL-RL training launcher.
exec "${REPO_ROOT}/isaaclab_hexapod/scripts/launch_latest_scene.sh" \
  --device cuda:0 "$@"
