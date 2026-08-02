#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$HOME/Desktop/Hexapod-MJX-가이드"
ARGS=(--skip-train)

if (($#)); then
  case "$1" in
    latest|최근|best)
      shift
      ;;
    -*)
      ;;
    *)
      ARGS+=(--policy-path "$1")
      shift
      ;;
  esac
fi

exec "$SCRIPT_DIR/residual_rl_run.sh" \
  "${ARGS[@]}" \
  "$@"
