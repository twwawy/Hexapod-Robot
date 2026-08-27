#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/inspect_asset.sh"
"${SCRIPT_DIR}/verify_torch_contract.sh"
"${SCRIPT_DIR}/validate_configs.sh"

echo "HEXAPOD_ISAACLAB_STATIC_VERIFY_OK"
