"""Validated access to the current MJX-to-Isaac training handoff."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

from .observation_contract import LEGACY_SIZE


REPO_ROOT = Path(__file__).resolve().parents[5]
TRAINING_HANDOFF_PATH = (
    REPO_ROOT / "isaaclab_hexapod/data/training/latest_mjx_training.json"
)
EXPECTED_ACTION_CONTRACT = "stm32_firmware_adaptive_swing_residual_100mm_v4"
EXPECTED_OBSERVATION_CONTRACT = (
    "firmware_state_collision_terrain_command5_pitch_v3"
)
EXPECTED_REWARD_CONTRACT = "commanded_progress_motion_gate_v1"


@lru_cache(maxsize=1)
def load_training_handoff() -> dict[str, Any]:
    handoff = json.loads(TRAINING_HANDOFF_PATH.read_text(encoding="utf-8"))
    contracts = handoff["contracts"]
    checks = {
        "action size": contracts["action"]["size"] == 18,
        "action version": (
            contracts["action"]["version"] == EXPECTED_ACTION_CONTRACT
        ),
        "observation size": contracts["observation"]["size"] == LEGACY_SIZE,
        "observation version": (
            contracts["observation"]["version"]
            == EXPECTED_OBSERVATION_CONTRACT
        ),
        "reward version": (
            contracts["reward"]["version"] == EXPECTED_REWARD_CONTRACT
        ),
        "physics timing": contracts["timing"]["physics_dt_s"] == 0.0025,
        "policy timing": contracts["timing"]["policy_dt_s"] == 0.02,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise RuntimeError(f"stale or incompatible MJX training handoff: {failed}")
    return handoff
