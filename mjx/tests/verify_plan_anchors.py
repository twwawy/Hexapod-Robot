#!/usr/bin/env python3
"""Verify the source anchors required by the stair/posture execution plan."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CANONICAL_BASE = "990e82607267cd38f18608a6c53840d2e9beb2f3"

ANCHORS: dict[str, tuple[str, ...]] = {
    "mjx/firmware_mjx_controller.py": (
        r"FIRMWARE_CONTROL_DT\s*=\s*0\.005",
        r"SWING_HEIGHT\s*=\s*0\.06",
        r"SWING_HEIGHT_MIN\s*=\s*0\.04",
        r"SWING_HEIGHT_MAX\s*=\s*0\.25",
        r"JOINT_RATE\s*=\s*jp\.deg2rad\(315\.8\)",
        r"RESIDUAL_SCALE\s*=\s*jp\.array\(\(0\.10,\s*0\.10,\s*0\.10\)\)",
        r"class FirmwareState\(NamedTuple\)",
        r"class FirmwareOutput\(NamedTuple\)",
        r"def _rotation_matrix\(",
        r"def _rotate_inverse\(",
        r"def _all_feet_valid\(",
        r"def _adaptive_swing_height\(",
        r"def _phase_gated_policy_residual\(",
        r"swing_envelope\s*=\s*4\.0\s*\*\s*scaled_progress",
        r"def step\(",
        r"twist_step\s*=\s*jp\.array",
        r"roll_cmd - attitude_rpy\[0\],\s*effective_pitch - attitude_rpy\[1\]",
        r"jp\.deg2rad\(15\.0\)",
        r"jp\.deg2rad\(45\.0\)",
        r"posture_accepted\s*=\s*_all_feet_valid",
        r"residual_alpha\s*=\s*jp\.exp\(-FIRMWARE_CONTROL_DT\s*/\s*0\.10\)",
        r"candidate_local\s*=\s*_body_to_leg",
        r"residual_feet\s*=\s*_rotate_inverse",
        r"safe_feet\s*=\s*jp\.where\(policy_valid",
        r"limited_feet,\s*foot_limited\s*=\s*_limit_foot_reach",
        r"joint_step\s*=\s*JOINT_RATE\s*\*\s*FIRMWARE_CONTROL_DT",
    ),
    "mjx/rough_terrain_env.py": (
        r"OBSERVATION_SIZE\s*=\s*146",
        r"OBSERVATION_CONTRACT_VERSION\s*=\s*\"firmware_state_collision_terrain_command5_pitch_v3\"",
        r"max_tilt=0\.7853981633974483",
        r"min_clearance=0\.14",
        r"joint_limit_margin=0\.017453292519943295",
        r"velocity=2\.5",
        r"upright=1\.0",
        r"ascent_bonus=8\.0",
        r"success_bonus=30\.0",
        r"failure_penalty=-30\.0",
        r"self\._step_centers\s*=",
        r"self\._height_samples\s*=",
        r"if spec\.kind == \"stairs\"",
        r"def _relative_attitude\(",
        r"def _terrain_height\(",
        r"def _body_contact\(",
        r"def _terrain_features\(",
        r"def _get_obs\(",
        r"attitude_rpy=attitude_before",
        r"tilt_failure\s*=",
        r"clearance_failure\s*=",
        r"joint_limit_failure\s*=",
        r"& _posture_success\(attitude, posture_target\)",
        r"torque_limit\s*=\s*SERVO_STALL_TORQUE_NM",
        r"torque_saturation\s*=",
        r"reward_terms\s*=\s*_base_reward_terms\(",
        r"\"upright\": motion_gate \* _upright_reward\(attitude, posture_target\)",
        r"scaled\s*=\s*_scale_reward_terms\(",
    ),
    "mjx/train_rough_terrain.py": (
        r"class ScoreMonitor",
        r"parser\.add_argument\(\"--timesteps\",\s*type=int,\s*default=50_000_000\)",
        r"parser\.add_argument\(\"--num-envs\",\s*type=int,\s*default=2048\)",
        r"parser\.add_argument\(\"--score-key\",\s*default=\"eval/episode_reward\"\)",
        r"def _infer_stage\(",
        r"def _resolve_checkpoint\(",
        r"contract\.get\(\"action_size\"\)\s*!=\s*ACTION_SIZE",
        r"observation_shape\s*!=\s*\[OBSERVATION_SIZE\]",
        r"metadata\.get\(\"action_contract_version\"\)",
        r"spec\.kind == \"stairs\" and spec\.stair_count == 10",
        r"config\.collision_mode\s*=\s*args\.collision_mode",
        r"best_checkpoint\.json",
        r"functools\.partial\(\s*ppo\.train,",
    ),
    "mjx/servo_model.py": (
        r"SERVO_POSITION_KP\s*=\s*500\.0",
        r"SERVO_POSITION_KV\s*=\s*10\.0",
        r"SERVO_OUTPUT_ARMATURE_KGM2\s*=\s*0\.02",
        r"SERVO_OUTPUT_DAMPING_NMS_RAD\s*=\s*0\.15",
        r"SERVO_GEAR_FRICTION_NM\s*=\s*0\.8",
        r"SERVO_SATURATION_START_FRACTION\s*=\s*0\.85",
    ),
    "mjx/terrain_curriculum.py": (
        r"STAIR_DEPTH\s*=\s*0\.25",
        r"MAX_STAIR_COUNT\s*=\s*10",
        r"TERRAIN_LEVELS\s*=",
        r"_stairs\(5,\s*7,\s*0\.05\)",
        r"_stairs\(10,\s*7,\s*0\.20\)",
        r"_stairs\(11,\s*10,\s*0\.05\)",
        r"_stairs\(16,\s*10,\s*0\.20\)",
    ),
}


def main() -> int:
    failures: list[str] = []
    print(f"CANONICAL_BASE:{CANONICAL_BASE}")
    for relative_path, patterns in ANCHORS.items():
        path = ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.MULTILINE)
            if match is None:
                failures.append(f"FAIL:{relative_path}:{pattern}")
                continue
            line = text.count("\n", 0, match.start()) + 1
            print(f"CONFIRMED:{relative_path}:{line}:{pattern}")
    for failure in failures:
        print(failure)
    print(f"SUMMARY:confirmed={sum(len(items) for items in ANCHORS.values()) - len(failures)} failures={len(failures)}")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
