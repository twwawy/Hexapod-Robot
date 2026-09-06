"""DS51150-270 actuator parameters shared by scene and training code.

The datasheet fixes voltage, no-load speed, gear ratio and stall torque.  The
manufacturer does not publish closed-loop stiffness or output-side friction,
so those values are explicit calibration priors instead of hidden MuJoCo
defaults.  Replace the priors after measuring a powered joint on the robot.
"""

from __future__ import annotations


KGF_CM_TO_NM = 0.0980665

SERVO_MODEL_NAME = "DS51150-270"
SERVO_SUPPLY_VOLTAGE_V = 12.6
SERVO_GEAR_RATIO = 357.0
SERVO_STALL_TORQUE_KGF_CM = 150.0
SERVO_STALL_TORQUE_NM = SERVO_STALL_TORQUE_KGF_CM * KGF_CM_TO_NM
SERVO_NO_LOAD_SPEED_DEG_S = 60.0 / 0.19

# Output-side digital position-loop approximation.  The previous kp=120,
# kv=3 model was deliberately soft and became freely backdrivable at 8 Nm.
# These priors keep the powered geared joint stiff while retaining the real
# finite stall torque.  They are intentionally centralized for bench tuning.
SERVO_POSITION_KP = 500.0
SERVO_POSITION_KV = 10.0
SERVO_OUTPUT_ARMATURE_KGM2 = 0.02
SERVO_OUTPUT_DAMPING_NMS_RAD = 0.15
SERVO_GEAR_FRICTION_NM = 0.8

SERVO_SATURATION_START_FRACTION = 0.85


def metadata() -> dict[str, float | str]:
    """Return the exact actuator contract stored with every training run."""
    return {
        "name": SERVO_MODEL_NAME,
        "supply_voltage_v": SERVO_SUPPLY_VOLTAGE_V,
        "gear_ratio": SERVO_GEAR_RATIO,
        "stall_torque_nm": SERVO_STALL_TORQUE_NM,
        "no_load_speed_deg_s": SERVO_NO_LOAD_SPEED_DEG_S,
        "position_kp": SERVO_POSITION_KP,
        "position_kv": SERVO_POSITION_KV,
        "output_armature_kgm2": SERVO_OUTPUT_ARMATURE_KGM2,
        "output_damping_nms_rad": SERVO_OUTPUT_DAMPING_NMS_RAD,
        "gear_friction_nm": SERVO_GEAR_FRICTION_NM,
    }
