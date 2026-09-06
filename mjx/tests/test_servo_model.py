"""Regression tests for the powered DS51150-270 MuJoCo model."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import mujoco
import numpy as np


MJX_DIR = Path(__file__).resolve().parents[1]
if str(MJX_DIR) not in sys.path:
    sys.path.insert(0, str(MJX_DIR))

from prepare_rl_scene import TARGET_ROBOT_MASS_KG, prepare_rl_scene  # noqa: E402
from servo_model import (  # noqa: E402
    SERVO_GEAR_FRICTION_NM,
    SERVO_OUTPUT_ARMATURE_KGM2,
    SERVO_OUTPUT_DAMPING_NMS_RAD,
    SERVO_POSITION_KP,
    SERVO_POSITION_KV,
    SERVO_STALL_TORQUE_NM,
)
from tripod_controller import LEG_PREFIXES  # noqa: E402


class ServoModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.model = mujoco.MjModel.from_xml_path(str(prepare_rl_scene()))
        cls.joint_ids = [
            cls.model.joint(f"{prefix}_{joint_number}").id
            for prefix in LEG_PREFIXES
            for joint_number in (1, 2, 3)
        ]
        cls.dof_ids = [cls.model.jnt_dofadr[joint_id] for joint_id in cls.joint_ids]

    def test_robot_mass_and_ds51150_actuator_contract(self) -> None:
        self.assertAlmostEqual(
            float(np.sum(self.model.body_mass[1:])), TARGET_ROBOT_MASS_KG, places=6
        )
        expected_force_range = np.tile(
            (-SERVO_STALL_TORQUE_NM, SERVO_STALL_TORQUE_NM),
            (self.model.nu, 1),
        )
        np.testing.assert_allclose(
            self.model.actuator_forcerange, expected_force_range, atol=1.0e-8
        )
        np.testing.assert_allclose(
            self.model.actuator_gainprm[:, 0], SERVO_POSITION_KP
        )
        np.testing.assert_allclose(
            -self.model.actuator_biasprm[:, 2], SERVO_POSITION_KV
        )
        np.testing.assert_allclose(
            self.model.dof_armature[self.dof_ids], SERVO_OUTPUT_ARMATURE_KGM2
        )
        np.testing.assert_allclose(
            self.model.dof_damping[self.dof_ids], SERVO_OUTPUT_DAMPING_NMS_RAD
        )
        np.testing.assert_allclose(
            self.model.dof_frictionloss[self.dof_ids], SERVO_GEAR_FRICTION_NM
        )

    def test_powered_home_pose_does_not_splay_under_static_load(self) -> None:
        data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, data, self.model.key("home").id)
        for _ in range(2000):
            mujoco.mj_step(self.model, data)

        qpos_ids = [self.model.jnt_qposadr[joint_id] for joint_id in self.joint_ids]
        max_position_error = float(
            np.max(np.abs(data.qpos[qpos_ids] - data.ctrl))
        )
        self.assertLess(max_position_error, np.deg2rad(0.5))
        self.assertGreater(float(data.qpos[2]), 0.30)


if __name__ == "__main__":
    unittest.main()
