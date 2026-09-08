"""Tripod gait controller translated from SW/Controller design documents."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


LEG_PREFIXES = ("RF", "RM", "RB", "LF", "LM", "LB")
TRIPOD_A = frozenset(("RF", "RB", "LM"))
RIGHT_LEGS = frozenset(("RF", "RM", "RB"))


@dataclass(frozen=True)
class GaitConfig:
    control_dt: float = 0.005
    phase_time: float = 0.5
    stand_time: float = 1.0
    ramp_time: float = 1.0
    speed: float = 0.06
    swing_height: float = 0.06
    radial_offset: float = 0.01
    max_joint_speed: float = math.radians(240.0)
    joint_limit: float = math.radians(135.0)


class TripodGaitController:
    """Generate position-actuator targets for the documented tripod gait."""

    L1 = 0.074
    L2 = 0.121
    L3 = 0.230
    NOMINAL_FOOT = np.array((0.218728, 0.0, -0.287006))
    # The CAD/URDF frame is rotated -90 degrees around z from the controller
    # document frame: documented +x (forward) is model-local -y.
    MODEL_FORWARD = np.array((0.0, -1.0, 0.0))

    def __init__(self, model, config: GaitConfig = GaitConfig()):
        import mujoco

        self.model = model
        self.config = config
        self._mujoco = mujoco
        self._actuator_ids: dict[tuple[str, int], int] = {}
        self._origins: dict[str, np.ndarray] = {}
        self._outward: dict[str, np.ndarray] = {}

        for prefix in LEG_PREFIXES:
            body_name = f"{prefix}_motor_horn_1_1"
            body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, body_name
            )
            if body_id < 0:
                raise ValueError(f"Missing body: {body_name}")
            origin = np.asarray(model.body_pos[body_id], dtype=float).copy()
            outward = origin.copy()
            outward[2] = 0.0
            outward /= np.linalg.norm(outward)
            self._origins[prefix] = origin
            self._outward[prefix] = outward

            for joint_number in (1, 2, 3):
                actuator_name = f"{prefix}_{joint_number}_position"
                actuator_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name
                )
                if actuator_id < 0:
                    raise ValueError(f"Missing actuator: {actuator_name}")
                self._actuator_ids[(prefix, joint_number)] = actuator_id

        self._previous = self.home_targets()

    @staticmethod
    def _quintic(tau: float) -> float:
        return 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5

    @classmethod
    def _inverse_kinematics(cls, foot: np.ndarray) -> np.ndarray:
        x, y, z = foot
        theta1 = math.atan2(y, x)
        radius = math.hypot(x, y)
        rho = radius - cls.L1
        cosine3 = (
            rho * rho + z * z - cls.L2**2 - cls.L3**2
        ) / (2.0 * cls.L2 * cls.L3)
        if cosine3 < -1.000001 or cosine3 > 1.000001:
            raise ValueError(f"IK target is outside workspace: {foot.tolist()}")
        cosine3 = float(np.clip(cosine3, -1.0, 1.0))
        theta3 = math.atan2(-math.sqrt(max(0.0, 1.0 - cosine3**2)), cosine3)
        theta2 = math.atan2(z, rho) - math.atan2(
            cls.L3 * math.sin(theta3), cls.L2 + cls.L3 * math.cos(theta3)
        )
        # The design document's servo convention is the opposite sign of the
        # planar geometric angles for joints 2 and 3.
        return np.array((theta1, -theta2, -theta3))

    @staticmethod
    def _servo_to_model(prefix: str, servo_angles: np.ndarray) -> np.ndarray:
        q1, q2, q3 = servo_angles
        if prefix in RIGHT_LEGS:
            return np.array((q1, -q2, q3))
        return np.array((q1, q2, -q3))

    def home_targets(self) -> np.ndarray:
        targets = np.zeros(self.model.nu)
        servo = self._inverse_kinematics(self.NOMINAL_FOOT)
        for prefix in LEG_PREFIXES:
            raw = self._servo_to_model(prefix, servo)
            for index, value in enumerate(raw, start=1):
                targets[self._actuator_ids[(prefix, index)]] = value
        return targets

    def _foot_target(
        self, prefix: str, tau: float, swing: bool, step_length: float
    ) -> np.ndarray:
        smooth = self._quintic(tau)
        if swing:
            offset = step_length * (smooth - 0.5)
            lift = 4.0 * self.config.swing_height * smooth * (1.0 - smooth)
            radial = 4.0 * self.config.radial_offset * smooth * (1.0 - smooth)
        else:
            offset = step_length * (0.5 - tau)
            lift = 0.0
            radial = 0.0

        origin = self._origins[prefix]
        outward = self._outward[prefix]
        nominal_body = origin + outward * self.NOMINAL_FOOT[0]
        nominal_body[2] = origin[2] + self.NOMINAL_FOOT[2]
        target_body = (
            nominal_body
            + self.MODEL_FORWARD * offset
            + outward * radial
            + np.array((0.0, 0.0, lift))
        )

        tangent = np.array((-outward[1], outward[0], 0.0))
        relative = target_body - origin
        return np.array(
            (np.dot(relative, outward), np.dot(relative, tangent), relative[2])
        )

    def targets(self, simulation_time: float) -> np.ndarray:
        if simulation_time < self.config.stand_time:
            desired = self.home_targets()
        else:
            gait_time = simulation_time - self.config.stand_time
            phase_index = int(gait_time // self.config.phase_time)
            tau = (gait_time % self.config.phase_time) / self.config.phase_time
            phase_a = phase_index % 2 == 0
            ramp = min(1.0, gait_time / self.config.ramp_time)
            step_length = self.config.speed * self.config.phase_time * ramp
            desired = np.zeros(self.model.nu)

            for prefix in LEG_PREFIXES:
                swing = (prefix in TRIPOD_A) == phase_a
                foot = self._foot_target(prefix, tau, swing, step_length)
                servo = self._inverse_kinematics(foot)
                raw = self._servo_to_model(prefix, servo)
                for index, value in enumerate(raw, start=1):
                    desired[self._actuator_ids[(prefix, index)]] = value

        desired = np.clip(
            desired, -self.config.joint_limit, self.config.joint_limit
        )
        max_delta = self.config.max_joint_speed * self.config.control_dt
        limited = self._previous + np.clip(
            desired - self._previous, -max_delta, max_delta
        )
        self._previous = limited
        return limited.copy()
