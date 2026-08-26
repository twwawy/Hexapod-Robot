"""Firmware-based MJX residual environment for the terrain curriculum."""

from __future__ import annotations

from typing import Any, Optional

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
from mujoco_playground._src import mjx_env
import numpy as np

import firmware_mjx_controller as firmware
from prepare_rl_scene import (
    RL_SCENE_OUTPUT,
    prepare_rl_scene,
)
from terrain_curriculum import (
    MAX_STAIR_COUNT,
    PLATEAU_DEPTH,
    RAMP_END_X,
    RAMP_LENGTH,
    ROUGH_HFIELD_NCOL,
    ROUGH_HFIELD_NROW,
    ROUGH_LENGTH,
    STAIR_DEPTH,
    TERRAIN_HALF_WIDTH,
    TERRAIN_START_X,
    rough_heightfield_grid,
    terrain_level as terrain_level_spec,
)
from tripod_controller import LEG_PREFIXES
from servo_model import (
    SERVO_SATURATION_START_FRACTION,
    SERVO_STALL_TORQUE_NM,
)


ACTION_SIZE = 18
OBSERVATION_SIZE = 143
ACTION_CONTRACT_VERSION = "stm32_firmware_adaptive_swing_residual_v3"
OBSERVATION_CONTRACT_VERSION = "firmware_state_collision_terrain_pitch_v3"
LEGACY_OBSERVATION_CONTRACT_VERSION = "firmware_state_collision_contact_stairs_v1"
MODEL_FORWARD = jp.array((0.0, -1.0, 0.0))
MODEL_LATERAL = jp.array((1.0, 0.0, 0.0))
PITCH_TARGET_LOOKAHEAD_M = 0.65
PITCH_TARGET_DEADBAND_M = 0.005
PITCH_TARGET_MAX_RAD = 0.48869
PITCH_TARGET_FILTER_TAU_S = 0.15
SWING_BOOST_RISE_M = 0.20
SWING_BOOST_MAX_M = 0.06
SUCCESS_POSTURE_TOLERANCE_RAD = jp.deg2rad(12.0)
FOOT_CLEARANCE_RADIUS_M = 0.05
FOOT_CLEARANCE_MIN_M = 0.02
STAIR_EDGE_MARGIN_M = 0.03
DR_ROOT_POSITION_JITTER_M = 0.01
DR_ROOT_ROTATION_JITTER_RAD = jp.deg2rad(3.0)
DR_JOINT_POSITION_JITTER_RAD = 0.05
DR_PUSH_VELOCITY_MPS = 0.5
DR_PUSH_INTERVAL_MIN_S = 4.0
DR_PUSH_INTERVAL_MAX_S = 8.0
DR_MAX_ACTION_DELAY_TICKS = 2
_DIAGONAL_CLEARANCE_OFFSET = FOOT_CLEARANCE_RADIUS_M / np.sqrt(2.0)
FOOT_CLEARANCE_OFFSETS = jp.asarray(
    (
        (0.0, 0.0),
        (FOOT_CLEARANCE_RADIUS_M, 0.0),
        (-FOOT_CLEARANCE_RADIUS_M, 0.0),
        (0.0, FOOT_CLEARANCE_RADIUS_M),
        (0.0, -FOOT_CLEARANCE_RADIUS_M),
        (_DIAGONAL_CLEARANCE_OFFSET, _DIAGONAL_CLEARANCE_OFFSET),
        (_DIAGONAL_CLEARANCE_OFFSET, -_DIAGONAL_CLEARANCE_OFFSET),
        (-_DIAGONAL_CLEARANCE_OFFSET, _DIAGONAL_CLEARANCE_OFFSET),
        (-_DIAGONAL_CLEARANCE_OFFSET, -_DIAGONAL_CLEARANCE_OFFSET),
    )
)


def _pitch_target(
    forward_heights: jax.Array,
    support_height: jax.Array,
    previous_target: jax.Array,
    dt: float,
) -> jax.Array:
    """Estimate and filter pitch from nine absolute forward terrain heights."""
    if forward_heights.shape != (9,):
        raise ValueError("pitch target requires exactly nine forward heights")
    mean_ahead = jp.nan_to_num(jp.mean(forward_heights), nan=0.0)
    relative_rise = mean_ahead - jp.nan_to_num(support_height, nan=0.0)
    raw_target = jp.arctan2(relative_rise, PITCH_TARGET_LOOKAHEAD_M)
    deadbanded = jp.where(
        jp.abs(relative_rise) < PITCH_TARGET_DEADBAND_M,
        0.0,
        raw_target,
    )
    bounded = jp.clip(deadbanded, -PITCH_TARGET_MAX_RAD, PITCH_TARGET_MAX_RAD)
    alpha = jp.exp(-dt / PITCH_TARGET_FILTER_TAU_S)
    return alpha * jp.nan_to_num(previous_target, nan=0.0) + (1.0 - alpha) * bounded


def _swing_boost(
    forward_heights: jax.Array,
    support_height: jax.Array,
) -> jax.Array:
    """Map the tallest of nine forward samples to a bounded swing lift."""
    if forward_heights.shape != (9,):
        raise ValueError("swing boost requires exactly nine forward heights")
    highest_ahead = jp.nan_to_num(jp.nanmax(forward_heights), nan=0.0)
    relative_rise = highest_ahead - jp.nan_to_num(support_height, nan=0.0)
    return (
        jp.clip(relative_rise / SWING_BOOST_RISE_M, 0.0, 1.0)
        * SWING_BOOST_MAX_M
    )


def _target_relative_attitude(
    attitude: jax.Array,
    pitch_target: jax.Array,
) -> jax.Array:
    return attitude[:2] - jp.stack((jp.asarray(0.0), pitch_target))


def _upright_reward(attitude: jax.Array, pitch_target: jax.Array) -> jax.Array:
    error = _target_relative_attitude(attitude, pitch_target)
    return jp.exp(-jp.sum(jp.square(error)) / 0.12)


def _posture_success(attitude: jax.Array, pitch_target: jax.Array) -> jax.Array:
    error = _target_relative_attitude(attitude, pitch_target)
    return jp.max(jp.abs(error)) < SUCCESS_POSTURE_TOLERANCE_RAD


def _absolute_tilt_failure(attitude: jax.Array, max_tilt: float) -> jax.Array:
    return jp.any(jp.abs(attitude[:2]) > max_tilt)


def _foot_clearance_terrain_cost(
    foot_z: jax.Array,
    local_terrain_height: jax.Array,
    swing_mask: jax.Array,
) -> jax.Array:
    clearance = foot_z - local_terrain_height
    cost = jp.maximum(FOOT_CLEARANCE_MIN_M - clearance, 0.0)
    swing_count = jp.maximum(jp.sum(swing_mask.astype(jp.float32)), 1.0)
    return jp.sum(jp.where(swing_mask, cost, 0.0)) / swing_count


def _edge_margin_cost(foot_x: jax.Array, riser_edges: jax.Array) -> jax.Array:
    if riser_edges.shape[0] == 0:
        return jp.zeros(())
    nearest = jp.min(jp.abs(foot_x[:, None] - riser_edges[None, :]), axis=1)
    return jp.mean(jp.maximum(STAIR_EDGE_MARGIN_M - nearest, 0.0))


def _touchdown_impact_cost(
    previous_contacts: jax.Array,
    contacts: jax.Array,
    pre_contact_vertical_velocity: jax.Array,
) -> jax.Array:
    touchdown = (~previous_contacts) & contacts
    return jp.mean(
        touchdown.astype(jp.float32) * jp.square(pre_contact_vertical_velocity)
    )


def _base_reward_terms(
    *,
    forward_velocity: jax.Array,
    command: jax.Array,
    yaw_velocity: jax.Array,
    attitude: jax.Array,
    pitch_target: jax.Array,
    clearance: jax.Array,
    target_clearance: jax.Array,
    root_angular_speed: jax.Array,
    joint_proximity: jax.Array,
    action: jax.Array,
    last_action: jax.Array,
    swing_height_cost: jax.Array,
    early_swing_contact: jax.Array,
    vertical_velocity: jax.Array,
    lateral_velocity: jax.Array,
    joint_velocity: jax.Array,
    actuator_force: jax.Array,
    torque_limit: jax.Array,
    torque_saturation: jax.Array,
    gait_accepted: jax.Array,
    posture_accepted: jax.Array,
    policy_valid: jax.Array,
    foot_limited: jax.Array,
    body_contact: jax.Array,
    self_collision: jax.Array,
) -> dict[str, jax.Array]:
    """Compute the reward terms that predate stair-contact penalties."""
    return {
        "velocity": jp.exp(
            -jp.square(forward_velocity - command[0]) / 0.01
        ),
        "yaw": jp.exp(-jp.square(yaw_velocity - command[1]) / 0.09),
        "upright": _upright_reward(attitude, pitch_target),
        "height": jp.exp(-jp.square(clearance - target_clearance) / 0.01),
        "progress": jp.clip(forward_velocity, -0.20, 0.25),
        "stability": jp.exp(-jp.square(root_angular_speed / 2.0)),
        "joint_margin": 1.0 - jp.mean(joint_proximity),
        "action_rate": jp.mean(jp.square(action - last_action)),
        "residual": jp.mean(jp.square(action)),
        "swing_height": swing_height_cost,
        "early_swing_contact": early_swing_contact,
        "vertical_velocity": jp.square(vertical_velocity),
        "lateral_velocity": jp.square(lateral_velocity),
        "joint_velocity": jp.mean(jp.square(joint_velocity / 10.0)),
        "torque": jp.mean(jp.square(actuator_force / torque_limit)),
        "torque_saturation": torque_saturation,
        "gait_rejected": (~gait_accepted).astype(jp.float32),
        "posture_rejected": (~posture_accepted).astype(jp.float32),
        "policy_rejected": jp.mean((~policy_valid).astype(jp.float32)),
        "foot_limited": jp.mean(foot_limited.astype(jp.float32)),
        "body_contact": body_contact.astype(jp.float32),
        "self_collision": self_collision.astype(jp.float32),
    }


def _scale_reward_terms(
    terms: dict[str, jax.Array], reward_config: config_dict.ConfigDict
) -> dict[str, jax.Array]:
    return {name: value * reward_config[name] for name, value in terms.items()}


def default_config() -> config_dict.ConfigDict:
    """Training defaults biased toward a stable complete staircase ascent."""
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.0025,
        episode_length=2000,
        impl="jax",
        nconmax=256,
        njmax=128,
        collision_mode="lower_leg",
        command_min_speed=0.06,
        command_max_speed=0.12,
        command_max_yaw_rate=0.0,
        command_delay=1.0,
        dr_enabled=False,
        target_clearance=0.316,
        safety=config_dict.create(
            max_tilt=0.7853981633974483,
            min_clearance=0.14,
            max_root_linear_speed=1.5,
            max_root_angular_speed=6.0,
            max_joint_speed=20.0,
            joint_limit_margin=0.017453292519943295,
        ),
        reward=config_dict.create(
            velocity=2.5,
            yaw=0.25,
            upright=1.0,
            height=0.6,
            progress=1.0,
            stability=0.5,
            joint_margin=1.0,
            action_rate=-0.02,
            residual=-0.005,
            swing_height=-0.10,
            early_swing_contact=-1.0,
            vertical_velocity=-0.10,
            lateral_velocity=-0.10,
            joint_velocity=-0.04,
            torque=-0.03,
            torque_saturation=-0.25,
            gait_rejected=-0.50,
            posture_rejected=-0.50,
            policy_rejected=-2.0,
            foot_limited=-2.0,
            body_contact=-2.0,
            self_collision=-0.5,
            foot_clearance_terrain=-2.0,
            edge_margin=-0.50,
            touchdown_impact=-0.05,
        ),
        ascent_bonus=8.0,
        success_bonus=30.0,
        failure_penalty=-30.0,
    )


def _quat_conjugate(quaternion: jax.Array) -> jax.Array:
    return quaternion * jp.array((1.0, -1.0, -1.0, -1.0))


def _quat_multiply(left: jax.Array, right: jax.Array) -> jax.Array:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return jp.array(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        )
    )


def _euler_to_quat(rpy: jax.Array) -> jax.Array:
    half_roll, half_pitch, half_yaw = rpy * 0.5
    cr, sr = jp.cos(half_roll), jp.sin(half_roll)
    cp, sp = jp.cos(half_pitch), jp.sin(half_pitch)
    cy, sy = jp.cos(half_yaw), jp.sin(half_yaw)
    return jp.asarray(
        (
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        )
    )


def _push_interval_steps(key: jax.Array, dt: float) -> jax.Array:
    interval_s = jax.random.uniform(
        key, (), minval=DR_PUSH_INTERVAL_MIN_S, maxval=DR_PUSH_INTERVAL_MAX_S
    )
    return jp.ceil(interval_s / dt).astype(jp.int32)


def _quat_rotate(quat: jax.Array, vector: jax.Array) -> jax.Array:
    scalar = quat[0]
    xyz = quat[1:]
    temp = 2.0 * jp.cross(xyz, vector)
    return vector + scalar * temp + jp.cross(xyz, temp)


def _quat_rotate_inverse(quat: jax.Array, vector: jax.Array) -> jax.Array:
    return _quat_rotate(_quat_conjugate(quat), vector)


def _quat_to_euler(quaternion: jax.Array) -> jax.Array:
    quaternion = quaternion / jp.maximum(jp.linalg.norm(quaternion), 1.0e-8)
    w, x, y, z = quaternion
    return jp.array(
        (
            jp.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            jp.arcsin(jp.clip(2.0 * (w * y - z * x), -1.0, 1.0)),
            jp.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        )
    )


def mujoco_terrain_foot_contacts(
    contact_geom: jax.Array,
    contact_distance: jax.Array,
    foot_geom_ids: jax.Array,
    geom_body_ids: jax.Array,
) -> jax.Array:
    """Use only real foot-to-world collisions as firmware contact input."""
    safe_geom = jp.maximum(contact_geom, 0)
    first_geom, second_geom = safe_geom[:, 0], safe_geom[:, 1]
    first_world = geom_body_ids[first_geom] == 0
    second_world = geom_body_ids[second_geom] == 0
    first_foot = first_geom[:, None] == foot_geom_ids[None, :]
    second_foot = second_geom[:, None] == foot_geom_ids[None, :]
    pair = (first_foot & second_world[:, None]) | (
        second_foot & first_world[:, None]
    )
    active = (
        (contact_geom[:, 0] >= 0)
        & (contact_geom[:, 1] >= 0)
        & (contact_distance <= 0.0)
    )
    return jp.any(active[:, None] & pair, axis=0)


class HexapodRoughTerrainEnv(mjx_env.MjxEnv):
    """Exact STM32 base controller plus a safety-gated 18-D foot residual.

    The firmware owns command filtering, pose feedback, gait phase, contact
    adaptation, swing/stance trajectories, posture feedback, IK and joint
    rate limiting.  The policy may adjust swing XYZ and stance Z only.  An
    unreachable policy request is rejected before it can reach an actuator.
    """

    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[dict[str, Any]] = None,
        *,
        terrain_level: int = 0,
    ) -> None:
        spec = terrain_level_spec(terrain_level)
        self._terrain_level = terrain_level
        self._terrain_spec = spec
        self._terrain_total_rise = spec.final_height
        self._terrain_step_height = spec.stair_riser
        self._terrain_goal_x = spec.goal_x
        super().__init__(config, config_overrides)
        ratio = self.dt / firmware.FIRMWARE_CONTROL_DT
        if abs(ratio - round(ratio)) > 1.0e-9:
            raise ValueError("ctrl_dt must be an integer multiple of the 5 ms firmware tick")
        self._firmware_steps = int(round(ratio))

        prepare_rl_scene(RL_SCENE_OUTPUT)
        self._xml_path = str(RL_SCENE_OUTPUT)
        self._mj_model = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._configure_terrain_geometry()
        self._configure_collision_masks()
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
        template_data = mjx.make_data(
            self._mj_model,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        self._contact_slots = int(template_data._impl.contact.geom.shape[0])
        self._constraint_rows = int(template_data._impl.efc_J.shape[0])
        if self._contact_slots > 1024:
            raise ValueError(
                "MJX collision graph is unexpectedly large: "
                f"{self._contact_slots} contact slots"
            )

        home_id = self._mj_model.key("home").id
        self._home_qpos = jp.array(self._mj_model.key_qpos[home_id])
        self._home_quaternion = self._home_qpos[3:7]
        self._home_ctrl = jp.array(self._mj_model.key_ctrl[home_id])
        self._joint_qpos_ids = jp.array(
            [
                self._mj_model.joint(f"{prefix}_{joint}").qposadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._joint_qvel_ids = jp.array(
            [
                self._mj_model.joint(f"{prefix}_{joint}").dofadr[0]
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._actuator_ids = jp.array(
            [
                self._mj_model.actuator(f"{prefix}_{joint}_position").id
                for prefix in LEG_PREFIXES
                for joint in (1, 2, 3)
            ]
        )
        self._foot_site_ids = jp.array(
            [self._mj_model.site(f"{prefix}_foot_site").id for prefix in LEG_PREFIXES]
        )
        self._foot_body_ids = jp.array(self._mj_model.site_bodyid)[
            self._foot_site_ids
        ]
        self._foot_geom_ids = jp.array(
            [
                self._mj_model.geom(f"{prefix}_foot_collision").id
                for prefix in LEG_PREFIXES
            ]
        )
        self._geom_body_ids = jp.array(self._mj_model.geom_bodyid)
        self._torso_geom_id = self._mj_model.geom("torso_collision").id
        self._rough_height_grid = jp.array(
            rough_heightfield_grid(spec.rough_amplitude)
        )
        self._step_centers = jp.array(
            [
                TERRAIN_START_X + STAIR_DEPTH * (index + 0.5)
                for index in range(spec.stair_count)
            ]
        )
        self._step_heights = jp.array(
            [
                self._terrain_step_height * (index + 1)
                for index in range(spec.stair_count)
            ]
        )
        self._riser_edges = jp.asarray(
            [
                TERRAIN_START_X + STAIR_DEPTH * index
                for index in range(spec.stair_count)
            ]
        )
        sample_x, sample_y = np.meshgrid(
            np.array((-0.10, 0.15, 0.40, 0.65, 0.90)),
            np.array((-0.24, 0.0, 0.24)),
        )
        self._height_samples = jp.array(
            np.stack((sample_x.ravel(), sample_y.ravel()), axis=-1)
        )
        self._forward_height_samples = self._height_samples[
            self._height_samples[:, 0] >= 0.40
        ]

    def _set_geom_active(
        self, name: str, active: bool, rgba: tuple[float, float, float, float]
    ) -> Any:
        geom = self._mj_model.geom(name)
        self._mj_model.geom_contype[geom.id] = int(active)
        self._mj_model.geom_conaffinity[geom.id] = int(active)
        self._mj_model.geom_rgba[geom.id] = rgba if active else (0.0, 0.0, 0.0, 0.0)
        return geom

    def _configure_terrain_geometry(self) -> None:
        self._set_geom_active("rough_hfield_geom", False, (0, 0, 0, 0))
        self._set_geom_active("terrain_ramp", False, (0, 0, 0, 0))
        for index in range(MAX_STAIR_COUNT):
            self._set_geom_active(f"stair_{index + 1}", False, (0, 0, 0, 0))
        self._set_geom_active("terrain_plateau", False, (0, 0, 0, 0))

        spec = self._terrain_spec
        if spec.kind == "rough":
            self._set_geom_active(
                "rough_hfield_geom", True, (0.32, 0.42, 0.31, 1.0)
            )
            hfield = self._mj_model.hfield("rough_hfield")
            grid = np.asarray(rough_heightfield_grid(spec.rough_amplitude))
            start = self._mj_model.hfield_adr[hfield.id]
            count = (
                self._mj_model.hfield_nrow[hfield.id]
                * self._mj_model.hfield_ncol[hfield.id]
            )
            self._mj_model.hfield_data[start : start + count] = (
                grid.reshape(-1) / spec.rough_amplitude
            )
            self._mj_model.hfield_size[hfield.id, 2] = spec.rough_amplitude
        elif spec.kind == "ramp":
            angle = np.deg2rad(spec.slope_degrees)
            rise = spec.final_height
            half_thickness = 0.025
            ramp = self._set_geom_active(
                "terrain_ramp", True, (0.37, 0.39, 0.45, 1.0)
            )
            self._mj_model.geom_pos[ramp.id] = (
                TERRAIN_START_X + RAMP_LENGTH / 2.0,
                0.0,
                rise / 2.0 - half_thickness * np.cos(angle),
            )
            self._mj_model.geom_size[ramp.id] = (
                RAMP_LENGTH / (2.0 * np.cos(angle)),
                TERRAIN_HALF_WIDTH,
                half_thickness,
            )
            self._mj_model.geom_quat[ramp.id] = (
                np.cos(angle / 2.0),
                0.0,
                -np.sin(angle / 2.0),
                0.0,
            )
            self._configure_plateau(RAMP_END_X, rise)
        elif spec.kind == "stairs":
            for index in range(spec.stair_count):
                height = spec.stair_riser * (index + 1)
                geom = self._set_geom_active(
                    f"stair_{index + 1}", True, (0.34, 0.42, 0.50, 1.0)
                )
                self._mj_model.geom_pos[geom.id, 2] = height / 2.0
                self._mj_model.geom_size[geom.id, 2] = height / 2.0
            stair_end = TERRAIN_START_X + spec.stair_count * STAIR_DEPTH
            self._configure_plateau(stair_end, spec.final_height)

    def _configure_collision_masks(self) -> None:
        """Limit collision pairs while retaining locomotion-critical contacts."""
        mode = self._config.collision_mode
        if mode == "full":
            return
        if mode not in {"terrain", "lower_leg", "feet"}:
            raise ValueError(f"unsupported collision_mode: {mode}")

        for geom_id in range(self._mj_model.ngeom):
            if self._mj_model.geom_bodyid[geom_id] == 0:
                if (
                    self._mj_model.geom_contype[geom_id]
                    or self._mj_model.geom_conaffinity[geom_id]
                ):
                    self._mj_model.geom_contype[geom_id] = 1
                    self._mj_model.geom_conaffinity[geom_id] = 0
                continue

            name = self._mj_model.geom(geom_id).name or ""
            terrain_contact = mode == "terrain"
            if mode in {"lower_leg", "feet"}:
                terrain_contact = name == "torso_collision" or name.endswith(
                    "_foot_collision"
                )
            if mode == "lower_leg":
                terrain_contact |= name.endswith("_tibia_collision")

            # World geoms advertise collision bit 1.  Robot geoms only accept
            # that bit, so robot-vs-robot pairs are absent from the static MJX
            # graph while selected robot-vs-terrain pairs remain physical.
            self._mj_model.geom_contype[geom_id] = 0
            self._mj_model.geom_conaffinity[geom_id] = int(terrain_contact)

    def _configure_plateau(self, start_x: float, height: float) -> None:
        plateau = self._set_geom_active(
            "terrain_plateau", True, (0.30, 0.39, 0.48, 1.0)
        )
        self._mj_model.geom_pos[plateau.id] = (
            start_x + PLATEAU_DEPTH / 2.0,
            0.0,
            height / 2.0,
        )
        self._mj_model.geom_size[plateau.id] = (
            PLATEAU_DEPTH / 2.0,
            TERRAIN_HALF_WIDTH,
            height / 2.0,
        )

    def _relative_attitude(self, data: mjx.Data) -> jax.Array:
        relative = _quat_multiply(
            data.qpos[3:7], _quat_conjugate(self._home_quaternion)
        )
        return _quat_to_euler(relative)

    def _terrain_height(self, xy: jax.Array) -> jax.Array:
        x, y = xy[..., 0], xy[..., 1]
        spec = self._terrain_spec
        if spec.kind == "flat":
            return jp.zeros(x.shape)
        if spec.kind == "rough":
            column = (x - TERRAIN_START_X) * (ROUGH_HFIELD_NCOL - 1) / ROUGH_LENGTH
            row = (y + TERRAIN_HALF_WIDTH) * (ROUGH_HFIELD_NROW - 1) / (
                2.0 * TERRAIN_HALF_WIDTH
            )
            column0 = jp.clip(
                jp.floor(column).astype(jp.int32), 0, ROUGH_HFIELD_NCOL - 2
            )
            row0 = jp.clip(
                jp.floor(row).astype(jp.int32), 0, ROUGH_HFIELD_NROW - 2
            )
            column_fraction = jp.clip(column - column0, 0.0, 1.0)
            row_fraction = jp.clip(row - row0, 0.0, 1.0)
            low = (
                self._rough_height_grid[row0, column0] * (1.0 - column_fraction)
                + self._rough_height_grid[row0, column0 + 1] * column_fraction
            )
            high = (
                self._rough_height_grid[row0 + 1, column0]
                * (1.0 - column_fraction)
                + self._rough_height_grid[row0 + 1, column0 + 1]
                * column_fraction
            )
            height = low * (1.0 - row_fraction) + high * row_fraction
            inside = (
                (x >= TERRAIN_START_X)
                & (x <= TERRAIN_START_X + ROUGH_LENGTH)
                & (jp.abs(y) <= TERRAIN_HALF_WIDTH)
            )
            return jp.where(inside, height, 0.0)
        inside_width = jp.abs(y) <= TERRAIN_HALF_WIDTH
        if spec.kind == "ramp":
            on_ramp = (
                (x >= TERRAIN_START_X) & (x <= RAMP_END_X) & inside_width
            )
            on_plateau = (
                (x > RAMP_END_X)
                & (x <= RAMP_END_X + PLATEAU_DEPTH)
                & inside_width
            )
            ramp_height = (x - TERRAIN_START_X) * jp.tan(
                jp.deg2rad(spec.slope_degrees)
            )
            return jp.where(
                on_ramp,
                jp.clip(ramp_height, 0.0, spec.final_height),
                jp.where(on_plateau, spec.final_height, 0.0),
            )
        inside_step = (
            (jp.abs(x[..., None] - self._step_centers) <= STAIR_DEPTH / 2.0)
            & inside_width[..., None]
        )
        stair_height = jp.max(jp.where(inside_step, self._step_heights, 0.0), axis=-1)
        stair_end = TERRAIN_START_X + spec.stair_count * STAIR_DEPTH
        on_plateau = (
            (x > stair_end)
            & (x <= stair_end + PLATEAU_DEPTH)
            & inside_width
        )
        return jp.where(on_plateau, spec.final_height, stair_height)

    def _foot_contacts(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return mujoco_terrain_foot_contacts(
            contact.geom, contact.dist, self._foot_geom_ids, self._geom_body_ids
        )

    def _body_contact(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        return jp.any(
            (contact.dist < 0.0)
            & jp.any(contact.geom == self._torso_geom_id, axis=-1)
        )

    def _self_collision(self, data: mjx.Data) -> jax.Array:
        contact = data._impl.contact
        safe_geom = jp.maximum(contact.geom, 0)
        first_body = self._geom_body_ids[safe_geom[:, 0]]
        second_body = self._geom_body_ids[safe_geom[:, 1]]
        active = (
            (contact.geom[:, 0] >= 0)
            & (contact.geom[:, 1] >= 0)
            & (contact.dist < 0.0)
        )
        return jp.any(active & (first_body > 0) & (second_body > 0))

    def _support_height(self, data: mjx.Data, contacts: jax.Array) -> jax.Array:
        feet = data.site_xpos[self._foot_site_ids]
        heights = self._terrain_height(feet[:, :2])
        count = jp.sum(contacts.astype(jp.int32))
        ordered = jp.sort(jp.where(contacts, heights, jp.inf))
        median_index = jp.maximum((count - 1) // 2, 0)
        median = ordered[median_index]
        fallback = self._terrain_height(data.qpos[:2])
        return jp.where(count > 0, median, fallback)

    def _feet_controller_body(self, data: mjx.Data) -> jax.Array:
        relative_world = data.site_xpos[self._foot_site_ids] - data.qpos[None, :3]
        model_body = jax.vmap(_quat_rotate_inverse, in_axes=(None, 0))(
            data.qpos[3:7], relative_world
        )
        return jp.stack(
            (
                model_body @ MODEL_FORWARD,
                model_body @ MODEL_LATERAL,
                model_body[:, 2],
            ),
            axis=-1,
        )

    def _local_terrain_max(self, foot_xy: jax.Array) -> jax.Array:
        samples = foot_xy[:, None, :] + FOOT_CLEARANCE_OFFSETS[None, :, :]
        heights = self._terrain_height(samples.reshape((-1, 2)))
        return jp.max(heights.reshape((6, -1)), axis=1)

    def _foot_vertical_velocity(self, data: mjx.Data) -> jax.Array:
        def site_velocity(point: jax.Array, body_id: jax.Array) -> jax.Array:
            jacobian_position, _ = mjx.jac(self.mjx_model, data, point, body_id)
            return data.qvel @ jacobian_position[:, 2]

        return jax.vmap(site_velocity)(
            data.site_xpos[self._foot_site_ids], self._foot_body_ids
        )

    def _terrain_features(self, data: mjx.Data, support_height: jax.Array) -> jax.Array:
        attitude = self._relative_attitude(data)
        cosine, sine = jp.cos(attitude[2]), jp.sin(attitude[2])
        rotation = jp.array(((cosine, -sine), (sine, cosine)))
        sample_world = data.qpos[None, :2] + self._height_samples @ rotation.T
        return self._terrain_height(sample_world) - support_height

    def _terrain_pitch_target(
        self,
        data: mjx.Data,
        support_height: jax.Array,
        previous_target: jax.Array,
    ) -> jax.Array:
        attitude = self._relative_attitude(data)
        cosine, sine = jp.cos(attitude[2]), jp.sin(attitude[2])
        rotation = jp.array(((cosine, -sine), (sine, cosine)))
        sample_world = (
            data.qpos[None, :2] + self._forward_height_samples @ rotation.T
        )
        return _pitch_target(
            self._terrain_height(sample_world),
            support_height,
            previous_target,
            self.dt,
        )

    def _terrain_swing_boost(
        self,
        data: mjx.Data,
        support_height: jax.Array,
    ) -> jax.Array:
        attitude = self._relative_attitude(data)
        cosine, sine = jp.cos(attitude[2]), jp.sin(attitude[2])
        rotation = jp.array(((cosine, -sine), (sine, cosine)))
        sample_world = (
            data.qpos[None, :2] + self._forward_height_samples @ rotation.T
        )
        return _swing_boost(self._terrain_height(sample_world), support_height)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        quat = data.qpos[3:7]
        local_velocity = _quat_rotate_inverse(quat, data.qvel[:3])
        local_gravity = _quat_rotate_inverse(quat, jp.array((0.0, 0.0, -1.0)))
        output = info["controller_output"]
        observation = jp.concatenate(
            (
                info["command"],
                local_velocity,
                0.2 * data.qvel[3:6],
                local_gravity,
                self._relative_attitude(data)[:2],
                data.qpos[self._joint_qpos_ids]
                - self._home_qpos[self._joint_qpos_ids],
                0.1 * data.qvel[self._joint_qvel_ids],
                jp.ravel(self._feet_controller_body(data)),
                info["contact_state"].astype(jp.float32),
                self._terrain_features(data, info["support_height"]),
                output.gait_progress,
                output.gait_state.astype(jp.float32) / float(firmware.LEG_LATE_LANDING),
                output.applied_twist,
                output.ik_valid.astype(jp.float32),
                output.policy_valid.astype(jp.float32),
                output.foot_limited.astype(jp.float32),
                jp.array(
                    (
                        output.gait_accepted.astype(jp.float32),
                        output.posture_accepted.astype(jp.float32),
                    )
                ),
                info["last_action"],
                jp.atleast_1d(info["pitch_target"]),
            )
        )
        return observation

    def reset(self, rng: jax.Array) -> mjx_env.State:
        if self._config.dr_enabled:
            (
                rng,
                q_key,
                vel_key,
                cmd_key,
                yaw_key,
                root_key,
                rotation_key,
                push_key,
                delay_key,
            ) = jax.random.split(rng, 9)
            joint_jitter = DR_JOINT_POSITION_JITTER_RAD
        else:
            rng, q_key, vel_key, cmd_key, yaw_key = jax.random.split(rng, 5)
            joint_jitter = 0.01
        qpos = self._home_qpos.at[self._joint_qpos_ids].add(
            jax.random.uniform(
                q_key, (18,), minval=-joint_jitter, maxval=joint_jitter
            )
        )
        if self._config.dr_enabled:
            root_position_delta = jax.random.uniform(
                root_key,
                (3,),
                minval=-DR_ROOT_POSITION_JITTER_M,
                maxval=DR_ROOT_POSITION_JITTER_M,
            )
            rotation_delta = jax.random.uniform(
                rotation_key,
                (3,),
                minval=-DR_ROOT_ROTATION_JITTER_RAD,
                maxval=DR_ROOT_ROTATION_JITTER_RAD,
            )
            qpos = qpos.at[:3].add(root_position_delta)
            qpos = qpos.at[3:7].set(
                _quat_multiply(_euler_to_quat(rotation_delta), qpos[3:7])
            )
            next_push_step = _push_interval_steps(push_key, self.dt)
            action_delay_ticks = jax.random.randint(
                delay_key, (), 0, DR_MAX_ACTION_DELAY_TICKS + 1
            )
        else:
            next_push_step = jp.asarray(jp.iinfo(jp.int32).max, dtype=jp.int32)
            action_delay_ticks = jp.zeros((), dtype=jp.int32)
        qvel = jp.zeros(self.mjx_model.nv).at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.01, maxval=0.01)
        )
        speed = jax.random.uniform(
            cmd_key,
            (),
            minval=self._config.command_min_speed,
            maxval=self._config.command_max_speed,
        )
        yaw_rate = jax.random.uniform(
            yaw_key,
            (),
            minval=-self._config.command_max_yaw_rate,
            maxval=self._config.command_max_yaw_rate,
        )
        data = mjx.make_data(
            self.mj_model,
            impl=self.mjx_model.impl.value,
            naconmax=self._config.nconmax,
            njmax=self._config.njmax,
        )
        data = data.replace(qpos=qpos, qvel=qvel, ctrl=self._home_ctrl)
        data = mjx.forward(self.mjx_model, data)
        contacts = self._foot_contacts(data)
        support_height = self._support_height(data, contacts)
        controller_output = firmware.initial_output()
        info = {
            "rng": rng,
            "command": jp.array((speed, yaw_rate)),
            "steps": jp.zeros((), dtype=jp.int32),
            "last_action": jp.zeros(ACTION_SIZE),
            # The policy tick is 20 ms, so delays {0,1,2} mean {0,20,40} ms.
            "action_delay_buffer": jp.zeros(
                (DR_MAX_ACTION_DELAY_TICKS + 1, ACTION_SIZE)
            ),
            "action_delay_ticks": action_delay_ticks,
            "next_push_step": next_push_step,
            "controller_state": firmware.initial_state(),
            "controller_output": controller_output,
            "contact_state": contacts,
            "support_height": support_height,
            "pitch_target": jp.zeros(()),
            "swing_boost": jp.zeros(()),
            "max_terrain_height": support_height,
            "previous_root_x": data.qpos[0],
            "controller_rejection_steps": jp.zeros((), dtype=jp.int32),
        }
        metrics = {
            f"reward/{name}": jp.zeros(()) for name in self._config.reward.keys()
        }
        metrics.update(
            {
                "reward/ascent": jp.zeros(()),
                "reward/success": jp.zeros(()),
                "reward/failure": jp.zeros(()),
                "terrain_level": jp.zeros(()),
                "terrain_success": jp.zeros(()),
                "controller_rejection_steps": jp.zeros(()),
                "policy_rejection_fraction": jp.zeros(()),
                "foot_limited_fraction": jp.zeros(()),
                "joint_limit_margin_rad": jp.zeros(()),
                "root_linear_speed": jp.zeros(()),
                "root_angular_speed": jp.zeros(()),
                "joint_speed_max": jp.zeros(()),
                "swing_height_mean_m": jp.asarray(firmware.SWING_HEIGHT),
                "swing_height_max_m": jp.asarray(firmware.SWING_HEIGHT),
                "swing_height_boost_fraction": jp.zeros(()),
                "early_swing_contact_fraction": jp.zeros(()),
                "termination/controller_invalid": jp.zeros(()),
                "termination/joint_limit": jp.zeros(()),
                "termination/dynamics": jp.zeros(()),
                "termination/tilt": jp.zeros(()),
                "termination/clearance": jp.zeros(()),
                "termination/body_contact": jp.zeros(()),
                "termination/nonfinite": jp.zeros(()),
            }
        )
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        action = jp.clip(action, -1.0, 1.0)
        action_delay_buffer = jp.concatenate(
            (action[None, :], state.info["action_delay_buffer"][:-1]), axis=0
        )
        action = action_delay_buffer[state.info["action_delay_ticks"]]
        next_push_step = state.info["next_push_step"]
        info_rng = state.info["rng"]
        if self._config.dr_enabled:
            info_rng, push_velocity_key, push_interval_key = jax.random.split(
                info_rng, 3
            )
            push_due = state.info["steps"] >= next_push_step
            push_velocity = jax.random.uniform(
                push_velocity_key,
                (2,),
                minval=-DR_PUSH_VELOCITY_MPS,
                maxval=DR_PUSH_VELOCITY_MPS,
            )
            qvel = state.data.qvel.at[:2].add(
                jp.where(push_due, push_velocity, jp.zeros(2))
            )
            state = state.replace(data=state.data.replace(qvel=qvel))
            next_push_step = jp.where(
                push_due,
                state.info["steps"]
                + _push_interval_steps(push_interval_key, self.dt),
                next_push_step,
            )
        contacts_before = self._foot_contacts(state.data)
        pre_contact_foot_vz = self._foot_vertical_velocity(state.data)
        attitude_before = self._relative_attitude(state.data)
        pitch_target = self._terrain_pitch_target(
            state.data,
            state.info["support_height"],
            state.info["pitch_target"],
        )
        swing_boost = self._terrain_swing_boost(
            state.data,
            state.info["support_height"],
        )
        command_active = state.info["steps"] * self.dt >= self._config.command_delay
        firmware_command = jp.where(command_active, state.info["command"], jp.zeros(2))

        def controller_tick(controller_state, _):
            return firmware.step(
                controller_state,
                target_velocity=firmware_command,
                body_position_world=state.data.qpos[:3],
                attitude_rpy=attitude_before,
                contacts=contacts_before,
                policy_action=action,
                pitch_target=pitch_target,
                swing_boost=swing_boost,
            )

        controller_state, controller_history = jax.lax.scan(
            controller_tick,
            state.info["controller_state"],
            xs=None,
            length=self._firmware_steps,
        )
        controller = jax.tree_util.tree_map(lambda value: value[-1], controller_history)
        targets = self._home_ctrl.at[self._actuator_ids].set(
            controller.model_joint_targets.reshape(18)
        )
        data = mjx_env.step(self.mjx_model, state.data, targets, self.n_substeps)

        attitude = self._relative_attitude(data)
        raw_local_velocity = _quat_rotate_inverse(data.qpos[3:7], data.qvel[:3])
        forward_velocity = jp.dot(raw_local_velocity, MODEL_FORWARD)
        lateral_velocity = jp.dot(raw_local_velocity, MODEL_LATERAL)
        contacts = self._foot_contacts(data)
        support_height = self._support_height(data, contacts)
        clearance = data.qpos[2] - support_height
        body_contact = self._body_contact(data)
        self_collision = self._self_collision(data)

        root_linear_speed = jp.linalg.norm(data.qvel[:3])
        root_angular_speed = jp.linalg.norm(data.qvel[3:6])
        joint_speed_max = jp.max(jp.abs(data.qvel[self._joint_qvel_ids]))
        joint_limit_margin = firmware.JOINT_LIMIT - jp.max(
            jp.abs(controller.servo_joint_targets)
        )
        controller_invalid = jp.any(~controller.ik_valid)
        joint_limit_failure = joint_limit_margin <= self._config.safety.joint_limit_margin
        dynamics_failure = (
            (root_linear_speed > self._config.safety.max_root_linear_speed)
            | (root_angular_speed > self._config.safety.max_root_angular_speed)
            | (joint_speed_max > self._config.safety.max_joint_speed)
        )
        tilt_failure = _absolute_tilt_failure(
            attitude, self._config.safety.max_tilt
        )
        clearance_failure = clearance < self._config.safety.min_clearance
        finite = (
            jp.all(jp.isfinite(data.qpos))
            & jp.all(jp.isfinite(data.qvel))
            & jp.all(jp.isfinite(controller.servo_joint_targets))
        )
        failure = (
            controller_invalid
            | joint_limit_failure
            | dynamics_failure
            | tilt_failure
            | clearance_failure
            | body_contact
            | (~finite)
        )
        final_height_ready = (
            support_height >= self._terrain_total_rise - 1.0e-3
            if self._terrain_spec.requires_final_height
            else jp.ones((), dtype=jp.bool_)
        )
        success = (
            (data.qpos[0] >= self._terrain_goal_x)
            & final_height_ready
            & _posture_success(attitude, pitch_target)
            & (~failure)
        )
        terminated = failure | success

        torque_limit = SERVO_STALL_TORQUE_NM
        saturation_start = (
            SERVO_SATURATION_START_FRACTION * SERVO_STALL_TORQUE_NM
        )
        saturation_width = SERVO_STALL_TORQUE_NM - saturation_start
        torque_saturation = jp.mean(
            jp.square(
                jp.maximum(jp.abs(data.actuator_force) - saturation_start, 0.0)
                / saturation_width
            )
        )
        joint_proximity = jp.clip(
            (jp.abs(controller.servo_joint_targets) - jp.deg2rad(105.0))
            / jp.deg2rad(30.0),
            0.0,
            1.0,
        )
        rejection = (~controller.gait_accepted) | (~controller.posture_accepted)
        rejection_steps = jp.where(
            rejection, state.info["controller_rejection_steps"] + 1, 0
        )
        policy_rejected = jp.mean((~controller.policy_valid).astype(jp.float32))
        foot_limited = jp.mean(controller.foot_limited.astype(jp.float32))
        swing_mask = controller.gait_state == firmware.LEG_SWING
        swing_count = jp.maximum(jp.sum(swing_mask.astype(jp.float32)), 1.0)
        normalized_swing_height = jp.clip(
            (
                controller.swing_height_command - firmware.SWING_HEIGHT_MIN
            )
            / (firmware.SWING_HEIGHT_MAX - firmware.SWING_HEIGHT_MIN),
            0.0,
            1.0,
        )
        swing_height_cost = jp.sum(
            jp.where(swing_mask, jp.square(normalized_swing_height), 0.0)
        ) / swing_count
        swing_height_mean = jp.sum(
            jp.where(swing_mask, controller.swing_height_command, 0.0)
        ) / swing_count
        swing_height_mean = jp.where(
            jp.any(swing_mask), swing_height_mean, firmware.SWING_HEIGHT
        )
        swing_height_max = jp.max(
            jp.where(
                swing_mask,
                controller.swing_height_command,
                firmware.SWING_HEIGHT,
            )
        )
        swing_height_boost = jp.sum(
            jp.where(
                swing_mask,
                jp.maximum(
                    controller.swing_height_command - firmware.SWING_HEIGHT,
                    0.0,
                )
                / (firmware.SWING_HEIGHT_MAX - firmware.SWING_HEIGHT),
                0.0,
            )
        ) / swing_count
        early_swing_contact = jp.mean(
            (
                swing_mask
                & controller_state.airborne_seen
                & contacts
                & (controller.gait_progress < firmware.EARLY_LANDING_PROGRESS)
            ).astype(jp.float32)
        )
        feet_world = data.site_xpos[self._foot_site_ids]
        reward_terms = _base_reward_terms(
            forward_velocity=forward_velocity,
            command=state.info["command"],
            yaw_velocity=data.qvel[5],
            attitude=attitude,
            pitch_target=pitch_target,
            clearance=clearance,
            target_clearance=self._config.target_clearance,
            root_angular_speed=root_angular_speed,
            joint_proximity=joint_proximity,
            action=action,
            last_action=state.info["last_action"],
            swing_height_cost=swing_height_cost,
            early_swing_contact=early_swing_contact,
            vertical_velocity=data.qvel[2],
            lateral_velocity=lateral_velocity,
            joint_velocity=data.qvel[self._joint_qvel_ids],
            actuator_force=data.actuator_force,
            torque_limit=torque_limit,
            torque_saturation=torque_saturation,
            gait_accepted=controller.gait_accepted,
            posture_accepted=controller.posture_accepted,
            policy_valid=controller.policy_valid,
            foot_limited=controller.foot_limited,
            body_contact=body_contact,
            self_collision=self_collision,
        )
        reward_terms.update(
            {
                "foot_clearance_terrain": _foot_clearance_terrain_cost(
                    feet_world[:, 2],
                    self._local_terrain_max(feet_world[:, :2]),
                    swing_mask,
                ),
                "edge_margin": _edge_margin_cost(
                    feet_world[:, 0], self._riser_edges
                ),
                "touchdown_impact": _touchdown_impact_cost(
                    contacts_before, contacts, pre_contact_foot_vz
                ),
            }
        )
        scaled = _scale_reward_terms(reward_terms, self._config.reward)
        running_reward = sum(scaled.values()) * self.dt
        positive_ascent = jp.maximum(
            support_height - state.info["max_terrain_height"], 0.0
        )
        max_terrain_height = jp.maximum(
            state.info["max_terrain_height"], support_height
        )
        ascent_bonus = (
            self._config.ascent_bonus
            * positive_ascent
            / self._terrain_total_rise
            if self._terrain_spec.requires_final_height
            else jp.zeros(())
        )
        success_bonus = jp.where(success, self._config.success_bonus, 0.0)
        failure_penalty = jp.where(failure, self._config.failure_penalty, 0.0)
        reward = jp.clip(
            running_reward + ascent_bonus + success_bonus + failure_penalty,
            -50.0,
            50.0,
        )

        state.info["steps"] += 1
        state.info["rng"] = info_rng
        state.info["last_action"] = action
        state.info["action_delay_buffer"] = action_delay_buffer
        state.info["next_push_step"] = next_push_step
        state.info["controller_state"] = controller_state
        state.info["controller_output"] = controller
        state.info["contact_state"] = contacts
        state.info["support_height"] = support_height
        state.info["pitch_target"] = pitch_target
        state.info["swing_boost"] = swing_boost
        state.info["max_terrain_height"] = max_terrain_height
        state.info["previous_root_x"] = data.qpos[0]
        state.info["controller_rejection_steps"] = rejection_steps
        obs = self._get_obs(data, state.info)
        for name, value in scaled.items():
            state.metrics[f"reward/{name}"] = value
        state.metrics["reward/ascent"] = ascent_bonus
        state.metrics["reward/success"] = success_bonus
        state.metrics["reward/failure"] = failure_penalty
        state.metrics["terrain_level"] = jp.asarray(float(self._terrain_level))
        state.metrics["terrain_success"] = success.astype(jp.float32)
        state.metrics["controller_rejection_steps"] = rejection_steps.astype(jp.float32)
        state.metrics["policy_rejection_fraction"] = policy_rejected
        state.metrics["foot_limited_fraction"] = foot_limited
        state.metrics["joint_limit_margin_rad"] = joint_limit_margin
        state.metrics["root_linear_speed"] = root_linear_speed
        state.metrics["root_angular_speed"] = root_angular_speed
        state.metrics["joint_speed_max"] = joint_speed_max
        state.metrics["swing_height_mean_m"] = swing_height_mean
        state.metrics["swing_height_max_m"] = swing_height_max
        state.metrics["swing_height_boost_fraction"] = swing_height_boost
        state.metrics["early_swing_contact_fraction"] = early_swing_contact
        state.metrics["termination/controller_invalid"] = controller_invalid.astype(jp.float32)
        state.metrics["termination/joint_limit"] = joint_limit_failure.astype(jp.float32)
        state.metrics["termination/dynamics"] = dynamics_failure.astype(jp.float32)
        state.metrics["termination/tilt"] = tilt_failure.astype(jp.float32)
        state.metrics["termination/clearance"] = clearance_failure.astype(jp.float32)
        state.metrics["termination/body_contact"] = body_contact.astype(jp.float32)
        state.metrics["termination/nonfinite"] = (~finite).astype(jp.float32)
        return state.replace(
            data=data,
            obs=obs,
            reward=reward,
            done=terminated.astype(jp.float32),
        )

    @property
    def action_size(self) -> int:
        return ACTION_SIZE

    @property
    def observation_size(self) -> int:
        return OBSERVATION_SIZE

    @property
    def episode_length(self) -> int:
        return int(self._config.episode_length)

    @property
    def curriculum_level(self) -> int:
        return self._terrain_level

    @property
    def terrain_step_height(self) -> float:
        return self._terrain_step_height

    @property
    def terrain_total_rise(self) -> float:
        return self._terrain_total_rise

    @property
    def terrain_name(self) -> str:
        return self._terrain_spec.name

    @property
    def terrain_kind(self) -> str:
        return self._terrain_spec.kind

    @property
    def terrain_description(self) -> str:
        return self._terrain_spec.description

    @property
    def terrain_goal_x(self) -> float:
        return self._terrain_goal_x

    @property
    def terrain_stair_count(self) -> int:
        return self._terrain_spec.stair_count

    @property
    def contact_slots(self) -> int:
        """Static MJX contact slots compiled for one environment."""
        return self._contact_slots

    @property
    def constraint_rows(self) -> int:
        """Static scalar constraint rows compiled for one environment."""
        return self._constraint_rows

    @property
    def collision_mode(self) -> str:
        return str(self._config.collision_mode)

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
