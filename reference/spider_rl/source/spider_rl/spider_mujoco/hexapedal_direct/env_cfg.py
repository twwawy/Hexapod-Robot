from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

COMMAND_SCHEMA: Final[tuple[str, str, str]] = ("vx", "vy", "wz")
ACTION_DIM: Final[int] = 18
OBSERVATION_DIM: Final[int] = 48
LEG_NAMES: Final[tuple[str, ...]] = ("LF", "LM", "LB", "RF", "RM", "RB")
JOINT_NAME_TEMPLATES: Final[tuple[str, str, str]] = (
    "{leg}_motor_horn_1_joint",
    "{leg}_motor_horn_2_joint",
    "{leg}_motor_horn_3_joint",
)
JOINT_NAMES: Final[tuple[str, ...]] = tuple(
    template.format(leg=leg)
    for leg in LEG_NAMES
    for template in JOINT_NAME_TEMPLATES
)
CONTACT_SITE_NAMES: Final[tuple[str, ...]] = tuple(
    f"{leg}_motor_horn_3_1_contact_site" for leg in LEG_NAMES
)
UNDESIRED_CONTACT_BODY_NAMES: Final[tuple[str, ...]] = (
    "base_link",
    *(f"{leg}_motor_horn_1_1" for leg in LEG_NAMES),
    *(f"{leg}_DS51150_270_2_1" for leg in LEG_NAMES),
)
DEFAULT_JOINT_POSITIONS: Final[dict[str, float]] = {
    **{f"{leg}_motor_horn_1_joint": 0.0 for leg in LEG_NAMES},
    **{f"{leg}_motor_horn_2_joint": 0.5 for leg in ("LF", "LM", "LB")},
    **{f"{leg}_motor_horn_2_joint": -0.5 for leg in ("RF", "RM", "RB")},
    **{f"{leg}_motor_horn_3_joint": 0.5 for leg in ("LF", "LM", "LB")},
    **{f"{leg}_motor_horn_3_joint": -0.5 for leg in ("RF", "RM", "RB")},
}
ASSET_DIR: Final[Path] = Path(__file__).resolve().parent / "assets"
XML_PATH: Final[Path] = ASSET_DIR / "hexapedal.xml"
SOURCE_MAP_PATH: Final[Path] = ASSET_DIR / "source_map.yaml"


@dataclass(frozen=True)
class SimulationCfg:
    dt: float = 1.0 / 120.0
    frame_skip: int = 4
    episode_length_s: float = 20.0
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    @property
    def policy_dt(self) -> float:
        return self.dt * self.frame_skip

    @property
    def max_episode_steps(self) -> int:
        return int(round(self.episode_length_s / self.policy_dt))


@dataclass(frozen=True)
class CommandSamplingCfg:
    schema: tuple[str, str, str] = COMMAND_SCHEMA
    vx_range: tuple[float, float] = (-0.3, 0.6)
    vy_range: tuple[float, float] = (0.0, 0.0)
    wz_range: tuple[float, float] = (-0.5, 0.5)
    resample_time_range_s: tuple[float, float] = (10.0, 10.0)


@dataclass(frozen=True)
class PdControllerCfg:
    action_scale: float = 0.5
    stiffness: float = 40.0
    damping: float = 4.0
    action_clip: float = 1.0
    target_velocity: float = 0.0
    use_mujoco_position_actuators: bool = False
    torque_limit: float | None = None


@dataclass(frozen=True)
class RewardScalesCfg:
    lin_vel_xy: float = 2.0
    ang_vel_z: float = 1.0
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    joint_torques: float = -1.0e-5
    joint_acc: float = -2.5e-7
    action_rate: float = -0.01
    action_l2: float = -0.001
    energy: float = -0.0005
    flat_orientation: float = -1.0
    base_height: float = -1.0
    stand_still: float = -0.5
    desired_contact: float = 0.5
    undesired_contact: float = -1.0
    feet_air_time: float = 0.5
    tracking_sigma: float = 0.25


@dataclass(frozen=True)
class ResetNoiseCfg:
    joint_position: float = 0.1
    joint_velocity: float = 0.1
    root_xy: float = 0.0
    root_yaw: float = 0.0


@dataclass(frozen=True)
class TerminationCfg:
    min_base_height: float = 0.10
    upright_threshold: float = 0.3


@dataclass(frozen=True)
class RenderCfg:
    width: int = 960
    height: int = 720
    camera: str | int = -1


@dataclass(frozen=True)
class HexapedalDirectEnvCfg:
    model_path: Path = XML_PATH
    sim: SimulationCfg = field(default_factory=SimulationCfg)
    commands: CommandSamplingCfg = field(default_factory=CommandSamplingCfg)
    pd: PdControllerCfg = field(default_factory=PdControllerCfg)
    rewards: RewardScalesCfg = field(default_factory=RewardScalesCfg)
    reset_noise: ResetNoiseCfg = field(default_factory=ResetNoiseCfg)
    termination: TerminationCfg = field(default_factory=TerminationCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    base_body_name: str = "base_link"
    action_dim: int = ACTION_DIM
    observation_dim: int = OBSERVATION_DIM
    contact_force_threshold: float = 1.0
    target_base_height: float = 0.18
    default_base_position: tuple[float, float, float] = (0.0, 0.0, 0.15)
    command_dim: int = len(COMMAND_SCHEMA)
    joint_names: tuple[str, ...] = JOINT_NAMES
    contact_site_names: tuple[str, ...] = CONTACT_SITE_NAMES
    undesired_contact_body_names: tuple[str, ...] = UNDESIRED_CONTACT_BODY_NAMES
    default_joint_positions: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_JOINT_POSITIONS)
    )
