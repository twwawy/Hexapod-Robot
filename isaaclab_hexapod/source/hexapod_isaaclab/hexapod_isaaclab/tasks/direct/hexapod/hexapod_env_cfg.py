"""DirectRLEnv timing and space contract for deterministic MJX replay."""

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from ....assets import HEXAPOD_CFG


@configclass
class HexapodEnvCfg(DirectRLEnvCfg):
    decimation = 8
    episode_length_s = 10.0
    action_space = 18
    observation_space = 146
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=0.0025, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=4.0, replicate_physics=False
    )
    robot_cfg = HEXAPOD_CFG
    golden_replay = True
    firmware_ticks_per_policy_step = 4
    default_command = (0.0, 0.0, 0.0, 0.0, 0.0)
    contact_force_threshold = 1.0
    terrain = None
    command_speed_range = (0.06, 0.12)
    target_clearance = 0.316
    maximum_tilt_rad = 0.7853981633974483
    minimum_clearance = 0.14
