"""Register the deterministic Hexapod MJX replay task."""

import gymnasium as gym


gym.register(
    id="Hexapod-Firmware-Flat-Direct-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.hexapod_env_cfg:HexapodEnvCfg",
    },
)


gym.register(
    id="Hexapod-Perceptive-Direct-v0",
    entry_point=f"{__name__}.hexapod_env:HexapodEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.perceptive_env_cfg:HexapodPerceptiveEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.agents.rsl_rl_ppo_cfg:HexapodPerceptivePPORunnerCfg"
        ),
    },
)
