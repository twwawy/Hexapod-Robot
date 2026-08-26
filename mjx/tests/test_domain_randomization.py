from __future__ import annotations

import functools
from pathlib import Path
import sys
import unittest

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from domain_randomization import (
    ARMATURE_RANGE,
    DAMPING_RANGE,
    FRICTION_RANGE,
    KP_RANGE,
    MASS_RANGE,
    build_bank,
    randomize_batch,
)
from rough_terrain_env import (
    DR_JOINT_POSITION_JITTER_RAD,
    DR_ROOT_POSITION_JITTER_M,
    DR_ROOT_ROTATION_JITTER_RAD,
    HexapodRoughTerrainEnv,
    _quat_conjugate,
    _quat_multiply,
    _quat_to_euler,
    default_config,
)


class DomainRandomizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.baseline = HexapodRoughTerrainEnv(terrain_level=0)

    def test_fixed_seed_models_have_distinct_foot_friction(self) -> None:
        bank, _ = build_bank(self.baseline.mjx_model, 16, seed=7)
        foot_id = int(self.baseline._foot_geom_ids[0])
        self.assertNotEqual(
            float(bank.geom_friction[0, foot_id, 0]),
            float(bank.geom_friction[1, foot_id, 0]),
        )

    def test_nonfinite_base_friction_is_rejected_at_build_time(self) -> None:
        model = self.baseline.mjx_model
        foot_id = int(self.baseline._foot_geom_ids[0])
        invalid = model.tree_replace(
            {
                "geom_friction": model.geom_friction.at[foot_id, 0].set(jp.nan)
            }
        )
        with self.assertRaisesRegex(ValueError, "geom_friction.*NaN/Inf"):
            build_bank(invalid, 16, seed=0)

    def test_one_hundred_seeds_stay_inside_all_bounds(self) -> None:
        model = self.baseline.mjx_model
        foot_id = int(self.baseline._foot_geom_ids[0])
        for seed in range(100):
            bank, _ = build_bank(model, 16, seed)
            ratios = {
                "friction": np.asarray(
                    bank.geom_friction[:, foot_id, 0]
                    / model.geom_friction[foot_id, 0]
                ),
                "mass": np.asarray(bank.body_mass[:, 1] / model.body_mass[1]),
                "inertia": np.asarray(
                    bank.body_inertia[:, 1, 0] / model.body_inertia[1, 0]
                ),
                "kp": np.asarray(
                    bank.actuator_gainprm[:, 0, 0]
                    / model.actuator_gainprm[0, 0]
                ),
                "armature": np.asarray(
                    bank.dof_armature[:, 6] / model.dof_armature[6]
                ),
                "damping": np.asarray(
                    bank.dof_damping[:, 6] / model.dof_damping[6]
                ),
            }
            for name, bounds in (
                ("friction", FRICTION_RANGE),
                ("mass", MASS_RANGE),
                ("inertia", MASS_RANGE),
                ("kp", KP_RANGE),
                ("armature", ARMATURE_RANGE),
                ("damping", DAMPING_RANGE),
            ):
                self.assertTrue(np.all(ratios[name] >= bounds[0]))
                self.assertTrue(np.all(ratios[name] <= bounds[1]))
            np.testing.assert_allclose(ratios["mass"], ratios["inertia"], atol=1e-7)

    def test_bank_size_one_is_exact_baseline(self) -> None:
        model = self.baseline.mjx_model
        bank, _ = build_bank(model, 1, seed=999)
        for field in (
            "geom_friction",
            "body_mass",
            "body_inertia",
            "actuator_gainprm",
            "actuator_biasprm",
            "dof_armature",
            "dof_damping",
        ):
            np.testing.assert_allclose(
                np.asarray(getattr(bank, field)[0]),
                np.asarray(getattr(model, field)),
                atol=0.0,
                rtol=0.0,
            )

        key = jax.random.PRNGKey(11)
        _, q_key, vel_key, cmd_key, yaw_key = jax.random.split(key, 5)
        state = self.baseline.reset(key)
        expected_qpos = self.baseline._home_qpos.at[
            self.baseline._joint_qpos_ids
        ].add(jax.random.uniform(q_key, (18,), minval=-0.01, maxval=0.01))
        expected_qvel = jp.zeros(self.baseline.mjx_model.nv).at[:6].set(
            jax.random.uniform(vel_key, (6,), minval=-0.01, maxval=0.01)
        )
        expected_command = jp.concatenate(
            (
                jp.asarray(
                    (
                        jax.random.uniform(
                            cmd_key,
                            (),
                            minval=self.baseline._config.command_min_speed,
                            maxval=self.baseline._config.command_max_speed,
                        ),
                        jax.random.uniform(
                            yaw_key,
                            (),
                            minval=-self.baseline._config.command_max_yaw_rate,
                            maxval=self.baseline._config.command_max_yaw_rate,
                        ),
                    )
                ),
                jp.zeros(3),
            )
        )
        np.testing.assert_allclose(np.asarray(state.data.qpos), expected_qpos, atol=1e-7)
        np.testing.assert_allclose(np.asarray(state.data.qvel), expected_qvel, atol=1e-7)
        np.testing.assert_allclose(np.asarray(state.info["command"]), expected_command)

    def test_state_randomization_bounds_for_one_hundred_resets(self) -> None:
        config = default_config()
        config.dr_enabled = True
        env = HexapodRoughTerrainEnv(config=config, terrain_level=0)
        states = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.PRNGKey(3), 100))
        root_delta = np.asarray(states.data.qpos[:, :3] - env._home_qpos[:3])
        joint_delta = np.asarray(
            states.data.qpos[:, env._joint_qpos_ids]
            - env._home_qpos[env._joint_qpos_ids]
        )
        relative_quaternion = jax.vmap(
            lambda quat: _quat_multiply(
                quat, _quat_conjugate(env._home_qpos[3:7])
            )
        )(states.data.qpos[:, 3:7])
        rotation_delta = np.asarray(jax.vmap(_quat_to_euler)(relative_quaternion))
        self.assertTrue(np.all(np.abs(root_delta) <= DR_ROOT_POSITION_JITTER_M + 1e-7))
        self.assertTrue(
            np.all(np.abs(joint_delta) <= DR_JOINT_POSITION_JITTER_RAD + 1e-7)
        )
        self.assertTrue(
            np.all(np.abs(rotation_delta) <= float(DR_ROOT_ROTATION_JITTER_RAD) + 1e-6)
        )
        self.assertTrue(np.all(np.asarray(states.info["action_delay_ticks"]) <= 2))
        self.assertTrue(np.all(np.asarray(states.info["action_delay_ticks"]) >= 0))
        self.assertTrue(np.all(np.asarray(states.info["next_push_step"]) >= 200))
        self.assertTrue(np.all(np.asarray(states.info["next_push_step"]) <= 400))

    def test_k16_training_wrapper_rollout_preserves_info_dtypes(self) -> None:
        from mujoco_playground import wrapper

        config = default_config()
        config.dr_enabled = True
        env = HexapodRoughTerrainEnv(config=config, terrain_level=0)
        rng = jax.random.split(jax.random.PRNGKey(5), 8)
        randomizer = functools.partial(
            randomize_batch, rng=rng, bank_size=16, seed=5
        )
        wrapped = wrapper.wrap_for_brax_training(
            env,
            episode_length=config.episode_length,
            randomization_fn=randomizer,
            full_reset=True,
        )
        reset = jax.jit(wrapped.reset)
        step = jax.jit(wrapped.step)
        state = reset(rng)
        action = jp.zeros((8, env.action_size))
        for _ in range(32):
            state = step(state, action)
        state.reward.block_until_ready()
        self.assertEqual(state.info["policy_steps"].dtype, jp.int32)
        self.assertEqual(state.info["next_push_step"].dtype, jp.int32)
        self.assertTrue(bool(jp.all(jp.isfinite(state.obs))))
        self.assertTrue(bool(jp.all(jp.isfinite(state.reward))))


if __name__ == "__main__":
    unittest.main()
