"""User-run RC command unit checks. No environment rollout or training."""
import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
from operator_commands import next_command, motion_score


class OperatorCommandsTest(unittest.TestCase):
    def test_signed_reverse_turn_and_stop(self):
        for vx, wz in ((.06, 0.), (-.06, 0.), (0., .2), (0., -.2), (-.06, -.2)):
            speed, moving, good, progress = motion_score(vx, wz, jp.array((vx, wz)))
            self.assertTrue(bool(moving & good))
            self.assertGreater(float(speed), 0.)
            self.assertAlmostEqual(float(progress), 1., places=5)
            _, _, wrong, _ = motion_score(-vx, -wz, jp.array((vx, wz)))
            self.assertFalse(bool(wrong))
        _, moving, good, progress = motion_score(0., 0., jp.zeros(2))
        self.assertFalse(bool(moving))
        self.assertTrue(bool(good))
        self.assertEqual(float(progress), 0.)

    def test_standing_does_not_pass_low_speed_commands(self):
        for command in ((.03, 0.), (-.03, 0.), (0., .12), (0., -.12)):
            self.assertFalse(bool(motion_score(0., 0., jp.array(command))[2]))

    def test_slew_leaves_deadband_and_targets_are_held(self):
        slew = jp.zeros(2)
        target = jp.array((.08, -.25))
        remaining = jp.asarray(4.)
        key = jax.random.PRNGKey(3)
        for _ in range(10):
            applied, new_slew, new_target, remaining, key = next_command(slew, target, remaining, key, .02)
            self.assertTrue(bool(jp.all(jp.abs(new_slew-slew) <= jp.array((.08, .4))*.02+1e-6)))
            np.testing.assert_allclose(new_target, target)
            slew = new_slew
        self.assertGreater(float(applied[0]), 0.)
        self.assertLess(float(applied[1]), 0.)

    def test_static_batch_sampler(self):
        keys = jax.random.split(jax.random.PRNGKey(4), 32)
        result = jax.vmap(lambda key: next_command(jp.zeros(2), jp.zeros(2), jp.asarray(0.), key, .02))(keys)
        self.assertEqual(result[0].shape, (32, 2))
        self.assertTrue(bool(jp.all(jp.abs(result[2][:, 0]) <= .08)))
        self.assertTrue(bool(jp.all(jp.abs(result[2][:, 1]) <= .25)))


if __name__ == '__main__':
    unittest.main()
