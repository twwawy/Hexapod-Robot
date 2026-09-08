"""User-run pure geometry checks; no simulation or PPO."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
from adaptive_foothold_estimator import path_bottleneck, select_path_variant, FOOT_RADIUS
from adaptive_gait_controller import planned_swing


class PathBottleneckTest(unittest.TestCase):
    def test_unknown_not_invented_as_clearance(self):
        fractions = jp.linspace(0., 1., 21)
        world = jp.zeros((6, 25, 21, 3))
        height = jp.zeros((6, 25, 21, 5))
        margin, phase, known, _ = jax.jit(path_bottleneck)(world, height, jp.zeros_like(height, dtype=bool), fractions)
        self.assertFalse(bool(jp.any(known)))
        self.assertTrue(bool(jp.all(jp.isfinite(margin))))
        self.assertTrue(bool(jp.all(phase == 0.)))

    def test_observed_deficit_and_location(self):
        f = jp.linspace(0., 1., 21)
        world = jp.zeros((21, 3)).at[:, 2].set(FOOT_RADIUS+.10)
        height = jp.zeros((21, 5)).at[8, :].set(.12)
        margin, phase, known, index = path_bottleneck(world, height, jp.ones_like(height, dtype=bool), f)
        self.assertTrue(bool(known))
        self.assertAlmostEqual(float(margin), -.02, places=5)
        self.assertAlmostEqual(float(phase), .4, places=5)
        self.assertEqual(int(index), 8)

    def test_preserve_safe_request_and_reject_no_solution(self):
        feasible = jp.array([[True, False, False], [True, True, False],
                             [False, True, False], [False, True, False]])
        cost = jp.array([[9., 0., 0.], [0., .2, .2], [1., .5, .5], [2., .7, .7]])
        np.testing.assert_array_equal(jax.jit(select_path_variant)(feasible, cost), [0, 1, 0])

    def test_timing_repair_can_clear_riser_without_extra_height(self):
        f = jp.linspace(0., 1., 21)
        start, end = jp.array((0., 0., FOOT_RADIUS)), jp.array((.2, 0., FOOT_RADIUS+.05))
        def margin(apex, transfer):
            points = jax.vmap(lambda t: planned_swing(t, start, end, .06, apex, transfer))(f)
            # Close riser encountered early by a deliberately late-lift request.
            heights = jp.where(points[:, 0] >= .008, .05, 0.)[:, None]
            return path_bottleneck(points, heights, jp.ones_like(heights, dtype=bool), f)[0]
        self.assertGreater(float(margin(.6, .45)), float(margin(.7, .35)))
        for apex, transfer in ((.7, .35), (.6, .45)):
            np.testing.assert_allclose(planned_swing(0., start, end, .06, apex, transfer), start, atol=1e-6)
            np.testing.assert_allclose(planned_swing(1., start, end, .06, apex, transfer), end, atol=1e-6)


if __name__ == '__main__':
    unittest.main()
