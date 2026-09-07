"""User-run unit checks; no environment rollout or PPO. Not executed by agent."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
from adaptive_gait_perception import fuse_surface
from adaptive_grid import local_grid, GRID_SIDE, GRID_CHANNELS
import hybrid_gait_supervisor as supervisor


class ElevationGridTest(unittest.TestCase):
    def test_old_outlier_clears_but_repeated_vertical_spread_does_not(self):
        height, stamp, spread = jp.asarray(.04), jp.asarray(0.), jp.asarray(.04)
        for tick in range(1, 101):
            height, stamp, spread = fuse_surface(height, stamp, spread, jp.asarray(.003), jp.asarray(-.003), tick*.1)
        self.assertLess(float(spread), .01)
        self.assertAlmostEqual(float(height), .003, places=6)
        for tick in range(101, 121):
            height, stamp, spread = fuse_surface(height, stamp, spread, jp.asarray(.08), jp.asarray(0.), tick*.1)
        self.assertGreaterEqual(float(spread), .079)

    def test_unseen_cells_do_not_refresh(self):
        values = fuse_surface(jp.asarray(.08), jp.asarray(1.), jp.asarray(.08),
                              jp.asarray(-jp.inf), jp.asarray(jp.inf), 20.)
        np.testing.assert_allclose(np.asarray(values), [.08, 1., .08])

    def test_unknown_is_masked_and_yaw_rotates_slope(self):
        def query(xy, now):
            height = .1*xy[..., 0]
            known = xy[..., 0] >= 0.
            return height, known, jp.zeros_like(height), jp.zeros_like(height)
        grid = local_grid(query, jp.array((0., 0., .3)), jp.array((0., 1., 0.)), 1.)
        self.assertEqual(grid.shape, (GRID_SIDE, GRID_SIDE, GRID_CHANNELS))
        unknown = np.asarray(grid[..., 1]) == 0
        self.assertTrue(np.all(np.asarray(grid[..., 0])[unknown] == 0))
        self.assertTrue(np.all(np.asarray(grid[..., 2])[unknown] == 0))
        self.assertTrue(np.all(np.asarray(grid[..., 3])[unknown] == 1))
        self.assertAlmostEqual(float(grid[10, 5, 5]), -.1, places=5)

    def test_stride_preference_cannot_hide_only_safe_step(self):
        feasible = jp.array((False, True, False, False, False, False))
        index, available = supervisor.stride_choice(feasible, .7)
        self.assertTrue(bool(available))
        self.assertEqual(int(index), 1)
        decision, _ = supervisor.decide(supervisor.initial_supervisor(), tripod_feasible=feasible,
            tripod_known_bad=jp.zeros(6, dtype=bool), wave_feasible=True, two_tripod_phases=False,
            current_mode=0, requested_scale=.7, dt=.02)
        self.assertEqual(int(decision.decision), supervisor.NORMAL)
        _, available = supervisor.stride_choice(jp.zeros(6, dtype=bool), 1.)
        self.assertFalse(bool(available))

    def test_cnn_uses_grid_and_preserves_batch_shapes(self):
        from adaptive_grid_network import make_grid_networks
        from adaptive_gait_env import ACTOR_SIZE, CRITIC_SIZE, VECTOR_SIZE
        net = make_grid_networks({'state': ACTOR_SIZE, 'privileged_state': CRITIC_SIZE}, 24)
        params = net.policy_network.init(jax.random.PRNGKey(0))
        x = jp.zeros((2, 3, ACTOR_SIZE))
        y = net.policy_network.apply(None, params, {'state': x})
        self.assertEqual(y.shape, (2, 3, 48))
        gradient = jax.grad(lambda obs: jp.sum(net.policy_network.apply(None, params, {'state': obs})))(x)
        self.assertGreater(float(jp.linalg.norm(gradient[..., VECTOR_SIZE:])), 0.)


if __name__ == '__main__':
    unittest.main()
