"""Small user/agent-run controller tests, no PPO or terrain rollout."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax.numpy as jp
import numpy as np
import adaptive_foot_retry as retry
import adaptive_gait_controller as controller


class FootRetryTest(unittest.TestCase):
    def test_wall_is_not_support_and_order_is_symmetric(self):
        # 0 world; 1 and 2 foot geoms. Reversed pair normal reverses too.
        geom = jp.array(((0, 1), (2, 0)))
        frame = jp.broadcast_to(jp.eye(3), (2, 3, 3))
        support, wall = retry.contact_kinds(geom, jp.zeros(2), frame, jp.array((1, 2)), jp.array((0, 1, 2)))
        np.testing.assert_array_equal(support, [False, False])
        np.testing.assert_array_equal(wall, [True, True])
        frame = frame.at[0, 0].set(jp.array((0., 0., 1.))).at[1, 0].set(jp.array((0., 0., -1.)))
        support, wall = retry.contact_kinds(geom, jp.zeros(2), frame, jp.array((1, 2)), jp.array((0, 1, 2)))
        np.testing.assert_array_equal(support, [True, True])
        np.testing.assert_array_equal(wall, [False, False])

    def test_retract_before_forward_and_keep_endpoint(self):
        points = jp.array(((0., 0., .1), (-.04, 0., .115), (-.04, 0., .2), (.06, 0., .2), (.06, 0., .15)))
        np.testing.assert_allclose(retry.trajectory(retry.KNOTS, points), points, atol=1e-6)
        self.assertLess(float(retry.trajectory(jp.asarray(.1), points)[0]), 0.)
        self.assertAlmostEqual(float(retry.trajectory(jp.asarray(.3), points)[0]), -.04, places=6)
        self.assertGreater(float(retry.trajectory(jp.asarray(.6), points)[2]), .19)
        np.testing.assert_allclose(retry.trajectory(jp.asarray(1.), points), points[-1])

    def test_unknown_retraction_path_is_rejected_and_attempt_is_bounded(self):
        from types import SimpleNamespace
        s = controller.initial_state()
        body = s.foot_memory
        root = jp.array((0., 0., .032-body[0, 2]))
        feet = jp.stack((body[:, 1], -body[:, 0], body[:, 2]), axis=-1)+root
        contacts = jp.array((False, True, True, True, True, True))
        c = SimpleNamespace(geom=jp.array(((0, 1),)), dist=jp.array((-.001,)), frame=jp.eye(3)[None])
        data = SimpleNamespace(site_xpos=feet, qpos=root, xmat=jp.eye(3)[None],
            subtree_com=root[None], time=jp.asarray(0.), _impl=SimpleNamespace(contact=c))
        def unknown(grid, xy, now):
            z = jp.zeros(xy.shape[:-1])
            return z, jp.zeros_like(z, dtype=bool), z, z
        env = SimpleNamespace(_foot_site_ids=jp.arange(6), _root_id=0,
            _foot_geom_ids=jp.arange(1, 7), _geom_body_ids=jp.arange(7), dt=.02, _query=unknown)
        r = s.foot_retry._replace(last_world=feet, blocked_time=jp.full(6, .2))
        s = s._replace(foot_retry=r, scheduler=s.scheduler._replace(
            running=jp.asarray(True), elapsed=jp.asarray(.6), epoch=jp.asarray(3)))
        info = dict(confirmed_contacts=contacts, lidar_map=None)
        result = retry.prepare(env, data, info, s)
        self.assertTrue(bool(result.rejected))
        self.assertFalse(bool(result.requested))
        self.assertEqual(int(result.attempted_epoch[0]), 3)
        again = retry.prepare(env, data, info, s._replace(foot_retry=result))
        self.assertFalse(bool(again.requested))
        self.assertFalse(bool(again.rejected))  # no unbounded replanning
        # With the same geometry observed as flat, the short backoff must be usable.
        def flat(grid, xy, now):
            z = jp.zeros(xy.shape[:-1])
            return z, jp.ones_like(z, dtype=bool), z, z
        env._query = flat
        allowed = retry.prepare(env, data, info, s)
        self.assertTrue(bool(allowed.requested))
        self.assertLess(float(allowed.waypoints[1, 0]), float(allowed.waypoints[0, 0]))
        c.dist = jp.array((1.,))  # no spherical-foot wall contact
        pause = retry.prepare(env, data, info, s)
        self.assertFalse(bool(pause.requested))  # a stationary foot alone is not blocked
        stalled = s._replace(foot_memory=s.foot_memory.at[0, 0].add(.04),
            foot_retry=r._replace(blocked_time=jp.full(6, .3)))
        tracking_retry = retry.prepare(env, data, info, stalled)
        self.assertTrue(bool(tracking_retry.requested))

    def test_active_retry_moves_only_selected_foot_and_freezes_phase(self):
        s = controller.initial_state()
        start = s.foot_memory[0]
        points = jp.stack((start, start+jp.array((-.025, 0., .015)),
            start+jp.array((-.025, 0., .06)), start+jp.array((0., 0., .06)), start))
        r = s.foot_retry._replace(active=jp.asarray(True), leg=jp.asarray(0),
                                  elapsed=jp.asarray(.1), waypoints=points)
        contacts = jp.array((False, True, True, True, True, True))
        s = s._replace(foot_retry=r, confirmed_contacts=contacts, raw_contacts=contacts,
            scheduler=s.scheduler._replace(running=jp.asarray(True), mode=jp.asarray(1), elapsed=jp.asarray(.4)))
        updated, out = controller.step(s, target_velocity=jp.array((.04, 0.)),
            body_position_world=jp.zeros(3), attitude_rpy=jp.zeros(3), contacts=contacts,
            policy_action=jp.zeros(24))
        self.assertTrue(bool(updated.foot_retry.active))
        self.assertLess(float(updated.foot_memory[0, 0]), float(s.foot_memory[0, 0]))
        np.testing.assert_allclose(updated.foot_memory[1:], s.foot_memory[1:])
        np.testing.assert_allclose(updated.swing_end, s.swing_end)
        self.assertAlmostEqual(float(updated.scheduler.elapsed), .4, places=6)
        np.testing.assert_allclose(out.applied_twist, jp.zeros(4))

    def test_retry_contact_loss_faults_without_moving_another_foot(self):
        s = controller.initial_state()
        r = s.foot_retry._replace(active=jp.asarray(True), leg=jp.asarray(0),
                                  waypoints=jp.broadcast_to(s.foot_memory[0], (5, 3)))
        contacts = jp.array((False, False, True, True, True, True))
        s = s._replace(foot_retry=r, confirmed_contacts=contacts, raw_contacts=contacts,
                      scheduler=s.scheduler._replace(running=jp.asarray(True)))
        updated, _ = controller.step(s, target_velocity=jp.array((.04, 0.)),
            body_position_world=jp.zeros(3), attitude_rpy=jp.zeros(3), contacts=contacts,
            policy_action=jp.zeros(24))
        self.assertFalse(bool(updated.foot_retry.active))
        self.assertTrue(bool(updated.scheduler.fault))
        np.testing.assert_allclose(updated.foot_memory, s.foot_memory)


if __name__ == '__main__':
    unittest.main()
