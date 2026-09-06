"""User-run checks for hybrid safety boundaries; no MuJoCo rollout required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
import jax.numpy as jp
import numpy as np
import adaptive_gait_controller as controller
import wave_gait_scheduler as scheduler
import hybrid_gait_supervisor as supervisor
from adaptive_foothold_estimator import support_quality, PATCH_OFFSETS
from foothold_feasibility import support_margin


class HybridSafetyTest(unittest.TestCase):
    def test_partial_plane_requires_center_and_noncollinear_support(self):
        height = PATCH_OFFSETS[:, 0]*.1
        spread = jp.zeros(5)
        good = support_quality(height, jp.array([True, True, False, True, False]), spread)
        self.assertTrue(bool(good['coverage_ok']))
        self.assertFalse(bool(good['rough']))
        for known in ([True, True, True, False, False], [False, True, True, True, True]):
            self.assertFalse(bool(support_quality(height, jp.array(known), spread)['coverage_ok']))

    def test_observed_step_rejected_even_with_partial_coverage(self):
        quality = support_quality(jp.array([0., .08, 0., 0., 0.]),
                                  jp.array([True, True, False, True, False]), jp.zeros(5))
        self.assertTrue(bool(quality['rough']))

    def test_hull_rejects_com_outside_and_insufficient_support(self):
        feet = jp.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [0., 0.], [.1, 0.]])
        mask = jp.array([True, True, True, True, False, False])
        self.assertAlmostEqual(float(support_margin(feet, mask, jp.zeros(2))), 1., places=5)
        self.assertLess(float(support_margin(feet, mask, jp.array([2., 0.]))), 0.)
        self.assertLess(float(support_margin(feet, jp.arange(6) < 2, jp.zeros(2))), 0.)

    def test_unknown_holds_known_failure_allows_wave_short_has_priority(self):
        def choose(feasible, known):
            return supervisor.decide(supervisor.initial_supervisor(), tripod_feasible=jp.array(feasible),
                tripod_known_bad=jp.array(known), wave_feasible=jp.asarray(True),
                two_tripod_phases=jp.asarray(False), current_mode=jp.asarray(scheduler.TRIPOD),
                requested_scale=jp.asarray(1.), dt=.02)[0].decision
        self.assertEqual(int(choose([False]*6, [False]*6)), supervisor.HOLD)
        self.assertEqual(int(choose([False]*6, [True]*6)), supervisor.WAVE_MODE)
        self.assertEqual(int(choose([False, False, True, False, False, False], [True]*6)), supervisor.SHORT)

    def test_wave_return_requires_hysteresis(self):
        def choose(state):
            return supervisor.decide(state, tripod_feasible=jp.ones(6, dtype=bool),
                tripod_known_bad=jp.zeros(6, dtype=bool), wave_feasible=jp.asarray(True),
                two_tripod_phases=jp.asarray(True), current_mode=jp.asarray(scheduler.WAVE),
                requested_scale=jp.asarray(1.), dt=.02)[0].decision
        self.assertEqual(int(choose(supervisor.initial_supervisor())), supervisor.WAVE_MODE)
        self.assertEqual(int(choose(supervisor.SupervisorState(jp.asarray(.6), jp.asarray(2.), jp.asarray(2)))), supervisor.NORMAL)

    def test_wave_order_and_no_mid_swing_mode_switch(self):
        self.assertEqual([int(jp.argmax(scheduler.swing_mask(scheduler.WAVE, i))) for i in range(6)],
                         [0, 5, 1, 3, 2, 4])
        state = scheduler.initial_scheduler()._replace(running=jp.asarray(True), elapsed=jp.asarray(.2))
        updated, _ = scheduler.advance(state, requested_mode=jp.asarray(scheduler.WAVE), permit=jp.asarray(True),
            proposal_epoch=state.epoch, command_active=jp.asarray(True), contacts=jp.ones(6, dtype=bool),
            raw_contacts=jp.ones(6, dtype=bool), duration=.5, dt=.005)
        self.assertEqual(int(updated.mode), scheduler.TRIPOD)

    def test_missing_contact_and_stale_proposal_prevent_launch(self):
        state = scheduler.initial_scheduler()._replace(start_wait=jp.asarray(.2))
        for contacts, epoch in ((jp.arange(6) != 0, state.epoch), (jp.ones(6, dtype=bool), state.epoch-1)):
            updated, gait = scheduler.advance(state, requested_mode=jp.asarray(scheduler.WAVE), permit=jp.asarray(True),
                proposal_epoch=epoch, command_active=jp.asarray(True), contacts=contacts,
                raw_contacts=contacts, duration=.5, dt=.005)
            self.assertFalse(bool(updated.running))
            self.assertFalse(bool(jp.any(gait['entering'])))

    def test_late_search_exhaustion_latches_hold(self):
        state = scheduler.initial_scheduler()._replace(running=jp.asarray(True), elapsed=jp.asarray(.5),
            airborne=jp.ones(6, dtype=bool), late_distance=jp.full(6, .0999))
        contacts = ~scheduler.swing_mask(scheduler.TRIPOD, 0)
        updated, gait = scheduler.advance(state, requested_mode=jp.asarray(0), permit=jp.asarray(True),
            proposal_epoch=state.epoch, command_active=jp.asarray(True), contacts=contacts,
            raw_contacts=contacts, duration=.5, dt=.005)
        self.assertTrue(bool(updated.fault))
        self.assertTrue(bool(jp.all(gait['state'] == scheduler.HOLD)))

    def test_trajectory_endpoints_and_latched_parameters(self):
        initial = controller.initial_state()
        self.assertEqual(initial.request.shape, (24,))
        start = initial.foot_memory
        end = start.at[:, 0].add(.03).at[:, 2].add(.04)
        fn = jax.jit(controller.planned_swing)
        np.testing.assert_allclose(fn(0., start, end, .08, .4, .6), start, atol=1e-6)
        np.testing.assert_allclose(fn(1., start, end, .08, .4, .6), end, atol=1e-6)
        np.testing.assert_allclose(fn(.4, start, end, .08, .4, .6)[:, 2], end[:, 2]+.08, atol=1e-6)
        gait = dict(entering=jp.array([True, False, False, False, False, False]),
                    state=jp.array([scheduler.SWING]+[scheduler.HOLD]*5), progress=jp.zeros(6), startup=False)
        prepared = initial._replace(proposal_end=end, proposal_clearance=jp.full(6, .08))
        updates, _ = controller._foot_trajectory(prepared, gait, jp.zeros(4), True)
        latched = prepared._replace(**updates)._replace(proposal_end=end+1.,
            proposal_clearance=jp.full(6,.18),proposal_apex_phase=jp.full(6,.7),
            proposal_transfer=jp.full(6,.65),proposal_mode=jp.asarray(scheduler.WAVE))
        gait['entering'] = jp.zeros(6, dtype=bool)
        again, _ = controller._foot_trajectory(latched, gait, jp.zeros(4), True)
        np.testing.assert_allclose(again['swing_end'][0], end[0])
        self.assertAlmostEqual(float(again['swing_clearance'][0]),.08,places=6)
        self.assertAlmostEqual(float(again['apex_phase'][0]),.5,places=6)
        self.assertAlmostEqual(float(again['transfer'][0]),.5,places=6)


if __name__ == '__main__':
    unittest.main()
