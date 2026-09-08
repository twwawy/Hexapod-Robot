"""Bounded all-contact stance recentering; no contact/IK limit bypass.

Translations are in the controller pre-posture frame. Rotations represent small
body attitude corrections through the inverse motion of the support feet.
The whole interpolated command path is checked, then executed with zero endpoint
velocity. This cannot recover a configuration whose initial IK is already invalid.
"""
import jax
import jax.numpy as jp
import firmware_mjx_controller as fw
from foothold_feasibility import support_margin

DURATION_S = 2.0
TRANSLATION_M = .02
ANGLE_RAD = .0523598776  # 3 degrees
# Fixed 13 alternatives, including the unchanged stance. One attempt per epoch.
DELTAS = jp.concatenate((jp.zeros((1, 6)), jp.eye(6), -jp.eye(6)), axis=0)
SCALE = jp.array((TRANSLATION_M, TRANSLATION_M, TRANSLATION_M,
                  ANGLE_RAD, ANGLE_RAD, 0.))


def reach_margin(body):
    local = fw._body_to_leg(body)
    reach = jp.sqrt((jp.linalg.norm(local[..., :2], axis=-1)-fw.LINK_1)**2 + local[..., 2]**2)
    return jp.minimum(fw.LINK_2+fw.LINK_3-fw.WORKSPACE_MARGIN-reach,
                      reach-abs(fw.LINK_2-fw.LINK_3)-fw.WORKSPACE_MARGIN)


def plan(start, height, posture):
    delta = DELTAS*SCALE
    targets = jax.vmap(lambda d: fw._rotate_inverse(start, d[3:])+d[:3])(delta)
    progress = fw._quintic(jp.linspace(0., 1., 21))
    samples = start[None, None, :, :] + progress[None, :, None, None]*(targets[:, None]-start)
    body = fw._rotate_inverse(samples.at[..., 2].add(-height), posture)
    _, valid = fw._solve_ik(body)
    _, limited = fw._limit_foot_reach(body)
    support = support_margin(body[..., :2], jp.ones(body.shape[:-1], dtype=jp.bool_), jp.zeros(2))
    safe = jp.all(valid & ~limited, axis=(1, 2)) & jp.all(support >= .02, axis=1)
    margins = jp.min(reach_margin(body[:, -1]), axis=-1)
    # No move unless the weakest reach margin actually improves by >= 1 mm.
    eligible = safe & (margins >= margins[0]+.001)
    chosen = jp.argmax(jp.where(eligible, margins, -jp.inf))
    return targets[chosen], jp.any(eligible), margins[chosen]
