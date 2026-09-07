"""Fixed-size, yaw-aligned elevation observation; no unknown-to-flat inference."""
import jax.numpy as jp

GRID_SIDE = 24
GRID_RESOLUTION = .05
GRID_CHANNELS = 6
GRID_SIZE = GRID_SIDE*GRID_SIDE*GRID_CHANNELS
GRID_CHANNEL_NAMES = ('relative_height', 'known', 'confidence', 'age', 'forward_slope', 'left_slope')


def local_grid(query, origin, forward, now):
    """1.2 m square centred on body, forward/left axes. query owns perception.

    Heights are relative to body Z. Unknown height/slope is masked and carries
    explicit known=0, confidence=0, age=1, never a fabricated flat observation.
    """
    heading = forward[:2]/jp.maximum(jp.linalg.norm(forward[:2]), 1e-6)
    left = jp.array((-heading[1], heading[0]))
    axis = (jp.arange(GRID_SIDE)-(GRID_SIDE-1)/2)*GRID_RESOLUTION
    x, y = jp.meshgrid(axis, axis, indexing='ij')
    xy = origin[:2]+x[..., None]*heading+y[..., None]*left
    height, known, age, spread = query(xy, now)
    height = jp.where(known, height, origin[2])
    slope_x = jp.where(known[1:] & known[:-1], (height[1:]-height[:-1])/GRID_RESOLUTION, 0.)
    slope_y = jp.where(known[:, 1:] & known[:, :-1], (height[:, 1:]-height[:, :-1])/GRID_RESOLUTION, 0.)
    slope_x = jp.pad(slope_x, ((0, 1), (0, 0)))
    slope_y = jp.pad(slope_y, ((0, 0), (0, 1)))
    confidence = jp.where(known, jp.exp(-jp.maximum(age, 0.)/30.)*jp.exp(-spread/.025), 0.)
    return jp.stack((jp.clip((height-origin[2])/.3, -5., 5.), known.astype(jp.float32),
        confidence, jp.where(known, jp.clip(age/60., 0., 1.), 1.),
        jp.clip(slope_x, -2., 2.), jp.clip(slope_y, -2., 2.)), axis=-1)
