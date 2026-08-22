"""Per-environment MJX dynamics randomization with fixed safety limits."""

from __future__ import annotations

import jax
from mujoco import mjx


def domain_randomize(
    model: mjx.Model,
    rng: jax.Array,
    *,
    friction_range: tuple[float, float] = (0.6, 1.3),
    mass_range: tuple[float, float] = (0.9, 1.1),
    actuator_range: tuple[float, float] = (0.85, 1.0),
    damping_range: tuple[float, float] = (0.8, 1.2),
):
    """Create one randomized dynamics model per vectorized environment.

    Actuator gain/bias realization is varied, while ``actuator_forcerange`` is
    intentionally absent from the replacement tree.  The ±8 Nm safety clamp
    from the source model therefore remains identical in every environment.
    """

    @jax.vmap
    def randomize_one(key):
        key, friction_key, mass_key, actuator_key, damping_key = jax.random.split(
            key, 5
        )
        friction_scale = jax.random.uniform(
            friction_key, (), minval=friction_range[0], maxval=friction_range[1]
        )
        geom_friction = model.geom_friction.at[:, 0].set(
            model.geom_friction[:, 0] * friction_scale
        )

        mass_scale = jax.random.uniform(
            mass_key, (), minval=mass_range[0], maxval=mass_range[1]
        )
        body_mass = model.body_mass * mass_scale
        body_inertia = model.body_inertia * mass_scale

        actuator_scale = jax.random.uniform(
            actuator_key,
            (),
            minval=actuator_range[0],
            maxval=actuator_range[1],
        )
        actuator_gainprm = model.actuator_gainprm.at[:, 0].set(
            model.actuator_gainprm[:, 0] * actuator_scale
        )
        actuator_biasprm = model.actuator_biasprm.at[:, 1:3].set(
            model.actuator_biasprm[:, 1:3] * actuator_scale
        )

        damping_scale = jax.random.uniform(
            damping_key, (), minval=damping_range[0], maxval=damping_range[1]
        )
        dof_damping = model.dof_damping * damping_scale
        return (
            geom_friction,
            body_mass,
            body_inertia,
            actuator_gainprm,
            actuator_biasprm,
            dof_damping,
        )

    (
        geom_friction,
        body_mass,
        body_inertia,
        actuator_gainprm,
        actuator_biasprm,
        dof_damping,
    ) = randomize_one(rng)
    in_axes = jax.tree_util.tree_map(lambda _: None, model)
    in_axes = in_axes.tree_replace(
        {
            "geom_friction": 0,
            "body_mass": 0,
            "body_inertia": 0,
            "actuator_gainprm": 0,
            "actuator_biasprm": 0,
            "dof_damping": 0,
        }
    )
    randomized = model.tree_replace(
        {
            "geom_friction": geom_friction,
            "body_mass": body_mass,
            "body_inertia": body_inertia,
            "actuator_gainprm": actuator_gainprm,
            "actuator_biasprm": actuator_biasprm,
            "dof_damping": dof_damping,
        }
    )
    return randomized, in_axes
