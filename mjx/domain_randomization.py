"""Finite MJX model-bank randomization for robust servo stair training.

The returned stacked model and ``in_axes`` follow Brax's
``DomainRandomizationVmapWrapper`` convention: only randomized leaves carry a
leading bank/environment axis, while every other model leaf is shared.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jp
import numpy as np


FRICTION_RANGE = (0.4, 1.25)
MASS_RANGE = (0.8, 1.2)
KP_RANGE = (0.9, 1.1)
ARMATURE_RANGE = (0.8, 1.2)
DAMPING_RANGE = (0.8, 1.2)
RANDOMIZED_FIELDS = (
    "geom_friction",
    "body_mass",
    "body_inertia",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_armature",
    "dof_damping",
)


def _geom_name(model: Any, geom_id: int) -> str:
    address = int(np.asarray(model.name_geomadr)[geom_id])
    end = model.names.find(b"\x00", address)
    return model.names[address:end].decode("utf-8")


def _foot_geom_ids(model: Any) -> jp.ndarray:
    ids = [
        geom_id
        for geom_id in range(int(model.ngeom))
        if _geom_name(model, geom_id).endswith("_foot_collision")
    ]
    if len(ids) != 6:
        raise ValueError(f"expected six foot collision geoms, found {len(ids)}")
    return jp.asarray(ids, dtype=jp.int32)


def _validate_base_model(model: Any) -> None:
    for field in RANDOMIZED_FIELDS:
        value = np.asarray(getattr(model, field))
        if not np.all(np.isfinite(value)):
            raise ValueError(f"base model field '{field}' contains NaN/Inf")
    foot_friction = np.asarray(model.geom_friction)[np.asarray(_foot_geom_ids(model))]
    if np.any(foot_friction <= 0.0):
        raise ValueError("base model foot friction must be finite and positive")


def _uniform_factors(key: jax.Array, size: int, bounds: tuple[float, float]) -> jax.Array:
    return jax.random.uniform(key, (size,), minval=bounds[0], maxval=bounds[1])


def build_bank(model: Any, bank_size: int, seed: int) -> tuple[Any, Any]:
    """Build K deterministic plant variants and their Brax vmap ``in_axes``.

    ``bank_size == 1`` is an exact, non-randomized copy for the baseline fast
    path.  Mass and rotational inertia use the same multiplier so each plant
    remains physically consistent.
    """
    if bank_size < 1:
        raise ValueError("bank_size must be positive")
    _validate_base_model(model)
    foot_ids = _foot_geom_ids(model)

    if bank_size == 1:
        friction_factor = mass_factor = kp_factor = armature_factor = damping_factor = jp.ones((1,))
    else:
        keys = jax.random.split(jax.random.PRNGKey(seed), 5)
        friction_factor = _uniform_factors(keys[0], bank_size, FRICTION_RANGE)
        mass_factor = _uniform_factors(keys[1], bank_size, MASS_RANGE)
        kp_factor = _uniform_factors(keys[2], bank_size, KP_RANGE)
        armature_factor = _uniform_factors(keys[3], bank_size, ARMATURE_RANGE)
        damping_factor = _uniform_factors(keys[4], bank_size, DAMPING_RANGE)

    geom_friction = jp.broadcast_to(
        model.geom_friction, (bank_size,) + model.geom_friction.shape
    )
    randomized_feet = (
        model.geom_friction[foot_ids][None, :, :]
        * friction_factor[:, None, None]
    )
    geom_friction = geom_friction.at[:, foot_ids, :].set(randomized_feet)

    body_mass = model.body_mass[None, :] * mass_factor[:, None]
    body_inertia = model.body_inertia[None, :, :] * mass_factor[:, None, None]
    actuator_gainprm = jp.broadcast_to(
        model.actuator_gainprm,
        (bank_size,) + model.actuator_gainprm.shape,
    )
    actuator_biasprm = jp.broadcast_to(
        model.actuator_biasprm,
        (bank_size,) + model.actuator_biasprm.shape,
    )
    actuator_gainprm = actuator_gainprm.at[:, :, 0].set(
        model.actuator_gainprm[None, :, 0] * kp_factor[:, None]
    )
    actuator_biasprm = actuator_biasprm.at[:, :, 1].set(
        model.actuator_biasprm[None, :, 1] * kp_factor[:, None]
    )
    dof_armature = model.dof_armature[None, :] * armature_factor[:, None]
    dof_damping = model.dof_damping[None, :] * damping_factor[:, None]

    replacements = {
        "geom_friction": geom_friction,
        "body_mass": body_mass,
        "body_inertia": body_inertia,
        "actuator_gainprm": actuator_gainprm,
        "actuator_biasprm": actuator_biasprm,
        "dof_armature": dof_armature,
        "dof_damping": dof_damping,
    }
    bank = model.tree_replace(replacements)
    in_axes = jax.tree.map(lambda _: None, model).tree_replace(
        {field: 0 for field in RANDOMIZED_FIELDS}
    )
    return bank, in_axes


def randomize_batch(
    model: Any,
    rng: jax.Array,
    *,
    bank_size: int,
    seed: int,
) -> tuple[Any, Any]:
    """Assign environment ``i`` to deterministic model ``bank[i % K]``."""
    bank, in_axes = build_bank(model, bank_size, seed)
    indices = jp.arange(rng.shape[0], dtype=jp.int32) % bank_size
    replacements = {
        field: getattr(bank, field)[indices] for field in RANDOMIZED_FIELDS
    }
    return bank.tree_replace(replacements), in_axes
