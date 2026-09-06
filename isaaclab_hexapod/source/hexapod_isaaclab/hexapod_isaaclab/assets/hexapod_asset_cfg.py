"""Asset configuration derived from the versioned MJX asset manifest."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from .joint_contract import HOME_ROOT_POS, HOME_ROOT_QUAT_WXYZ, JOINT_ORDER


REPO_ROOT = Path(__file__).resolve().parents[5]
USD_PATH = (
    REPO_ROOT
    / "isaaclab_hexapod/data/usd/hexapod_full_mesh_mjx_parity.usd"
)
HEXAPOD_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(USD_PATH),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=HOME_ROOT_POS,
        rot=HOME_ROOT_QUAT_WXYZ,
        joint_pos={
            "R[BMF]_1": 0.0,
            "R[BMF]_2": -0.5235987755982988,
            "R[BMF]_3": 0.8726646259971648,
            "L[BMF]_1": 0.0,
            "L[BMF]_2": 0.5235987755982988,
            "L[BMF]_3": -0.8726646259971648,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=list(JOINT_ORDER),
            stiffness=500.0,
            damping=10.0,
            effort_limit_sim=14.709975,
            armature=0.02,
            friction=0.8,
            dynamic_friction=0.8,
            viscous_friction=0.15,
        )
    },
    soft_joint_pos_limit_factor=1.0,
)
