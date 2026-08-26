"""Asset configuration derived from the versioned MJX asset manifest."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg


REPO_ROOT = Path(__file__).resolve().parents[5]
USD_PATH = REPO_ROOT / "isaaclab_hexapod/data/usd/hexapod_mjx_parity.usd"
JOINT_ORDER = [
    "RB_1", "RB_2", "RB_3",
    "RM_1", "RM_2", "RM_3",
    "RF_1", "RF_2", "RF_3",
    "LB_1", "LB_2", "LB_3",
    "LM_1", "LM_2", "LM_3",
    "LF_1", "LF_2", "LF_3",
]


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
        pos=(0.0, 0.0, 0.287006),
        rot=(0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
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
            joint_names_expr=JOINT_ORDER,
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
