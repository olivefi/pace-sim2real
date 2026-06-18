# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import torch
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg

from lunarleaper_isaaclab.assets.robots.delta_robot import DELTA_ALLREV_CFG, DELTA_MOTOR_PACE_CFG  # isort: skip
from . import mdp


##
# Physics backend — Kamino only (required for closed kinematic loops in the delta arm)
##


@configclass
class DeltaPacePhysicsCfg(PresetCfg):
    """Kamino-only physics preset for the delta robot.

    The delta robot's parallel-arm topology contains PhysicsSphericalJoint
    loop-closing constraints that PhysX cannot handle; Kamino is required.

    The ``default`` preset deliberately mirrors the solver settings used in
    ``replay_pace_traj.py`` so that rollout and replay are comparable:
    Kamino's internal collision detector with the "primitive" pipeline, looser
    PADMM tolerances (1e-4) and smaller rho_0 (0.1) that converge reliably
    within the 200-iteration budget, and "geom_pair_net_force" warm-start.
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            use_fk_solver=True,
            use_collision_detector=True,
            collision_detector_pipeline="primitive",
            collision_detector_max_contacts_per_pair=32,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.1,
            padmm_contact_warmstart_method="geom_pair_net_force",
        ),
        num_substeps=1,
        use_cuda_graph=True,
        default_shape_cfg=NewtonShapeCfg(gap=0.0),
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(use_fk_solver=False),
        num_substeps=1,
        use_cuda_graph=True,
    )


##
# Scene
##


@configclass
class DeltaPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene with the delta robot, motors replaced by PaceDCMotor for sys-id."""
    robot: ArticulationCfg = DELTA_ALLREV_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            "motors": DELTA_MOTOR_PACE_CFG,
            "bearings": ImplicitActuatorCfg(
                joint_names_expr=[".*bearing.*"],
                effort_limit_sim=1000.0,
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )


##
# PACE configuration
##

_N_JOINTS = 3  # motor1, motor2, motor3


@configclass
class DeltaPaceCfg(PaceCfg):
    """PACE optimisation configuration for the delta robot.

    Parameter layout (16 total):
        [0:3]   armature          [kg⋅m²]
        [3:6]   viscous damping   [Nm⋅s/rad]
        [6:9]   static friction   [Nm]
        [9:12]  dynamic friction  [Nm]
        [12:15] encoder bias      [rad]
        [15]    action delay      [sim steps]
    """

    robot_name: str = "delta_robot"
    data_dir: str = "delta_robot/chirp_data.pt"
    joint_order: list[str] = ["T_motor", "L_motor", "R_motor"]
    # extra_joint_order: list[str] = [".*bearing.*"]
    bounds_params: torch.Tensor = torch.zeros((_N_JOINTS * 5 + 1, 2))
    # extra_bounds_params: torch.Tensor = torch.tensor([
    #     [1e-7, 1e-4],  # armature [1e-5, 0.01] kg⋅m²
    #     [0.0, 2.0],    # friction [0, 2]
    # ])

    def __post_init__(self):
        n = _N_JOINTS
        self.bounds_params[:n, 0] = 1e-5
        self.bounds_params[:n, 1] = 1e-1        # armature kg⋅m²
        self.bounds_params[n : 2 * n, 1] = 2.0  # viscous damping [0, 2] Nm⋅s/rad
        self.bounds_params[2 * n : 3 * n, 1] = 2.0  # static friction [0, 2] Nm
        self.bounds_params[3 * n : 4 * n, 1] = 2.0  # dynamic friction [0, 2] Nm
        self.bounds_params[4 * n : 5 * n, 0] = -0.1
        self.bounds_params[4 * n : 5 * n, 1] = 0.1  # bias [-0.1, 0.1] rad
        self.bounds_params[5 * n, 1] = 2.0     # delay [0, 1] sim steps

@configclass
class DeltaActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=["T_motor", "L_motor", "R_motor"],
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )  # actions = absolute joint position targets, in [T, L, R] order


##
# Environment configuration
##


@configclass
class DeltaPaceEnvCfg(PaceSim2realEnvCfg):
    """PACE Sim2Real environment for the 3-DOF delta parallel robot."""

    scene: DeltaPaceSceneCfg = DeltaPaceSceneCfg()
    sim2real: DeltaPaceCfg = DeltaPaceCfg()
    actions: DeltaActionsCfg = DeltaActionsCfg()

    def __post_init__(self):
        # DELTA_ALLREV_CFG.spawn has articulation_props=None; the base class
        # __post_init__ unconditionally sets fix_root_link on it, so initialise first.
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            )
        super().__post_init__()

        self.sim.dt = 0.005
        self.decimation = 1

        # Kamino-only physics (required for closed-loop kinematics).
        self.sim.physics = DeltaPacePhysicsCfg()

