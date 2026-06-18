# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import torch
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg

from . import mdp

from lunarleaper_isaaclab.assets.robots.lunarleaper_delta import (  # isort: skip
    DELTA_QUADRUPED_MOTOR_PACE_CFG,
    LUNARLEAPER_ALU_DELTA_CFG,
)


##
# Physics backend — Kamino only (required for closed kinematic loops in the delta legs)
##


@configclass
class DeltaQuadrupedPacePhysicsCfg(PresetCfg):
    """Kamino-only physics preset for the delta-legged quadruped.

    Each leg's parallel-arm topology contains PhysicsSphericalJoint loop-closing
    constraints that PhysX cannot handle, so Kamino is required. The solver
    settings are kept identical to the training preset
    (``LunarleaperDeltaPhysicsCfg`` in ``delta_velocity_env_cfg``) so that the
    actuator parameters fitted here are valid for the dynamics seen at training
    time: ``unified`` collision pipeline, two solver substeps, default P-ADMM
    tolerances, and a 0.01 contact margin (set in :meth:`__post_init__`).
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            use_fk_solver=True,
            use_collision_detector=True,
            collision_detector_pipeline="unified",
        ),
        num_substeps=2,
        use_cuda_graph=True,
    )


##
# Scene
##


@configclass
class DeltaQuadrupedPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene with the aluminium-plate delta quadruped (the robot the chirp data
    was recorded on), motors replaced by PaceDCMotor for sys-id."""

    robot: ArticulationCfg = LUNARLEAPER_ALU_DELTA_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            # max_delay sizes the torque delay buffer (history = max_delay + 1); it
            # must be >= the delay upper bound (7) the optimiser samples, else
            # update_time_lags raises when CMA-ES proposes a large delay. The base
            # actuator cfg ships max_delay=4, so bump it here.
            "motors": DELTA_QUADRUPED_MOTOR_PACE_CFG.replace(max_delay=7),
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

# Leg-major joint order matching DELTA_QUADRUPED_MOTOR_PACE_CFG: LF, RF, LH, RH;
# each leg has three motors (T, L, R).
_JOINT_ORDER: list[str] = [
    "LF_T_motor",
    "LF_L_motor",
    "LF_R_motor",
    "RF_T_motor",
    "RF_L_motor",
    "RF_R_motor",
    "LH_T_motor",
    "LH_L_motor",
    "LH_R_motor",
    "RH_T_motor",
    "RH_L_motor",
    "RH_R_motor",
]
_N_JOINTS = len(_JOINT_ORDER)  # 12
_N_PER_LEG = 3

# One independent CMA-ES process per leg. Each group lists the indices (into
# joint_order) of the three motors belonging to that leg. With the root link
# fixed, the four legs are dynamically decoupled, so a single full-robot rollout
# evaluates all four legs and each leg is scored and evolved independently.
_JOINT_GROUPS: list[list[int]] = [
    [0, 1, 2],  # LF
    [3, 4, 5],  # RF
    [6, 7, 8],  # LH
    [9, 10, 11],  # RH
]


@configclass
class DeltaQuadrupedPaceCfg(PaceCfg):
    """PACE optimisation configuration for the delta-legged quadruped.

    Parameter layout (61 total):
        [0:12]   armature          [kg⋅m²]
        [12:24]  viscous damping   [Nm⋅s/rad]
        [24:36]  static friction   [Nm]
        [36:48]  dynamic friction  [Nm]
        [48:60]  encoder bias      [rad]
        [60]     action delay      [sim steps]

    The optimisation is split into four independent CMA-ES processes (one per
    leg) via :attr:`joint_groups`; see :class:`~pace_sim2real.CMAESOptimizer`.
    """

    robot_name: str = "delta_quadruped"
    data_dir: str = "delta_quadruped/chirp_data.pt"
    joint_order: list[str] = _JOINT_ORDER
    joint_groups: list[list[int]] = _JOINT_GROUPS
    bounds_params: torch.Tensor = torch.zeros((_N_JOINTS * 5 + 1, 2))

    def __post_init__(self):
        n = _N_JOINTS
        self.bounds_params[:n, 0] = 1e-3
        self.bounds_params[:n, 1] = 1e-0  # armature [1e-3, 1.0] kg⋅m²
        self.bounds_params[n : 2 * n, 1] = 0.1  # viscous damping [0, 0.1] Nm⋅s/rad
        self.bounds_params[2 * n : 3 * n, 1] = 5.0  # static friction [0, 5] Nm
        self.bounds_params[3 * n : 4 * n, 1] = 5.0  # dynamic friction [0, 5] Nm
        self.bounds_params[4 * n : 5 * n, 0] = -0.03
        self.bounds_params[4 * n : 5 * n, 1] = 0.03  # bias [-0.03, 0.03] rad
        self.bounds_params[5 * n, 1] = 7.0  # delay [0, 7] sim steps


@configclass
class DeltaQuadrupedActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_JOINT_ORDER,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )  # actions = absolute joint position targets, in leg-major [T, L, R] order


@configclass
class DeltaQuadrupedObservationsCfg:
    """Observation specifications for the MDP.

    The joint observations are restricted to the twelve motors. The default
    (all-joints) observation breaks for this robot because each leg's
    loop-closing spherical joints make ``joint_pos`` wider than
    ``default_joint_pos``; PACE only needs the actuated motors anyway.
    """

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_JOINT_ORDER, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=_JOINT_ORDER, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


##
# Environment configuration
##


@configclass
class DeltaQuadrupedPaceEnvCfg(PaceSim2realEnvCfg):
    """PACE Sim2Real environment for the 12-DOF delta-legged quadruped."""

    scene: DeltaQuadrupedPaceSceneCfg = DeltaQuadrupedPaceSceneCfg()
    sim2real: DeltaQuadrupedPaceCfg = DeltaQuadrupedPaceCfg()
    actions: DeltaQuadrupedActionsCfg = DeltaQuadrupedActionsCfg()
    observations: DeltaQuadrupedObservationsCfg = DeltaQuadrupedObservationsCfg()

    def __post_init__(self):
        # LUNARLEAPER_DELTA_CFG.spawn may have articulation_props=None; the base
        # class __post_init__ touches fix_root_link on it, so initialise first.
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            )
        # Fix the base in the air: this is the sys-id rig condition and, crucially,
        # makes the four legs dynamically independent so they can be fitted by four
        # separate CMA-ES processes within a single rollout.
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        super().__post_init__()

        # Replay the 200 Hz chirp data 1:1 with control steps (one data row per
        # env.step). The control decimation used at training time (4) is a
        # control-rate choice; the physics-solver fidelity below is what governs
        # the leg/loop-closure dynamics the fit observes, and that is matched.
        self.sim.dt = 0.005
        self.decimation = 1

        # Kamino-only physics, matched to the training preset
        # (LunarleaperDeltaPhysicsCfg in delta_velocity_env_cfg).
        self.sim.physics = DeltaQuadrupedPacePhysicsCfg()
        # A nonzero collision margin is required for stable contact on the
        # triangle-mesh terrain at training time; kept here so the solver setup
        # is identical to training.
        self.sim.physics.default.default_shape_cfg.margin = 0.01
