# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""PACE env for fitting a *single* delta leg against one leg of the quadruped HW data.

The chirp data is recorded on the full aluminium-plate quadruped (twelve motors).
Instead of replaying all four legs on the 12-DOF quadruped, this env replays a
single leg's three motor columns on the standalone single-leg aluminium robot
(:data:`DELTA_ALLREV_ALU_CFG`). The single-leg leg sub-assembly is geometrically
identical to one leg of the aluminium-plate quadruped, so a fit obtained here is
valid for that leg on the quadruped.

Splitting the recording into four independent single-leg fits (one per leg) lets
each fit run as its own process on the cheap 3-DOF / single-loop-closure sim,
which converges far faster than the 12-DOF quadruped rollout and parallelises
trivially across processes/GPUs. See
``scripts/pace/prepare_delta_quadruped_leg_data.py`` for the data split and
``scripts/pace/fit_quadruped_legs.sh`` for the four-way launcher.

The leg a given run fits is selected purely by which per-leg ``.pt`` the optimiser
loads (``--data_dir``); the env itself is leg-agnostic.
"""

import torch
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

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

from lunarleaper_isaaclab.assets.robots.delta_robot import (  # isort: skip
    DELTA_ALLREV_ALU_CFG,
    DELTA_MOTOR_PACE_CFG,
)


##
# Physics backend
##


@configclass
class DeltaQuadrupedLegPacePhysicsCfg(PresetCfg):
    """MuJoCo-Warp physics preset, matched to the quadruped PACE/training preset.

    Identical solver settings to ``DeltaQuadrupedPacePhysicsCfg`` so the dynamics
    a single leg sees here match those it sees as part of the quadruped at training
    time.
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            tolerance=1e-8,
            iterations=200,
            integrator="implicitfast",
        ),
        num_substeps=8,
        use_cuda_graph=True,
    )


##
# Scene
##

# Default stance — the same loop-closure-consistent crouched pose used per-leg on
# the quadruped (``_DEFAULT_JOINT_POS`` in ``lunarleaper_delta``), with the
# ``single_leg_`` prefix. Spawning at arbitrary motor angles tears the spherical
# loop closures open; this stance keeps both closures satisfied at spawn.
_DEFAULT_JOINT_POS: dict[str, float] = {
    "single_leg_T_motor": 0.0,
    "single_leg_joint_TU_bearing1": 0.0,
    "single_leg_joint_TU_bearing2": -0.698132,
    "single_leg_L_motor": -0.652897,
    "single_leg_joint_LU_bearing1": 0.068362,
    "single_leg_joint_LU_bearing2": 0.820272,
    "single_leg_R_motor": 0.652897,
    "single_leg_joint_RU_bearing1": -0.068362,
    "single_leg_joint_RU_bearing2": -0.820272,
}


@configclass
class DeltaQuadrupedLegPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene with the single-leg aluminium robot, motors replaced by PaceDCMotor."""

    robot: ArticulationCfg = DELTA_ALLREV_ALU_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=DELTA_ALLREV_ALU_CFG.init_state.replace(joint_pos=_DEFAULT_JOINT_POS),
        actuators={
            # The HW chirp was recorded on the quadruped with PD gains stiffness=20,
            # damping=1 (DELTA_QUADRUPED_MOTOR_PACE_CFG). The PD gains are part of the
            # control law, so the replay must use the same gains as the recording —
            # the single-leg DELTA_MOTOR_PACE_CFG default (10 / 0.6) would make the
            # closed-loop response un-matchable regardless of the fitted physical
            # params. max_delay sizes the torque delay buffer (history = max_delay + 1)
            # and must be >= the delay upper bound (7) the optimiser samples.
            # apply_coulomb_friction=False: MuJoCo-Warp applies Coulomb (dry) friction
            # natively via dof_frictionloss (PACE optimiser's friction block, slot 1).
            # apply_viscous_friction=True: velocity-proportional friction applied by the
            # actuator using static_friction coefficient (PACE optimiser's slot 2).
            "motors": DELTA_MOTOR_PACE_CFG.replace(
                max_delay=7,
                stiffness=20.0,
                damping=1.0,
                apply_coulomb_friction=False,
                apply_viscous_friction=True,
            ),
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

# Single-leg motor order, matching DELTA_MOTOR_PACE_CFG (T, L, R). The per-leg
# data files produced by prepare_delta_quadruped_leg_data.py store their three
# columns in this same order.
_JOINT_ORDER: list[str] = ["single_leg_T_motor", "single_leg_L_motor", "single_leg_R_motor"]
_N_JOINTS = len(_JOINT_ORDER)  # 3


@configclass
class DeltaQuadrupedLegPaceCfg(PaceCfg):
    """PACE optimisation configuration for a single delta leg.

    Parameter layout (16 total):
        [0:3]   armature        [kg⋅m²]     → written to dof_armature
        [3:6]   Coulomb friction [Nm]        → written to dof_frictionloss (MuJoCo Coulomb)
        [6:9]   viscous friction [Nm·s/rad]  → actuator: ``-viscous * vel``
        [9:12]  (unused)                     → zeroed; kept for CMA-ES layout compat
        [12:15] encoder bias    [rad]        → actuator encoder offset
        [15]    action delay    [sim steps]

    A single CMA-ES process optimises all three joints jointly (no
    :attr:`joint_groups`); the four-way parallelism is across processes, one per
    leg, not within a rollout. Bounds mirror ``DeltaQuadrupedPaceCfg`` so the
    fitted parameters live in the same space as the 12-DOF quadruped fit.

    ``data_dir``/``robot_name`` default to leg LF; override both per leg at the
    fit.py CLI (the four-way launcher does this).
    """

    robot_name: str = "delta_quadruped_leg/LF"
    data_dir: str = "delta_quadruped_leg/LF.pt"
    joint_order: list[str] = _JOINT_ORDER
    bounds_params: torch.Tensor = torch.zeros((_N_JOINTS * 5 + 1, 2))

    def __post_init__(self):
        n = _N_JOINTS
        self.bounds_params[:n, 0] = 1e-2
        self.bounds_params[:n, 1] = 1e-1  # armature [1e-2, 1e-1] kg⋅m²
        self.bounds_params[n : 2 * n, 1] = 3.0  # Coulomb → dof_frictionloss [0, 3] Nm
        self.bounds_params[2 * n : 3 * n, 1] = 2.0  # viscous → actuator [0, 2] Nm·s/rad
        self.bounds_params[3 * n : 4 * n, :] = 0.0  # slot 3 unused: fixed at 0
        self.bounds_params[4 * n : 5 * n, 0] = -0.1
        self.bounds_params[4 * n : 5 * n, 1] = 0.1  # bias [-0.1, 0.1] rad
        self.bounds_params[5 * n, 1] = 7.0  # delay [0, 7] sim steps


@configclass
class DeltaQuadrupedLegActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=_JOINT_ORDER,
        preserve_order=True,
        scale=1.0,
        use_default_offset=False,
    )  # actions = absolute joint position targets, in [T, L, R] order


@configclass
class DeltaQuadrupedLegObservationsCfg:
    """Observation specifications for the MDP.

    Restricted to the three motors: the default (all-joints) observation breaks
    for this robot because the loop-closing spherical joints make ``joint_pos``
    wider than ``default_joint_pos``; PACE only needs the actuated motors.
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
class DeltaQuadrupedLegPaceEnvCfg(PaceSim2realEnvCfg):
    """PACE Sim2Real env for a single 3-DOF aluminium delta leg."""

    scene: DeltaQuadrupedLegPaceSceneCfg = DeltaQuadrupedLegPaceSceneCfg()
    sim2real: DeltaQuadrupedLegPaceCfg = DeltaQuadrupedLegPaceCfg()
    actions: DeltaQuadrupedLegActionsCfg = DeltaQuadrupedLegActionsCfg()
    observations: DeltaQuadrupedLegObservationsCfg = DeltaQuadrupedLegObservationsCfg()

    def __post_init__(self):
        # DELTA_ALLREV_ALU_CFG.spawn may have articulation_props=None; the base
        # class __post_init__ touches fix_root_link on it, so initialise first.
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            )
        # The single-leg USD already fixes the base (joint_base_world); set the
        # flag too for parity with the quadruped rig.
        self.scene.robot.spawn.articulation_props.fix_root_link = True

        super().__post_init__()

        # Replay the 200 Hz chirp data 1:1 with control steps (one data row per
        # env.step), matching the quadruped PACE env.
        self.sim.dt = 0.005
        self.decimation = 1

        # MuJoCo-Warp physics, matched to the quadruped PACE/training preset.
        self.sim.physics = DeltaQuadrupedLegPacePhysicsCfg()
