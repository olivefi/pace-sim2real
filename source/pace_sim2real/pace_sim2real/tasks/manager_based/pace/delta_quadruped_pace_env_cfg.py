# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators.actuator_pd_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass
from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg
from isaaclab_tasks.utils import PresetCfg
from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg

from . import mdp

from lunarleaper_isaaclab.assets.robots.lunarleaper_delta import (  # isort: skip
    DELTA_QUADRUPED_MOTOR_PACE_CFG,
    LUNARLEAPER_ALU_DELTA_CFG,
    LUNARLEAPER_ALU_DELTA_MEDIUM_BACKLASH_NOFOOT_CFG,
)


##
# Physics backend
##


@configclass
class DeltaQuadrupedPacePhysicsCfg(PresetCfg):
    """MuJoCo-Warp physics preset for the delta-legged quadruped PACE environment.

    Identical to the training preset (``LunarleaperDeltaPhysicsCfg`` in
    ``delta_velocity_env_cfg``) so that the actuator parameters fitted here are
    valid for the dynamics seen at training time. MuJoCo-Warp handles the delta
    leg's closed kinematic loops via equality constraints and applies joint
    ``frictionloss`` correctly for closed-loop bodies (unlike Kamino, where
    sim-side dry friction could break loop-closure settlement).
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            use_mujoco_contacts=False,
            tolerance=1e-8,
            iterations=250,
            ls_iterations=100,
            integrator="implicitfast",
        ),
        num_substeps=1,
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
            # max_delay sizes the command delay buffer (history = max_delay + 1); it must be
            # >= the delay upper bound (7) the optimiser samples, else update_time_lags
            # raises when CMA-ES proposes a large delay. The base cfg ships max_delay=4.
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

# Upper bound [rad] of the fitted total gear-play band (see ``fit_backlash``).
_BACKLASH_BAND_MAX: float = 0.05


@configclass
class DeltaQuadrupedPaceCfg(PaceCfg):
    """PACE optimisation configuration for the delta-legged quadruped.

    Parameter layout (49 without gear play, 61 with):
        [0:12]   armature          [kg⋅m²]      → ``dof_armature``
        [12:24]  Coulomb friction  [N·m]        → ``dof_frictionloss`` (dry friction)
        [24:36]  viscous damping   [N·m·s/rad]  → ``dof_damping`` (integrated implicitly)
        [36:48]  encoder bias      [rad]        → actuator encoder offset
        [48]     action delay      [sim steps]  → actuator delay buffer (per environment)
        [49:61]  gear-play band    [rad]        → ``*_backlash`` joint limits (±band/2),
                                                  present only when :attr:`fit_backlash`

    Every physical block is a solver-side joint property; the actuator only applies the
    encoder bias and the command delay. The optimisation is split into four independent
    CMA-ES processes (one per leg) via :attr:`joint_groups`; see
    :class:`~pace_sim2real.CMAESOptimizer`.
    """

    robot_name: str = "delta_quadruped"
    data_dir: str = "delta_quadruped/chirp_data.pt"
    joint_order: list[str] = _JOINT_ORDER
    joint_groups: list[list[int]] = _JOINT_GROUPS
    bounds_params: torch.Tensor = torch.zeros((_N_JOINTS * 4 + 1, 2))

    def __post_init__(self):
        n = _N_JOINTS
        # Rebuilt from scratch (rather than mutated in place) so the class-level default
        # tensor is never shared/edited across cfg instances, and so the optional backlash
        # block can change the length.
        num_params = n * 5 + 1 if self.fit_backlash else n * 4 + 1
        bounds = torch.zeros((num_params, 2))
        bounds[:n, 0] = 1e-2
        bounds[:n, 1] = 1e-0  # armature [1e-2, 1.0] kg⋅m²
        bounds[n : 2 * n, 1] = 5.0  # Coulomb friction [0, 5] N·m
        bounds[2 * n : 3 * n, 1] = 2.0  # viscous damping [0, 2] N·m·s/rad
        bounds[3 * n : 4 * n, 0] = -0.03
        bounds[3 * n : 4 * n, 1] = 0.03  # bias [-0.03, 0.03] rad
        bounds[4 * n, 1] = 7.0  # delay [0, 7] sim steps
        if self.fit_backlash:
            # Total gear-play band [rad] per motor, applied as ±band/2 limits on the
            # ``*_backlash`` joints. Spans the training-time DR range with headroom;
            # 0 = a rigid transmission.
            bounds[4 * n + 1 : 5 * n + 1, 1] = _BACKLASH_BAND_MAX
        self.bounds_params = bounds


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


# Height [m] at which the body centre is pinned, matching the aluminium quadruped's
# spawn height (``_ALU_SPAWN_Z`` in ``lunarleaper_delta``). The PACE scene has no
# ground plane, so this only sets where the held-in-air base sits.
_FIXED_Z: float = 0.243


@configclass
class DeltaQuadrupedEventsCfg:
    """Events for the delta-quadruped PACE environment.

    The delta USD's articulation root is a non-rigid ``Xform``, so
    ``fix_root_link`` cannot materialise a fixed joint (Isaac Lab raises
    ``NotImplementedError`` because the root has no ``RigidBodyAPI``). Instead the
    base is pinned kinematically: :func:`~pace_sim2real.tasks...mdp.hold_root_pose`
    re-writes the root pose to :data:`_FIXED_Z` with identity orientation and zeroes
    the root velocity every step (zero interval), holding the base still so the four
    legs stay dynamically decoupled and each is fitted by its own CMA-ES process
    within a single rollout.
    """

    hold_base = EventTerm(
        func=mdp.hold_root_pose,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        is_global_time=True,
        params={"height": _FIXED_Z},
    )


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
    events: DeltaQuadrupedEventsCfg = DeltaQuadrupedEventsCfg()

    def __post_init__(self):
        # LUNARLEAPER_DELTA_CFG.spawn may have articulation_props=None; ensure it is
        # initialised so self-collisions stay disabled on the loop-closed articulation.
        # The base is NOT fixed via fix_root_link: the delta USD's articulation root is
        # a non-rigid Xform, so Isaac Lab cannot materialise a fixed joint for it and
        # raises NotImplementedError. It is instead pinned kinematically every step by
        # the hold_base interval event (see DeltaQuadrupedEventsCfg), which keeps the
        # four legs dynamically decoupled for the per-leg CMA-ES fit.
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False
            )

        super().__post_init__()

        # Replay the 200 Hz chirp data 1:1 with control steps (one data row per
        # env.step). The control decimation used at training time (4) is a
        # control-rate choice; the physics-solver fidelity below is what governs
        # the leg/loop-closure dynamics the fit observes, and that is matched.
        self.sim.dt = 0.00125
        self.decimation = 4

        # MuJoCo-Warp physics, matched to the training preset
        # (LunarleaperDeltaPhysicsCfg in delta_velocity_env_cfg).
        self.sim.physics = DeltaQuadrupedPacePhysicsCfg()


##
# Footless medium-arm variant with mechanism-level gearbox play
##

# Body-centre hold height [m] for the medium-arm geometry (``_ALU_MEDIUM_SPAWN_Z``
# in ``lunarleaper_delta``). Cosmetic only — the PACE scene has no ground plane.
_FIXED_Z_MEDIUM: float = 0.247


@configclass
class DeltaQuadrupedNofootPaceSceneCfg(PaceSim2realSceneCfg):
    """Scene with the footless medium-arm delta quadruped, gear play in the mechanism.

    This is the machine the chirps were recorded on: medium arms (15 cm lower /
    10.5 cm upper), the 93 g ball feet unbolted, and each motor split by a passive
    ``*_backlash`` play joint. The motors are swapped for PaceDCMotor so the sys-id
    parameters have somewhere to land; see :class:`DeltaQuadrupedPaceSceneCfg` for the
    ``max_delay`` rationale.
    """

    robot: ArticulationCfg = LUNARLEAPER_ALU_DELTA_MEDIUM_BACKLASH_NOFOOT_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        actuators={
            **LUNARLEAPER_ALU_DELTA_MEDIUM_BACKLASH_NOFOOT_CFG.actuators,
            "motors": DELTA_QUADRUPED_MOTOR_PACE_CFG.replace(max_delay=7),
        },
    )


@configclass
class DeltaQuadrupedNofootPaceCfg(DeltaQuadrupedPaceCfg):
    """PACE configuration for the footless quadruped, gear play held at the USD default."""

    robot_name: str = "delta_quadruped_nofoot"
    data_dir: str = "delta_quadruped_nofoot/chirp_data.pt"


@configclass
class DeltaQuadrupedNofootBacklashPaceCfg(DeltaQuadrupedNofootPaceCfg):
    """Same, but the per-motor gear-play band is fitted alongside the other parameters.

    Adds a per-joint parameter block (see :attr:`~PaceCfg.fit_backlash`). The hardware
    encoders sit on the motor side of the gearbox, so the play is not directly visible as a
    position offset; it is identified only through the inertia the motor sees while the gap
    is open versus engaged.
    """

    robot_name: str = "delta_quadruped_nofoot_backlash"
    fit_backlash: bool = True


@configclass
class DeltaQuadrupedNofootEventsCfg(DeltaQuadrupedEventsCfg):
    """Base-pinning event at the medium-arm spawn height."""

    def __post_init__(self):
        self.hold_base.params = {"height": _FIXED_Z_MEDIUM}


@configclass
class DeltaQuadrupedNofootPaceEnvCfg(DeltaQuadrupedPaceEnvCfg):
    """PACE environment for the footless, gear-play delta quadruped (play NOT fitted)."""

    scene: DeltaQuadrupedNofootPaceSceneCfg = DeltaQuadrupedNofootPaceSceneCfg()
    sim2real: DeltaQuadrupedNofootPaceCfg = DeltaQuadrupedNofootPaceCfg()
    events: DeltaQuadrupedNofootEventsCfg = DeltaQuadrupedNofootEventsCfg()


@configclass
class DeltaQuadrupedNofootBacklashPaceEnvCfg(DeltaQuadrupedNofootPaceEnvCfg):
    """Same environment, with the gear-play band added to the fitted parameters."""

    sim2real: DeltaQuadrupedNofootBacklashPaceCfg = DeltaQuadrupedNofootBacklashPaceCfg()
