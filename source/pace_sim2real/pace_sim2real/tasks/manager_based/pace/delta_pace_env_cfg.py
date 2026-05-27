# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

import torch
from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg

from pace_sim2real import PaceCfg, PaceSim2realEnvCfg, PaceSim2realSceneCfg
from pace_sim2real.utils import PaceDCMotorCfg

from lunarleaper_isaaclab.assets.robots.delta_robot import DELTA_ALLREV_CFG  # isort: skip


##
# Physics backend — Kamino only (required for closed kinematic loops in the delta arm)
##


@configclass
class DeltaPacePhysicsCfg(PresetCfg):
    """Kamino-only physics preset for the delta robot.

    The delta robot's parallel-arm topology contains PhysicsSphericalJoint
    loop-closing constraints that PhysX cannot handle; Kamino is required.
    """

    default: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(use_fk_solver=False),
        num_substeps=1,
        use_cuda_graph=True,
    )
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(use_fk_solver=False),
        num_substeps=1,
        use_cuda_graph=True,
    )


##
# Actuator
##

DELTA_MOTOR_PACE_CFG = PaceDCMotorCfg(
    joint_names_expr=["T_motor", "L_motor", "R_motor"],
    saturation_effort=20.0,
    effort_limit=10.0,
    velocity_limit=20.0,
    stiffness=5.0,
    damping=0.2,
    encoder_bias=0.0,
    friction=0.0,
    dynamic_friction=0.0,
    viscous_friction=0.0,
    max_delay=0,
)
"""PaceDCMotorCfg for the three delta robot motors.

Stiffness/damping match the RL environment defaults. Friction and bias are set to
zero as initial conditions; CMA-ES will optimise all of these.
"""


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
        },
    )


##
# PACE configuration
##

_N_JOINTS = 3  # motor1, motor2, motor3


@configclass
class DeltaPaceCfg(PaceCfg):
    """PACE optimisation configuration for the delta robot.

    Parameter layout (13 total):
        [0:3]   armature          [kg⋅m²]
        [3:6]   viscous damping   [Nm⋅s/rad]
        [6:9]   static/dynamic friction  [Nm]
        [9:12]  encoder bias      [rad]
        [12]    action delay      [sim steps]
    """

    robot_name: str = "delta_robot"
    data_dir: str = "delta_robot/chirp_data.pt"
    joint_order: list[str] = ["T_motor", "L_motor", "R_motor"]
    bounds_params: torch.Tensor = torch.zeros((_N_JOINTS * 4 + 1, 2))

    def __post_init__(self):
        n = _N_JOINTS
        self.bounds_params[:n, 0] = 1e-5
        self.bounds_params[:n, 1] = 0.01        # armature [1e-5, 0.01] kg⋅m²
        self.bounds_params[n : 2 * n, 1] = 2.0  # viscous damping [0, 2] Nm⋅s/rad
        self.bounds_params[2 * n : 3 * n, 1] = 2.0  # friction [0, 0.5] Nm
        self.bounds_params[3 * n : 4 * n, 0] = -0.1
        self.bounds_params[3 * n : 4 * n, 1] = 0.1  # bias [-0.1, 0.1] rad
        self.bounds_params[4 * n, 1] = 2.0     # delay [0, 10] sim steps


##
# Environment configuration
##


@configclass
class DeltaPaceEnvCfg(PaceSim2realEnvCfg):
    """PACE Sim2Real environment for the 3-DOF delta parallel robot."""

    scene: DeltaPaceSceneCfg = DeltaPaceSceneCfg()
    sim2real: DeltaPaceCfg = DeltaPaceCfg()

    def __post_init__(self):
        # DELTA_ALLREV_CFG.spawn has articulation_props=None; the base class
        # __post_init__ unconditionally sets fix_root_link on it, so initialise first.
        if self.scene.robot.spawn.articulation_props is None:
            self.scene.robot.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg()
        super().__post_init__()

        self.sim.dt = 0.005
        self.decimation = 1

        # Kamino-only physics (required for closed-loop kinematics).
        self.sim.physics = DeltaPacePhysicsCfg()

        # Restrict actions and observations to the 3 motor joints.
        # The base class uses joint_names=[".*"] / no asset_cfg, which would include
        # all 17 articulation joints and cause a size mismatch against default_joint_pos
        # (which only covers the 15 joints declared in DELTA_ALLREV_CFG.init_state).
        # Each term gets its own SceneEntityCfg instance — resolve() mutates joint_ids
        # in-place, so sharing one object between terms causes a consistency error on
        # the second resolve.
        _motor_names = ["T_motor", "L_motor", "R_motor"]
        self.actions.joint_pos.joint_names = _motor_names
        self.observations.policy.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=_motor_names)}
        self.observations.policy.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=_motor_names)}
