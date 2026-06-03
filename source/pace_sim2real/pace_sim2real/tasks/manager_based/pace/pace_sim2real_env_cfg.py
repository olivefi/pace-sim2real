# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

# import math
from dataclasses import MISSING
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_newton.physics import KaminoSolverCfg, NewtonCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg

from . import mdp


##
# Physics backend
##


@configclass
class PacePhysicsCfg(PresetCfg):
    """Physics backend presets for Pace Sim2Real environments."""

    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()
    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(),
        num_substeps=1,
        use_cuda_graph=True,
    )


##
# Scene definition
##


@configclass
class PaceSim2realSceneCfg(InteractiveSceneCfg):

    # ground plane
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )
    env_spacing: float = 2.5
    num_envs: int = 4096

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=1.0, use_default_offset=False)  # actions = absolute joint position targets


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=0.0)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=False)


@configclass
class CMAESOptimizerCfg:
    """CMA-ES optimizer configuration."""
    max_iteration: int = 200
    epsilon: float = 1e-2
    sigma: float = 0.5
    save_interval: int = 10
    save_optimization_process: bool = False  # consume more disk space if True, saves optimization process after finishing


@configclass
class PaceCfg:
    """Overall configuration for Pace Sim2Real task."""
    cmaes: CMAESOptimizerCfg = CMAESOptimizerCfg()

    robot_name: str = MISSING
    data_dir: str = MISSING
    joint_order: list = MISSING
    bounds_params: torch.Tensor = MISSING

    # Optional passive joints (e.g. bearings): their armature and friction are
    # optimised but they carry no real-data measurements, so they do not
    # contribute to the CMA-ES score directly.  Each entry is a regex pattern
    # matched against the articulation's joint names at runtime.
    extra_joint_order: list = []
    # Per-parameter-type bounds for the extra joints: shape (2, 2).
    #   Row 0 — armature    [min, max]  [kg⋅m²]
    #   Row 1 — friction    [min, max]  [Nm·s/rad]
    # Every matched joint receives the same bounds (tiled at runtime).
    # Required when extra_joint_order is non-empty; ignored otherwise.
    extra_bounds_params: torch.Tensor | None = None

##
# Environment configuration
##


@configclass
class PaceSim2realEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: PaceSim2realSceneCfg = PaceSim2realSceneCfg()
    # Basic settings
    actions: ActionsCfg = ActionsCfg()  # action = joint position targets (scale = 1.0 -> impedance control)

    observations: ObservationsCfg = ObservationsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    sim2real: PaceCfg = PaceCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        # general settings
        self.decimation = 1
        self.episode_length_s = 99999.0  # long episodes
        # viewer settings
        self.viewer.lookat = (0.0, 0.0, 0.8)
        self.viewer.eye = (2.0, 2.0, 1.5)
        # simulation settings
        self.sim.dt = 0.0025  # 400Hz simulation
        self.sim.render_interval = 4  # render at 100Hz

        self.sim.physics = PacePhysicsCfg()
        # self.scene.robot.spawn.articulation_props.fix_root_link = True
