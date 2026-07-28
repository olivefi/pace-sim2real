# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Layout of the PACE parameter vector and how each block is written into the simulator.

Shared by the CMA-ES fit (:class:`~pace_sim2real.CMAESOptimizer`) and by offline tools that
replay a fitted parameter set, so both drive the simulation through exactly the same sinks.
"""

from __future__ import annotations

import torch


class PaceParamIndex:
    """Block slices of the PACE parameter vector.

    The layout is ``[armature, coulomb, viscous, bias, delay, (backlash), (extra)]``, each
    per-joint block ordered like ``joint_order``. Every block names the physical quantity it
    holds and is written to the sink that applies it:

    * :attr:`armature` [kg*m^2] -> ``dof_armature``
    * :attr:`coulomb_friction` [N*m] -> ``dof_frictionloss`` (dry friction)
    * :attr:`viscous_damping` [N*m*s/rad] -> ``dof_damping`` (integrated implicitly)
    * :attr:`bias` [rad] -> the actuator's encoder offset
    * :attr:`delay` [sim steps] -> the actuator's command delay buffer (one scalar, per env)
    * :attr:`backlash` [rad] -> the ``*_backlash`` joints' position limits, as +/-band/2

    Args:
        num_joints: Number of actuated (scored) joints.
        num_extra: Number of passive/extra joints carrying an armature and a friction entry.
        fit_backlash: Whether the optional per-joint gear-play block is present. It is appended
            after the delay scalar, so the layout without it is unchanged.
    """

    def __init__(self, num_joints: int, num_extra: int = 0, fit_backlash: bool = False):
        n = num_joints
        self.num_joints = n
        self.num_extra = num_extra
        self.armature = slice(0, n)
        self.coulomb_friction = slice(n, 2 * n)
        self.viscous_damping = slice(2 * n, 3 * n)
        self.bias = slice(3 * n, 4 * n)
        self.delay = 4 * n
        base = 4 * n + 1
        if fit_backlash:
            self.backlash = slice(base, base + n)
            base += n
        else:
            self.backlash = None
        self.extra_armature = slice(base, base + num_extra)
        self.extra_friction = slice(base + num_extra, base + 2 * num_extra)
        self.num_params = base + 2 * num_extra


def apply_sim_params(
    articulation,
    sim_params: torch.Tensor,
    idx: PaceParamIndex,
    joint_ids: torch.Tensor,
    *,
    env_ids: torch.Tensor,
    initial_position: torch.Tensor | None = None,
    extra_joint_ids: torch.Tensor | None = None,
    backlash_joint_ids: torch.Tensor | None = None,
) -> None:
    """Write a population of PACE parameters into the simulator.

    Args:
        articulation: The Isaac Lab articulation asset.
        sim_params: Denormalised parameters, shape ``(num_envs, idx.num_params)``.
        idx: Block layout of *sim_params*.
        joint_ids: Indices of the actuated (scored) joints, ordered as the parameter blocks.
        env_ids: Environment indices the rows of *sim_params* correspond to.
        initial_position: Optional start positions for the actuated joints, shape
            ``(num_envs, num_joints)``. When None the joint state is left at the env-reset
            pose, which for a closed-loop robot is the only loop-consistent configuration.
        extra_joint_ids: Indices of the passive/extra joints, whose armature and friction are
            written from the trailing blocks.
        backlash_joint_ids: Indices of the ``*_backlash`` play joints, one per actuated joint
            in the same order. Required when ``idx.backlash`` is present: each fitted band
            ``b`` becomes symmetric position limits ``(-b/2, +b/2)``, matching the convention
            of the training-time ``randomize_backlash_joint_limits`` event.

    Raises:
        ValueError: If the backlash block is present without *backlash_joint_ids*.
    """
    articulation.write_joint_armature_to_sim_index(
        armature=sim_params[:, idx.armature], joint_ids=joint_ids, env_ids=env_ids
    )
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=sim_params[:, idx.coulomb_friction], joint_ids=joint_ids, env_ids=env_ids
    )
    articulation.write_joint_passive_damping_to_sim_index(
        damping=sim_params[:, idx.viscous_damping], joint_ids=joint_ids, env_ids=env_ids
    )
    if initial_position is not None:
        articulation.write_joint_position_to_sim_index(
            position=initial_position + sim_params[:, idx.bias], joint_ids=joint_ids
        )
        articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(initial_position), joint_ids=joint_ids)
    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "update_encoder_bias"):
            continue
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        actuator.update_encoder_bias(sim_params[:, idx.bias][:, drive_joint_idx])
        # NOTE: the delay buffer lags per *environment*, not per joint, so every joint in an
        # environment shares one delay. In a per-leg fit each leg's delay therefore only exists
        # in that leg's own block of environments.
        actuator.update_time_lags(sim_params[:, idx.delay].to(torch.int))
        actuator.reset(env_ids)

    # Gearbox play: the fitted per-motor band becomes the play joint's hard position limits.
    if idx.backlash is not None:
        if backlash_joint_ids is None:
            raise ValueError("A backlash block is present but backlash_joint_ids was not given.")
        half = 0.5 * sim_params[:, idx.backlash]
        articulation.write_joint_position_limit_to_sim_index(
            limits=torch.stack([-half, half], dim=-1),
            joint_ids=backlash_joint_ids,
            env_ids=env_ids,
            warn_limit_violation=False,
        )

    # Apply armature and friction for passive joints (bearings, etc.). These joints have no
    # actuator and no real measurements; they affect the dynamics and are optimised indirectly.
    if extra_joint_ids is not None and idx.num_extra > 0:
        articulation.write_joint_armature_to_sim_index(
            armature=sim_params[:, idx.extra_armature], joint_ids=extra_joint_ids, env_ids=env_ids
        )
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=sim_params[:, idx.extra_friction], joint_ids=extra_joint_ids, env_ids=env_ids
        )
