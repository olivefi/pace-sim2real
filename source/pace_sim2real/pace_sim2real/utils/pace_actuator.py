# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.actuators import DCMotor
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """DC motor actuator model with per-joint encoder bias, friction, and action delay.

    The controller receives joint positions in the encoder frame (true position minus bias).
    The commanded position targets are pushed through a delay buffer to model command latency.

    The actuator applies a velocity-dependent friction torque against the direction of motion:
    static (stiction) friction below :attr:`static_friction_threshold` and dynamic (Coulomb)
    friction at or above it. This is separate from the physical joint-friction coefficient.

    The software implementation is inspired by DelayedPDActuator.

    .. note::
        This implementation is kept in sync with
        ``lunarleaper_isaaclab.assets.actuators.pace_dc_motor.PaceDCMotor``; changes here
        should be mirrored there (and vice versa).
    """

    cfg: PaceDCMotorCfg

    def __init__(self, cfg: PaceDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if isinstance(cfg.encoder_bias, (list, tuple)):
            if len(cfg.encoder_bias) != self.num_joints:
                raise ValueError(
                    f"encoder_bias must have {self.num_joints} elements (one per joint), "
                    f"but got {len(cfg.encoder_bias)}: {cfg.encoder_bias}"
                )
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.static_friction = self._parse_joint_parameter(cfg.static_friction, 0.0)
        self.dynamic_friction = self._parse_joint_parameter(cfg.dynamic_friction, 0.0)
        self.static_friction_threshold = self._parse_joint_parameter(cfg.static_friction_threshold, 0.01)
        self.apply_coulomb_friction: bool = cfg.apply_coulomb_friction
        self.apply_viscous_friction: bool = cfg.apply_viscous_friction

        self.position_targets_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.position_targets_delay_buffer.set_time_lag(
            cfg.max_delay, torch.arange(self._num_envs, device=self._device)
        )

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        # reset buffers
        self.position_targets_delay_buffer.reset(env_ids)

    def update_encoder_bias(self, encoder_bias: torch.Tensor):
        self.encoder_bias = encoder_bias

    def update_static_friction(self, static_friction: torch.Tensor):
        self.static_friction = static_friction

    def update_dynamic_friction(self, dynamic_friction: torch.Tensor):
        self.dynamic_friction = dynamic_friction

    def update_time_lags(self, delay: int | torch.Tensor, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = torch.arange(self._num_envs, device=self._device)
        self.position_targets_delay_buffer.set_time_lag(delay, env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # delay the commanded position targets to model latency between command calculation and actuation
        if control_action.joint_positions is not None:
            control_action.joint_positions = self.position_targets_delay_buffer.compute(control_action.joint_positions)
        # compute actuator model with encoder bias added to joint positions (joint position in encoder frame, not simulation frame)
        control_action_sim = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        if control_action_sim.joint_efforts is not None and self.apply_coulomb_friction:
            # Stiction below the threshold speed, Coulomb (dynamic) friction at or above it.
            # Disabled when the physics backend applies dry friction natively via joint
            # frictionloss (e.g. MuJoCo-Warp), to avoid double-counting.
            friction = torch.where(
                torch.abs(joint_vel) < self.static_friction_threshold,
                self.static_friction,
                self.dynamic_friction,
            )
            control_action_sim.joint_efforts -= friction * torch.sign(joint_vel)
        if control_action_sim.joint_efforts is not None and self.apply_viscous_friction:
            # Velocity-proportional (viscous) friction. static_friction holds the fitted
            # viscous coefficient [N·m·s/rad] when apply_coulomb_friction=False.
            control_action_sim.joint_efforts -= self.static_friction * joint_vel
        return control_action_sim
