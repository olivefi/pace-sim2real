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
    """DC motor actuator model with a per-joint encoder bias and a command delay.

    The controller receives joint positions in the encoder frame (true position minus bias),
    and commanded position targets are pushed through a delay buffer to model command latency.

    Friction is *not* applied here. Both dry and viscous friction are solver-side joint
    properties on the MuJoCo backend — ``dof_frictionloss`` and ``dof_damping`` respectively,
    configured via :attr:`~PaceDCMotorCfg.friction` and :attr:`~PaceDCMotorCfg.viscous_friction`
    — where they are integrated implicitly rather than added as an explicit actuator torque
    computed from the previous step's velocity.

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
        return super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
