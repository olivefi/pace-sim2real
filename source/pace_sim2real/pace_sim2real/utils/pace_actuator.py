# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.actuators import DCMotor
from isaaclab.utils.types import ArticulationActions
from isaaclab.utils import DelayBuffer
if TYPE_CHECKING:
    # only for type checking
    from .pace_actuator_cfg import PaceDCMotorCfg


class PaceDCMotor(DCMotor):
    """Pace DC Motor actuator model with encoder bias and action delay.

    The actuator models a DC motor whose controller receives joint positions in the encoder
    frame by adding a per-joint encoder bias to the true joint positions. In other words,
    the controller operates on biased (encoder) positions rather than the true joint positions.

    The position targets fed to the PD controller are applied after a configurable delay
    (in simulation steps) to represent latency between command calculation and actuation.

    The software implementation is inspired by DelayedPDActuator.
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

        self.position_targets_delay_buffer = DelayBuffer(cfg.max_delay + 1, self._num_envs, device=self._device)
        self.position_targets_delay_buffer.set_time_lag(cfg.max_delay, torch.arange(self._num_envs, device=self._device))

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
        if control_action_sim.joint_efforts is not None:
            # Stiction below the threshold speed, Coulomb (dynamic) friction at or above it.
            friction = torch.where(
                torch.abs(joint_vel) < self.static_friction_threshold,
                self.static_friction,
                self.dynamic_friction,
            )
            control_action_sim.joint_efforts -= friction * torch.sign(joint_vel)
        return control_action_sim
