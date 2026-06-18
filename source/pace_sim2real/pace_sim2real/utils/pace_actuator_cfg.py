# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations
import torch

from isaaclab.utils.configclass import configclass

from isaaclab.actuators import DCMotorCfg
from pace_sim2real.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for Pace DC Motor actuator model.

    This class extends the base DCMotorCfg with Pace-specific parameters.
    """
    class_type: type = pace_actuator.PaceDCMotor
    encoder_bias: dict[str, float] | float | None = 0.0
    max_delay: torch.int | None = 0
    static_friction: dict[str, float] | float | None = 0.0
    """Static (stiction) friction torque [N·m] applied against the direction of motion when the
    joint velocity magnitude is below :attr:`static_friction_threshold`. Defaults to 0."""
    dynamic_friction: dict[str, float] | float | None = 0.0
    """Dynamic (Coulomb) friction torque [N·m] applied against the direction of motion when the
    joint velocity magnitude is at or above :attr:`static_friction_threshold`. Defaults to 0."""
    static_friction_threshold: dict[str, float] | float | None = 0.01
    """Joint speed [rad/s] below which :attr:`static_friction` is applied instead of
    :attr:`dynamic_friction`. Defaults to 0.01."""
