# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import torch
from isaaclab.actuators import DCMotorCfg
from isaaclab.utils.configclass import configclass
from pace_sim2real.utils import pace_actuator


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    """Configuration for the Pace DC motor actuator model.

    Extends :class:`~isaaclab.actuators.DCMotorCfg` with the two effects that genuinely belong
    to the controller — a per-joint encoder offset and a command delay. Everything physical is
    a solver-side joint property, so nothing else is applied here.

    Friction is entirely a solver-side property on the MuJoCo backend:

    * dry (Coulomb) friction → :attr:`~isaaclab.actuators.ActuatorBaseCfg.friction`, written to
      ``dof_frictionloss``,
    * viscous friction → :attr:`~isaaclab.actuators.ActuatorBaseCfg.viscous_friction`,
      written to ``dof_damping`` and integrated implicitly.

    Both are integrated by the solver, so neither costs an actuator-side torque term.
    """

    class_type: type = pace_actuator.PaceDCMotor

    encoder_bias: dict[str, float] | float | None = 0.0
    """Per-joint encoder offset [rad]. The controller sees ``joint_pos - encoder_bias``, so this
    models a calibration error the policy cannot observe. Defaults to 0."""

    max_delay: torch.int | None = 0
    """Depth of the command delay buffer in simulation steps. Sizes the buffer, so it must be at
    least the largest delay ever set at runtime. Defaults to 0."""

