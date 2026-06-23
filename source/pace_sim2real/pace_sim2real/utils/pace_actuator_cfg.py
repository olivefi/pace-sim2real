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

    apply_coulomb_friction: bool = True
    """Whether to apply static/dynamic Coulomb friction in :meth:`~PaceDCMotor.compute`.

    Set to ``False`` when the physics backend (e.g. MuJoCo-Warp) already applies
    dry friction natively via the joint's ``frictionloss`` property — the PACE
    optimiser writes that value through
    :meth:`~isaaclab.assets.Articulation.write_joint_friction_coefficient_to_sim_index`.
    Keeping this ``True`` with a backend that *also* applies ``frictionloss`` would
    double-count the friction torque. Defaults to ``True``."""

    apply_viscous_friction: bool = False
    """Whether to apply velocity-proportional (viscous) friction in :meth:`~PaceDCMotor.compute`.

    When ``True``, a torque of ``-static_friction * joint_vel`` is added to the motor output,
    where :attr:`static_friction` holds the fitted viscous coefficient [N·m·s/rad] updated by
    the PACE optimiser via :meth:`~PaceDCMotor.update_static_friction`.

    Enable when the physics backend does not expose per-environment intrinsic joint damping
    (``dof_damping``) as a writable property — e.g. MuJoCo-Warp, where
    :meth:`~isaaclab.assets.Articulation.write_joint_damping_to_sim_index` writes the actuator
    PD gain (``kd``) rather than the joint's intrinsic damping. In that case the PACE optimiser
    cannot identify viscous friction via the physics engine, so the actuator applies it instead.

    This flag and :attr:`apply_coulomb_friction` are independent: with MuJoCo-Warp set
    ``apply_coulomb_friction=False`` (Coulomb handled by ``frictionloss``) and
    ``apply_viscous_friction=True`` (viscous applied here). Defaults to ``False``."""
