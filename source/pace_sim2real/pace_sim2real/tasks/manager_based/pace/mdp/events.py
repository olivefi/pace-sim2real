# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Event functions specific to the PACE sim2real environments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def hold_root_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Pin an articulation's root upright at a fixed height above its env origin.

    Holds the base kinematically by writing the root pose (identity orientation,
    ``height`` above the environment origin) and zeroing the root velocity. The
    delta quadruped's USD articulation root is a non-rigid ``Xform``, so
    ``fix_root_link`` cannot materialize a fixed joint for it; use this as an
    ``interval`` event with a zero interval so the base is re-pinned on every step,
    keeping the four legs dynamically decoupled for the per-leg PACE fit.

    Args:
        height: Height [m] above the environment origin at which to hold the root.
        asset_cfg: The articulation whose root is held.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    origins = env.scene.env_origins if env_ids is None else env.scene.env_origins[env_ids]
    num = origins.shape[0]

    pose = torch.zeros(num, 7, device=asset.device)
    pose[:, :3] = origins
    pose[:, 2] += height
    pose[:, 6] = 1.0  # identity quaternion (x, y, z, w)
    asset.write_root_pose_to_sim_index(root_pose=pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(num, 6, device=asset.device), env_ids=env_ids)
