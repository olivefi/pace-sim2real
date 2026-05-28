# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Utilities for parsing real-robot CSV recordings into the .pt format used by fit.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

# Maps CSV column prefix → index in joint_order ["T_motor", "L_motor", "R_motor"]
_DELTA_CSV_PREFIX_TO_IDX: dict[str, int] = {
    "T_motor": 0,  # T_motor
    "L_motor": 1,  # L_motor
    "R_motor": 2,  # R_motor
}


def parse_delta_csv(
    csv_path: Path | str,
    invert_dofs: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Parse a delta-robot real-hardware chirp CSV into the format expected by fit.py.

    The CSV must contain the following columns (written by the delta robot's logging):
        t
        {T_motor, L_motor, R_motor}_target   — commanded position [rad]
        {T_motor, L_motor, R_motor}_pos      — measured position  [rad]
        {T_motor, L_motor, R_motor}_vel      — measured velocity  [rad/s]

    The returned dict maps to the three simulation joints in order
    ``["T_motor", "L_motor", "R_motor"]``, which is the ``joint_order`` defined in
    ``DeltaPaceCfg``.

    Args:
        csv_path: Path to the CSV file.
        invert_dofs: Indices of output joints (0=T_motor, 1=L_motor, 2=R_motor) whose
            sign should be flipped in both ``dof_pos`` and ``des_dof_pos`` after import.
            Useful when the hardware sign convention differs from the simulation convention.

    Returns:
        Dict with keys:
            ``time``        — shape ``[T]``,    float32, seconds
            ``dof_pos``     — shape ``[T, 3]``, float32, measured joint positions [rad]
            ``des_dof_pos`` — shape ``[T, 3]``, float32, commanded targets [rad]
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    data = np.genfromtxt(csv_path, delimiter=",", names=True)

    n = len(data)
    n_joints = len(_DELTA_CSV_PREFIX_TO_IDX)

    time = torch.tensor(data["t"].astype(np.float32))
    dof_pos = torch.zeros((n, n_joints), dtype=torch.float32)
    des_dof_pos = torch.zeros((n, n_joints), dtype=torch.float32)

    for prefix, idx in _DELTA_CSV_PREFIX_TO_IDX.items():
        dof_pos[:, idx] = torch.tensor(data[f"{prefix}_pos"].astype(np.float32))
        des_dof_pos[:, idx] = torch.tensor(data[f"{prefix}_target"].astype(np.float32))

    if invert_dofs:
        for idx in invert_dofs:
            dof_pos[:, idx] *= -1.0
            des_dof_pos[:, idx] *= -1.0

    return {"time": time, "dof_pos": dof_pos, "des_dof_pos": des_dof_pos}
