# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Split delta-quadruped HW chirp CSV(s) into four per-leg single-leg .pt files.

The chirp data is recorded on the full aluminium-plate quadruped: all twelve
motors are logged simultaneously. This script slices each leg's three motor
columns out of the recording and writes one ``.pt`` per leg, each holding a
3-DOF trajectory in single-leg motor order ``[T, L, R]``. Those files feed the
single-leg-aluminium PACE env (``Isaac-Pace-Delta-Quadruped-Leg-v0``), so each
leg can be fitted independently and in parallel on the cheap 3-DOF sim.

If several CSVs are given (e.g. different chirp frequencies), each leg's segments
are concatenated end-to-end so every leg is excited by the full frequency
content. Time is rebuilt as a uniform monotonic ramp from the mean dt.

Usage
-----
python scripts/pace/prepare_delta_quadruped_leg_data.py                    # all CSVs in the default dir
python scripts/pace/prepare_delta_quadruped_leg_data.py --csv 30s_0_3hz.csv
python scripts/pace/prepare_delta_quadruped_leg_data.py --csv a.csv b.csv  # concatenated per leg
python scripts/pace/prepare_delta_quadruped_leg_data.py --out_dir data/delta_quadruped_leg
"""

import argparse
import sys
from pathlib import Path

# Allow running without installing the package (plain Python, no Isaac Sim needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "pace_sim2real"))

import torch
from pace_sim2real.utils.csv_parser import parse_delta_quadruped_csv
from pace_sim2real.utils.paths import project_root

_DEFAULT_CSV_SUBDIR = Path("source") / "pace_sim2real" / "data" / "delta_quadruped"
_DEFAULT_OUT_SUBDIR = Path("data") / "delta_quadruped_leg"

# Leg name -> its three columns in the 12-joint, leg-major quadruped joint_order
# (LF, RF, LH, RH; each leg T, L, R). Must match ``_DELTA_QUADRUPED_CSV_PREFIX_TO_IDX``
# in csv_parser and ``_JOINT_GROUPS`` in delta_quadruped_pace_env_cfg.
_LEG_COLS: dict[str, list[int]] = {
    "LF": [0, 1, 2],
    "RF": [3, 4, 5],
    "LH": [6, 7, 8],
    "RH": [9, 10, 11],
}


def _find_csvs(search_dir: Path) -> list[Path]:
    """Return all .csv files in search_dir (filename-sorted), raise if none."""
    csvs = sorted(search_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {search_dir}")
    return csvs


def main():
    parser = argparse.ArgumentParser(description="Split delta-quadruped chirp CSV(s) into per-leg PACE data.")
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Path(s) to the input quadruped CSV(s). Multiple CSVs are concatenated per leg. "
            f"If omitted, all .csv in {{project_root}}/{_DEFAULT_CSV_SUBDIR} are used (filename order)."
        ),
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help=f"Output directory for the per-leg .pt files (default: {{project_root}}/{_DEFAULT_OUT_SUBDIR}).",
    )
    args = parser.parse_args()

    root = project_root()
    csv_paths = args.csv if args.csv is not None else _find_csvs(root / _DEFAULT_CSV_SUBDIR)
    # Resolve relative CSV paths against the default CSV dir for convenience.
    csv_paths = [p if p.is_absolute() or p.exists() else root / _DEFAULT_CSV_SUBDIR / p for p in csv_paths]
    out_dir = args.out_dir if args.out_dir is not None else root / _DEFAULT_OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    parsed = []
    for csv_path in csv_paths:
        print(f"[INFO] Reading CSV: {csv_path}")
        d = parse_delta_quadruped_csv(csv_path, invert_dofs=[])
        print(f"[INFO]   {d['time'].shape[0]} timesteps, duration={d['time'][-1].item():.2f}s")
        parsed.append(d)

    # Concatenate all recordings along time (each leg gets the full sequence).
    dof_pos = torch.cat([d["dof_pos"] for d in parsed], dim=0)
    des_dof_pos = torch.cat([d["des_dof_pos"] for d in parsed], dim=0)
    # Rebuild a uniform monotonic time axis from the mean dt of the first recording.
    t0 = parsed[0]["time"]
    dt = (t0[1:] - t0[:-1]).mean().item()
    T = dof_pos.shape[0]
    time = torch.arange(T, dtype=torch.float32) * dt

    print(f"[INFO] Combined trajectory: {T} timesteps, duration={time[-1].item():.2f}s, dt={dt * 1e3:.2f}ms")

    for leg, cols in _LEG_COLS.items():
        leg_data = {
            "time": time.clone(),
            "dof_pos": dof_pos[:, cols].contiguous(),
            "des_dof_pos": des_dof_pos[:, cols].contiguous(),
        }
        out_path = out_dir / f"{leg}.pt"
        torch.save(leg_data, out_path)
        print(
            f"[INFO] Leg {leg}: dof_pos range "
            f"[{leg_data['dof_pos'].min():.4f}, {leg_data['dof_pos'].max():.4f}] rad -> {out_path}"
        )


if __name__ == "__main__":
    main()
