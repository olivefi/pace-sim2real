# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Convert delta-quadruped real-hardware chirp CSVs to the .pt file expected by fit.py.

The four legs of the quadruped are dynamically independent in the PACE rig (the
base is fixed), and PACE fits them with one CMA-ES process each. Each leg can
therefore be excited by a *different* chirp within a single rollout: this script
builds one 30 s trajectory in which leg ``k`` is driven by the ``k``-th chirp
CSV. Compared with concatenating the chirps end-to-end this keeps every CMA-ES
iteration a single 30 s (≈6000-step) replay instead of an N-chirp-long one,
while still exercising all the recorded frequency content across the legs in one
rollout.

The CSVs are assigned to legs in filename order, matching the leg-major
``joint_order`` of ``DeltaQuadrupedPaceCfg`` (LF, RF, LH, RH). Provide exactly as
many CSVs as legs (four); leg ``k`` takes its own three motor columns from CSV
``k``.

Usage
-----
python scripts/pace/prepare_delta_quadruped_data.py                       # all CSVs in the dir
python scripts/pace/prepare_delta_quadruped_data.py --csv lf.csv rf.csv lh.csv rh.csv
python scripts/pace/prepare_delta_quadruped_data.py --out path.pt
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
_DEFAULT_OUT = Path("data") / "delta_quadruped" / "chirp_data.pt"

# Per-leg motor columns into the 12-joint, leg-major joint_order (LF, RF, LH, RH).
# Must match ``_JOINT_GROUPS`` in ``delta_quadruped_pace_env_cfg``.
_LEG_COLS: list[list[int]] = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]


def _find_csvs(search_dir: Path) -> list[Path]:
    """Return all .csv files in search_dir (filename-sorted), raise if none."""
    csvs = sorted(search_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {search_dir}")
    return csvs


def _assign_per_leg(parsed: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Build one trajectory driving each leg with its own chirp.

    ``parsed[k]`` supplies leg ``k``'s three motor columns. All chirps are
    truncated to the shortest common length so the columns stay time-aligned.
    """
    n_legs = len(_LEG_COLS)
    if len(parsed) != n_legs:
        raise ValueError(f"Per-leg assignment needs exactly {n_legs} CSVs (one per leg), got {len(parsed)}.")

    T = min(d["time"].shape[0] for d in parsed)
    n_joints = parsed[0]["dof_pos"].shape[1]
    dof_pos = torch.zeros((T, n_joints), dtype=torch.float32)
    des_dof_pos = torch.zeros((T, n_joints), dtype=torch.float32)
    for k, cols in enumerate(_LEG_COLS):
        dof_pos[:, cols] = parsed[k]["dof_pos"][:T, cols]
        des_dof_pos[:, cols] = parsed[k]["des_dof_pos"][:T, cols]

    return {"time": parsed[0]["time"][:T], "dof_pos": dof_pos, "des_dof_pos": des_dof_pos}


def main():
    parser = argparse.ArgumentParser(description="Prepare delta-quadruped chirp data for PACE fitting.")
    parser.add_argument(
        "--csv",
        type=Path,
        nargs="+",
        default=None,
        help=(
            "Path(s) to the input CSV(s), one per leg in LF/RF/LH/RH order. If omitted, "
            f"all .csv in {{project_root}}/{_DEFAULT_CSV_SUBDIR} are used (filename order)."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output .pt path (default: {{project_root}}/{_DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    root = project_root()

    csv_paths = args.csv if args.csv is not None else _find_csvs(root / _DEFAULT_CSV_SUBDIR)
    out_path = args.out if args.out is not None else root / _DEFAULT_OUT

    parsed = []
    for csv_path in csv_paths:
        print(f"[INFO] Reading CSV:   {csv_path}")
        d = parse_delta_quadruped_csv(csv_path, invert_dofs=[])
        print(f"[INFO]   {d['time'].shape[0]} timesteps, duration={d['time'][-1].item():.2f}s")
        parsed.append(d)

    if len(parsed) == 1:
        # Single recording: play every leg from this one file's targets (all 12
        # joints). All legs share the same chirp; each is still fit against its
        # own measured response, so they identify their own parameters.
        print(f"[INFO] Driving all legs from a single file: {csv_paths[0].name}")
        data = parsed[0]
    else:
        leg_names = ["LF", "RF", "LH", "RH"]
        for k, csv_path in enumerate(csv_paths):
            print(f"[INFO] Leg {leg_names[k]} <- {csv_path.name}")
        data = _assign_per_leg(parsed)

    T = data["time"].shape[0]
    duration = data["time"][-1].item()
    dt_mean = (data["time"][1:] - data["time"][:-1]).mean().item()
    print(f"[INFO] Per-leg trajectory: {T} timesteps, duration={duration:.2f}s, mean dt={dt_mean*1e3:.2f}ms")
    print(f"[INFO] dof_pos range:     [{data['dof_pos'].min():.4f}, {data['dof_pos'].max():.4f}] rad")
    print(f"[INFO] des_dof_pos range: [{data['des_dof_pos'].min():.4f}, {data['des_dof_pos'].max():.4f}] rad")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"[INFO] Saved to:     {out_path}")


if __name__ == "__main__":
    main()
