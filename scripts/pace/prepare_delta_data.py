# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Convert a delta-robot real-hardware chirp CSV to the .pt file expected by fit.py.

Usage
-----
python scripts/pace/prepare_delta_data.py                       # auto-detect first CSV
python scripts/pace/prepare_delta_data.py --csv path/to/file.csv
python scripts/pace/prepare_delta_data.py --out data/delta_robot/chirp_data.pt
"""

import argparse
import sys
from pathlib import Path

# Allow running without installing the package (plain Python, no Isaac Sim needed).
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "pace_sim2real"))

import torch

from pace_sim2real.utils.csv_parser import parse_delta_csv
from pace_sim2real.utils.paths import project_root

_DEFAULT_CSV_SUBDIR = Path("source") / "pace_sim2real" / "data"
_DEFAULT_OUT = Path("data") / "delta_robot" / "chirp_data.pt"


def _find_csv(search_dir: Path) -> Path:
    """Return the first .csv found in search_dir, raise if none."""
    csvs = sorted(search_dir.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {search_dir}")
    if len(csvs) > 1:
        print(f"[WARN] Multiple CSVs found, using: {csvs[0].name}")
    return csvs[0]


def main():
    parser = argparse.ArgumentParser(description="Prepare delta-robot chirp data for PACE fitting.")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Path to the input CSV. If omitted, the first .csv in "
             f"{{project_root}}/{_DEFAULT_CSV_SUBDIR} is used.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"Output .pt path (default: {{project_root}}/{_DEFAULT_OUT}).",
    )
    args = parser.parse_args()

    root = project_root()

    csv_path = args.csv if args.csv is not None else _find_csv(root / _DEFAULT_CSV_SUBDIR)
    out_path = args.out if args.out is not None else root / _DEFAULT_OUT

    print(f"[INFO] Reading CSV:   {csv_path}")
    data = parse_delta_csv(csv_path, invert_dofs=[])  # invert_T=true in the hw interface already maps T_motor to Kamino convention

    T = data["time"].shape[0]
    duration = data["time"][-1].item()
    dt_mean = (data["time"][1:] - data["time"][:-1]).mean().item()
    print(f"[INFO] Parsed {T} timesteps, duration={duration:.2f}s, mean dt={dt_mean*1e3:.2f}ms")
    print(f"[INFO] dof_pos range:     [{data['dof_pos'].min():.4f}, {data['dof_pos'].max():.4f}] rad")
    print(f"[INFO] des_dof_pos range: [{data['des_dof_pos'].min():.4f}, {data['des_dof_pos'].max():.4f}] rad")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, out_path)
    print(f"[INFO] Saved to:     {out_path}")


if __name__ == "__main__":
    main()
