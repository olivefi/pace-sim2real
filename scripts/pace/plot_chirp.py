# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Plot chirp-excitation data saved by data_collection.py or prepare_delta_data.py.

Usage
-----
python scripts/pace/plot_chirp.py                        # default delta_robot data
python scripts/pace/plot_chirp.py --data path/to/file.pt
python scripts/pace/plot_chirp.py --spectrogram          # also show per-joint spectrograms
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

_DEFAULT_DATA = Path("data") / "delta_robot" / "chirp_data.pt"
_DEFAULT_JOINT_NAMES = ["motor1", "motor2", "motor3"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Plot chirp excitation data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help=f"Path to a chirp .pt file. Defaults to {{project_root}}/{_DEFAULT_DATA}.",
    )
    parser.add_argument(
        "--joint_names",
        nargs="+",
        default=None,
        help="Joint names, one per DOF column. Defaults to motor1/motor2/motor3.",
    )
    parser.add_argument(
        "--spectrogram",
        action="store_true",
        help="Show a spectrogram below each joint's time-domain plot.",
    )
    args = parser.parse_args()

    data_path = args.data if args.data is not None else _project_root() / _DEFAULT_DATA
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = torch.load(data_path, weights_only=False)
    time = data["time"].cpu().numpy()
    dof_pos = data["dof_pos"].cpu().numpy()
    des_dof_pos = data["des_dof_pos"].cpu().numpy()

    n_joints = dof_pos.shape[1]
    joint_names = args.joint_names if args.joint_names is not None else _DEFAULT_JOINT_NAMES
    if len(joint_names) < n_joints:
        joint_names = joint_names + [f"joint{i}" for i in range(len(joint_names), n_joints)]

    dt = float(np.diff(time).mean())
    fs = 1.0 / dt

    n_rows = 2 if args.spectrogram else 1

    for i in range(n_joints):
        fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows), squeeze=False)
        ax_td = axes[0, 0]

        ax_td.plot(time, des_dof_pos[:, i], color="grey", linewidth=1, linestyle="--", alpha=0.7, label="Target")
        ax_td.plot(time, dof_pos[:, i], color="tab:blue", linewidth=1.5, label="Measured")
        ax_td.set_title(f"{joint_names[i]} — chirp trajectory")
        ax_td.set_xlabel("Time [s]")
        ax_td.set_ylabel("Joint position [rad]")
        ax_td.legend()
        ax_td.grid(True, linewidth=0.5)

        if args.spectrogram:
            ax_sg = axes[1, 0]
            error = dof_pos[:, i] - des_dof_pos[:, i]
            ax_sg.specgram(error, Fs=fs, cmap="viridis", NFFT=256, noverlap=128)
            ax_sg.set_title(f"{joint_names[i]} — tracking error spectrogram")
            ax_sg.set_xlabel("Time [s]")
            ax_sg.set_ylabel("Frequency [Hz]")

        fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
