# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-env rollout with best-fit PACE parameters, then plot real targets, real positions, sim positions."""

import argparse
import re
import sys
from pathlib import Path

from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="Single-env PACE rollout and plot.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Delta-v0", help="Name of the task.")
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of (identical) envs to roll out; env 0 is plotted. >1 avoids the "
    "single-env Kamino instability seen on the delta legs.",
)
parser.add_argument("--data_file", type=str, default=None, help="Path to a .pt trajectory file.")
parser.add_argument("--folder_name", type=str, default=None, help="Log folder name (default: most recent).")
parser.add_argument("--mean_name", type=str, default=None, help="Params file name, e.g. mean_050.pt (default: latest).")
parser.add_argument("--robot_name", type=str, default=None, help="Robot name for log dir (default: from env cfg).")
parser.add_argument("--save_plot", type=str, default=None, help="Save plots to this directory instead of showing.")
parser.add_argument(
    "--implicit_actuator",
    action="store_true",
    default=False,
    help="Replace PACE actuator with ImplicitActuatorCfg (same drive mode as replay_pace_traj.py). "
    "Disables PACE physics params; useful for isolating control-frequency effects.",
)
parser.add_argument(
    "--default_params",
    action="store_true",
    default=False,
    help="Roll out with the actuator's default config parameters (no CMA-ES fit): skip loading mean_*.pt "
    "and leave the PaceDCMotor at its built-in armature/friction/bias/delay.",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pace_sim2real.tasks  # noqa: F401, E402
import torch  # noqa: E402
from pace_sim2real.utils import project_root  # noqa: E402

from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

_MEAN_PATTERN = re.compile(r"^mean_(\d+)\.pt$")


def find_latest_params(root: Path):
    best = None
    for p in root.rglob("mean_*.pt"):
        m = _MEAN_PATTERN.match(p.name)
        if not m:
            continue
        num = int(m.group(1))
        if best is None or num > best[0]:
            best = (num, p)
    return (None, None) if best is None else (best[1], best[0])


def apply_params(articulation, sim_joint_ids, sim_params, device):
    """Apply physics parameters (armature, friction, actuator config) to the articulation.

    Joint position and velocity must be written separately, AFTER the Kamino reset mask has
    been consumed by the first env.step().  Writing joint state before that step would be
    silently overwritten by solver.reset() inside _forward_kamino.
    """
    # Canonical param layout (matches CMAESOptimizer / saved mean_*.pt):
    # [armature, viscous, static, dynamic, bias, delay].
    n = sim_joint_ids.shape[0]
    armature_idx = slice(0, n)
    viscous_friction_idx = slice(n, 2 * n)
    static_friction_idx = slice(2 * n, 3 * n)
    dynamic_friction_idx = slice(3 * n, 4 * n)
    bias_idx = slice(4 * n, 5 * n)
    delay_idx = 5 * n

    env_ids = torch.arange(sim_params.shape[0], dtype=torch.long, device=device)

    articulation.write_joint_armature_to_sim_index(
        armature=sim_params[:, armature_idx], joint_ids=sim_joint_ids, env_ids=env_ids
    )
    articulation.write_joint_friction_coefficient_to_sim_index(
        joint_friction_coeff=sim_params[:, viscous_friction_idx], joint_ids=sim_joint_ids, env_ids=env_ids
    )

    for actuator in articulation.actuators.values():
        if not hasattr(actuator, "update_encoder_bias"):
            continue
        drive_indices = actuator.joint_indices
        if isinstance(drive_indices, slice):
            all_idx = torch.arange(sim_joint_ids.shape[0], device=device)
            drive_indices = all_idx[drive_indices]
        comparison_matrix = sim_joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
        drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
        actuator.update_encoder_bias(sim_params[:, bias_idx][:, drive_joint_idx])
        actuator.update_static_friction(sim_params[:, static_friction_idx][:, drive_joint_idx])
        actuator.update_dynamic_friction(sim_params[:, dynamic_friction_idx][:, drive_joint_idx])
        actuator.update_time_lags(sim_params[:, delay_idx].to(torch.int))
        actuator.reset(env_ids)
    return bias_idx


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    if args_cli.implicit_actuator:
        print(
            "[INFO]: --implicit_actuator: replacing PACE actuator with ImplicitActuatorCfg "
            "(matches replay_pace_traj.py drive mode; PACE physics params will be skipped)."
        )
        env_cfg.scene.robot = env_cfg.scene.robot.replace(
            actuators={
                "motors": ImplicitActuatorCfg(
                    joint_names_expr=["T_motor", "L_motor", "R_motor"],
                    effort_limit_sim=1000.0,
                    stiffness=5.0,
                    damping=0.2,
                )
            }
        )

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        device = env_cfg.sim.device
        num_envs = args_cli.num_envs

        env = gym.make(args_cli.task, cfg=env_cfg)

        # ── load trajectory data ──────────────────────────────────────────────
        if args_cli.data_file is not None:
            data_file = Path(args_cli.data_file)
        else:
            data_file = project_root() / "data" / env_cfg.sim2real.data_dir
        print(f"[INFO]: Loading trajectory from {data_file}")
        data = torch.load(data_file)
        time_data = data["time"].to(device)
        target_dof_pos = data["des_dof_pos"].to(device)  # [T, n_joints]
        measured_dof_pos = data["dof_pos"].to(device)  # [T, n_joints]
        num_steps = time_data.shape[0]

        # ── joint mapping ─────────────────────────────────────────────────────
        joint_order = env_cfg.sim2real.joint_order
        articulation = env.unwrapped.scene["robot"]
        sim_joint_ids = torch.tensor(
            [articulation.joint_names.index(name) for name in joint_order],
            device=device,
        )
        n = len(joint_order)

        # ── load best-fit parameters (unless running with default params) ─────
        mean = None
        if not args_cli.default_params:
            robot_name = args_cli.robot_name if args_cli.robot_name is not None else env_cfg.sim2real.robot_name
            log_dir = project_root() / "logs" / "pace" / robot_name

            if not log_dir.exists():
                raise FileNotFoundError(f"No logs found at {log_dir}")

            if args_cli.folder_name is not None:
                run_dir = log_dir / args_cli.folder_name
            else:
                candidates = [p for p in log_dir.iterdir() if p.is_dir()]
                if not candidates:
                    raise FileNotFoundError(f"No run folders under {log_dir}")
                run_dir = max(candidates, key=lambda p: p.stat().st_mtime)
            print(f"[INFO]: Using log folder: {run_dir}")

            if args_cli.mean_name is not None:
                params_path = run_dir / args_cli.mean_name
                if not params_path.exists():
                    raise FileNotFoundError(f"Params file not found: {params_path}")
            else:
                params_path, params_num = find_latest_params(run_dir)
                if params_path is None:
                    raise FileNotFoundError(f"No mean_*.pt files under {run_dir}")
            print(f"[INFO]: Loading params from {params_path}")
            mean = torch.load(params_path).to(device)  # shape [n_params] — already in physical units

        # ── diagnostic: verify joint order and param sanity ───────────────────
        print(f"[DEBUG]: articulation.joint_names = {articulation.joint_names}")
        print(f"[DEBUG]: joint_order = {joint_order}")
        print(f"[DEBUG]: sim_joint_ids = {sim_joint_ids.tolist()}")
        if mean is not None:
            print(f"[DEBUG]: armature        = {mean[:n].tolist()}")
            print(f"[DEBUG]: viscous_fric    = {mean[n : 2 * n].tolist()}")
            print(f"[DEBUG]: static_friction = {mean[2 * n : 3 * n].tolist()}")
            print(f"[DEBUG]: dynamic_friction= {mean[3 * n : 4 * n].tolist()}")
            print(f"[DEBUG]: encoder_bias    = {mean[4 * n : 5 * n].tolist()}")
            print(f"[DEBUG]: delay_steps     = {mean[5 * n].item():.3f}")
        else:
            print("[DEBUG]: default_params: using PaceDCMotor built-in config parameters")

        # ── set up simulator ──────────────────────────────────────────────────
        # No warmup: env.reset() leaves the leg at the loop-consistent USD default
        # stance, which Kamino's FK solver makes a valid assembled configuration.
        # We skip the dynamic warmup (a high-armature leg can't be PD-driven cleanly
        # into a start pose) and replay straight from this FK-valid pose.
        env.reset()
        if mean is not None and not args_cli.implicit_actuator:
            apply_params(articulation, sim_joint_ids, mean.unsqueeze(0).repeat(num_envs, 1), device)

        # ── rollout ───────────────────────────────────────────────────────────
        sim_dof_pos_list = []
        with torch.inference_mode():
            for step in range(num_steps):
                actions = target_dof_pos[step].unsqueeze(0).repeat(num_envs, 1)  # [num_envs, n_joints]
                env.step(actions)
                sim_pos = articulation.data.joint_pos.torch[0, sim_joint_ids]  # env 0, [n_joints]
                sim_dof_pos_list.append(sim_pos.cpu())
                if (step + 1) % 200 == 0:
                    print(f"[INFO]: {step + 1}/{num_steps} steps done")

        sim_dof_pos = torch.stack(sim_dof_pos_list, dim=0)  # [T, n_joints]
        n_nan = int(torch.isnan(sim_dof_pos).any(dim=1).sum().item())
        print(
            f"[INFO]: sim_dof_pos range [{torch.nan_to_num(sim_dof_pos).min():.4f}, "
            f"{torch.nan_to_num(sim_dof_pos).max():.4f}] rad; NaN rows: {n_nan}/{sim_dof_pos.shape[0]}"
        )

        # encoder bias for display correction (zero with implicit actuator or default params)
        if mean is not None and not args_cli.implicit_actuator:
            encoder_bias = mean[4 * n : 5 * n].cpu()
        else:
            encoder_bias = torch.zeros(n)

        env.close()

    # ── plot ──────────────────────────────────────────────────────────────────
    time_np = time_data.cpu().numpy()
    save_dir = Path(args_cli.save_plot) if args_cli.save_plot else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for i, name in enumerate(joint_order):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(
            time_np,
            target_dof_pos[:, i].cpu().numpy(),
            color="grey",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Target (real)",
        )
        ax.plot(
            time_np,
            measured_dof_pos[:, i].cpu().numpy(),
            color="tab:green",
            linestyle="--",
            linewidth=2,
            label="Measured (real)",
        )
        ax.plot(
            time_np,
            sim_dof_pos[:, i].numpy() - encoder_bias[i].item(),
            color="tab:orange",
            linewidth=2,
            label="Simulated",
        )
        ax.set_title(f"Joint {name}")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Joint position [rad]")
        ax.legend()
        ax.grid()
        fig.tight_layout()

        if save_dir is not None:
            out = save_dir / f"joint_{name}.png"
            fig.savefig(out, dpi=150)
            print(f"[INFO]: Saved {out}")
            plt.close(fig)
        else:
            plt.show()

    # ── one combined image with all joints stacked ─────────────────────────────
    n = len(joint_order)
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.6 * n), sharex=True, squeeze=False)
    for i, name in enumerate(joint_order):
        ax = axes[i][0]
        ax.plot(
            time_np,
            target_dof_pos[:, i].cpu().numpy(),
            color="grey",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label="Target (cmd)",
        )
        ax.plot(
            time_np,
            measured_dof_pos[:, i].cpu().numpy(),
            color="tab:green",
            linestyle="--",
            linewidth=2,
            label="Measured (real)",
        )
        ax.plot(
            time_np,
            sim_dof_pos[:, i].numpy() - encoder_bias[i].item(),
            color="tab:orange",
            linewidth=2,
            label="Simulated",
        )
        ax.set_ylabel(f"{name} [rad]")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc="upper right", fontsize=8, ncol=3)
    axes[-1][0].set_xlabel("Time [s]")
    fig.suptitle("default params" if args_cli.default_params else "best-fit params")
    fig.tight_layout()
    if save_dir is not None:
        out = save_dir / "cmd_real_sim.png"
        fig.savefig(out, dpi=150)
        print(f"[INFO]: Saved {out}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    main()
