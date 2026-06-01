# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Newton-direct rollout for comparing against rollout_plot.py --implicit_actuator.

Uses identical Newton/Kamino setup to replay_pace_traj.py.  Two init modes:

  Default (data-init): starts from measured_dof_pos[0] with --num_warmup steps.
    → compare against: rollout_plot.py --implicit_actuator

  --replay_mode: starts from USD default (q=0), one-step-then-reset warmup.
    → compare against: replay_pace_traj.py (should produce identical output)

The goal is to isolate whether a mismatch between --implicit_actuator and replay is
in the Isaac-Lab wrapping or in the physics / initial-conditions themselves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import warp as wp

import newton
import newton.solvers

_SCRIPT_DIR = Path(__file__).resolve().parent
# scripts/pace → pace-sim2real → source → IsaacNewton
_REPO_ROOT = _SCRIPT_DIR.parents[3]

_ASSET_FILE = str(
    _REPO_ROOT
    / "source"
    / "lunarleaper-isaaclab"
    / "source"
    / "lunarleaper_isaaclab"
    / "lunarleaper_isaaclab"
    / "assets"
    / "data"
    / "delta_real.usda"
)

_DEFAULT_DATA = _REPO_ROOT / "source" / "pace-sim2real" / "data" / "delta_robot" / "chirp_data.pt"

# Data column order (T=0, L=1, R=2) — must match measured_dof_pos columns.
_JOINT_NAMES = ["T_motor", "L_motor", "R_motor"]


@wp.kernel
def _set_joint_targets(
    joint_target_pos: wp.array(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    targets: wp.array(dtype=wp.float32),
):
    """Write per-motor position targets into the full joint_target_pos array."""
    i = wp.tid()
    joint_target_pos[dof_indices[i]] = targets[i]


def main():
    parser = argparse.ArgumentParser(description="Newton-direct PACE rollout (companion to replay_pace_traj.py).")
    parser.add_argument("--data", type=Path, default=_DEFAULT_DATA, help="Path to .pt trajectory file.")
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per data timestep.")
    parser.add_argument("--ke", type=float, default=5.0, help="Position gain [N·m/rad].")
    parser.add_argument("--kd", type=float, default=0.2, help="Velocity gain [N·m·s/rad].")
    parser.add_argument("--device", type=str, default="cuda", help="Warp device.")
    parser.add_argument("--save", type=Path, default=None, help="Save plot to this path instead of showing.")
    parser.add_argument(
        "--replay_mode",
        action="store_true",
        help="Start from USD default (q=0) + 1 step warmup. Matches replay_pace_traj.py exactly.",
    )
    parser.add_argument(
        "--num_warmup",
        type=int,
        default=50,
        help="Warmup steps (data-init mode only). Default 50 matches rollout_plot.py --implicit_actuator.",
    )
    args = parser.parse_args()

    # --- Load trajectory ---
    print(f"[INFO] Loading data from {args.data}")
    data = torch.load(args.data, map_location="cpu", weights_only=False)
    time_np = data["time"].cpu().numpy()
    des_np = data["des_dof_pos"].cpu().numpy()   # [T, 3]  T, L, R columns
    real_np = data["dof_pos"].cpu().numpy()       # [T, 3]

    num_steps = time_np.shape[0]
    mean_dt = float(np.diff(time_np).mean()) if num_steps > 1 else 0.005
    sim_dt = mean_dt / args.substeps
    print(f"[INFO] {num_steps} steps, mean dt={mean_dt*1e3:.2f} ms, sim_dt={sim_dt*1e3:.3f} ms")

    mode_label = "replay_mode (USD default)" if args.replay_mode else f"data-init (measured_dof_pos[0], {args.num_warmup} warmup steps)"
    print(f"[INFO] Init mode: {mode_label}")

    # --- Build Newton model (identical to replay_pace_traj.py) ---
    device = wp.get_device(args.device)

    robot_builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    newton.solvers.SolverKamino.register_custom_attributes(robot_builder)
    robot_builder.default_shape_cfg.margin = 0.0
    robot_builder.default_shape_cfg.gap = 0.0

    robot_builder.add_usd(
        _ASSET_FILE,
        force_show_colliders=True,
        force_position_velocity_actuation=True,
        enable_self_collisions=False,
        hide_collision_shapes=False,
    )

    base_idx = next(i for i, lbl in enumerate(robot_builder.body_label) if lbl.endswith("base_link"))
    robot_builder.add_joint_fixed(
        parent=-1,
        child=base_idx,
        parent_xform=wp.transform_identity(dtype=wp.float32),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_world(robot_builder)
    model = builder.finalize(skip_validation_joints=True)

    # Map joint labels → DOF index (same logic as replay_pace_traj.py).
    qd_start_np = model.joint_qd_start.numpy()
    label_to_dof: dict[str, int] = {}
    for i, lbl in enumerate(model.joint_label):
        for name in _JOINT_NAMES:
            if name.lower() in lbl.lower() and name not in label_to_dof:
                label_to_dof[name] = int(qd_start_np[i])

    missing = [n for n in _JOINT_NAMES if n not in label_to_dof]
    if missing:
        raise RuntimeError(f"DOF indices not found for {missing}. Available: {model.joint_label}")

    motor_dof_indices = [label_to_dof[n] for n in _JOINT_NAMES]
    print(f"[INFO] Motor DOF indices: {dict(zip(_JOINT_NAMES, motor_dof_indices))}")

    # PD gains on motor DOFs (identical to replay_pace_traj.py).
    ke_np = model.joint_target_ke.numpy()
    kd_np = model.joint_target_kd.numpy()
    kd_np[:] = 0.0
    for dof in motor_dof_indices:
        ke_np[dof] = args.ke
        kd_np[dof] = args.kd
    model.joint_target_ke.assign(ke_np)
    model.joint_target_kd.assign(kd_np)
    model.rigid_contact_max = 528  # matches replay_pace_traj.py

    # --- Solver config (identical to replay_pace_traj.py) ---
    solver_config = newton.solvers.SolverKamino.Config.from_model(model)
    solver_config.use_collision_detector = True
    solver_config.use_fk_solver = False
    solver_config.collision_detector.pipeline = "primitive"
    solver_config.collision_detector.max_contacts = 32
    solver_config.dynamics.preconditioning = True
    solver_config.padmm.primal_tolerance = 1e-4
    solver_config.padmm.dual_tolerance = 1e-4
    solver_config.padmm.compl_tolerance = 1e-4
    solver_config.padmm.max_iterations = 200
    solver_config.padmm.rho_0 = 0.1
    solver_config.padmm.use_acceleration = True
    solver_config.padmm.warmstart_mode = "containers"
    solver_config.padmm.contact_warmstart_method = "geom_pair_net_force"

    solver = newton.solvers.SolverKamino(model=model, config=solver_config)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    motor_dof_wp = wp.array(motor_dof_indices, dtype=wp.int32, device=device)
    target_buf = wp.zeros(3, dtype=wp.float32, device=device)

    # --- Initialization ---
    if args.replay_mode:
        # Exactly matches replay_pace_traj.py: one step to init Kamino internal state, then reset.
        solver.step(state_0, state_1, control, None, sim_dt)
        solver.reset(state_0)
    else:
        # Data-init mode: start from measured_dof_pos[0], then warmup.
        # Step 1: one step + reset to init Kamino internal state (same as replay).
        solver.step(state_0, state_1, control, None, sim_dt)
        solver.reset(state_0)

        # Step 2: set motor joint positions from first measured sample.
        q_np = state_0.joint_q.numpy().copy()
        for name, col in zip(_JOINT_NAMES, range(len(_JOINT_NAMES))):
            dof_idx = label_to_dof[name]
            q_np[dof_idx] = real_np[0, col]
            print(f"[INFO] Setting {name} (DOF {dof_idx}) joint_q = {real_np[0, col]:.4f} rad")
        state_0.joint_q.assign(q_np)

        # Step 3: set targets to hold initial positions.
        target_buf.assign(real_np[0].astype(np.float32))
        wp.launch(_set_joint_targets, dim=3, inputs=[control.joint_target_pos, motor_dof_wp, target_buf])

        # Step 4: warmup — let Kamino settle constraints with initial target.
        print(f"[INFO] Running {args.num_warmup} warmup steps...")
        for _ in range(args.num_warmup):
            for _ in range(args.substeps):
                state_0.clear_forces()
                solver.step(state_0, state_1, control, None, sim_dt)
                solver.update_contacts(contacts, state_0)
                state_0, state_1 = state_1, state_0

    # --- Rollout ---
    sim_dof_pos = np.zeros((num_steps, 3), dtype=np.float32)
    print(f"[INFO] Starting rollout ({num_steps} steps)...")

    for step_i in range(num_steps):
        target_buf.assign(des_np[step_i].astype(np.float32))
        wp.launch(_set_joint_targets, dim=3, inputs=[control.joint_target_pos, motor_dof_wp, target_buf])

        for _ in range(args.substeps):
            state_0.clear_forces()
            solver.step(state_0, state_1, control, None, sim_dt)
            solver.update_contacts(contacts, state_0)
            state_0, state_1 = state_1, state_0

        q_np = state_0.joint_q.numpy()
        for j, dof_idx in enumerate(motor_dof_indices):
            sim_dof_pos[step_i, j] = q_np[dof_idx]

        if (step_i + 1) % 200 == 0:
            pct = (step_i + 1) / num_steps * 100
            print(f"[INFO] {pct:.0f}%  ({step_i + 1}/{num_steps}, t={time_np[step_i]:.2f}s)")

    print("[INFO] Rollout complete.")

    # --- Plot ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle(f"Newton-direct rollout — {mode_label}", fontsize=11)

    for j, (ax, name) in enumerate(zip(axes, _JOINT_NAMES)):
        ax.plot(time_np, des_np[:, j], color="grey", linewidth=1.0, linestyle="--", alpha=0.75, label="Desired")
        ax.plot(time_np, real_np[:, j], color="tab:blue", linewidth=1.5, label="Real (hardware)")
        ax.plot(time_np, sim_dof_pos[:, j], color="tab:orange", linewidth=1.5, label="Sim (Newton-direct)")
        ax.set_ylabel("Position [rad]")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(True, linewidth=0.5)

    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save, dpi=150)
        print(f"[INFO] Plot saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
