# © 2026 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Replay recorded joint targets from a CSV file into a Newton/Kamino simulation.

Reads motor position targets from a CSV produced by the PACE data logger
and feeds them into the delta-arm simulation at the correct simulation time.
Targets are linearly interpolated so the replay rate is determined by the
physics timestep, not the CSV sample rate.

CSV format (columns must be present):
  t, T_motor_target, T_motor_pos, T_motor_vel,
     L_motor_target, L_motor_pos, L_motor_vel,
     R_motor_target, R_motor_pos, R_motor_vel

Two replay modes
----------------
mirror (default):
    Target positions are written directly into the Kamino joint_q each step.
    Velocities are computed on-the-fly as finite differences of the targets.
    Kamino propagates motion through the delta arm's closed-loop constraints.

track:
    Target positions are set as PD targets; the sim dynamics run freely.

Usage::

    ./isaaclab.sh -p source/pace-sim2real/scripts/pace/csv_replay.py \\
        --csv source/pace-sim2real/source/pace_sim2real/data/chirp_1780299582_internal.csv
    ./isaaclab.sh -p source/pace-sim2real/scripts/pace/csv_replay.py \\
        --csv path/to/data.csv --mode track --ke 20 --kd 0.5
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import warp as wp

try:
    import matplotlib.pyplot as plt

    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

import newton
import newton.solvers
import newton.viewer

# ── asset paths ───────────────────────────────────────────────────────────────

_SCRIPT_DIR = Path(__file__).resolve().parent
# scripts/pace → scripts → pace-sim2real → source → IsaacNewton (repo root)
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

_JOINT_NAMES: list[str] = ["T_motor", "L_motor", "R_motor"]

# ── Warp kernels ──────────────────────────────────────────────────────────────


@wp.kernel
def _write_q(
    joint_q: wp.array(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    """Scatter position values into joint_q at selected DOF indices."""
    i = wp.tid()
    joint_q[dof_indices[i]] = values[i]


@wp.kernel
def _write_qd(
    joint_qd: wp.array(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    """Scatter velocity values into joint_qd at selected DOF indices."""
    i = wp.tid()
    joint_qd[dof_indices[i]] = values[i]


@wp.kernel
def _write_target(
    joint_target_pos: wp.array(dtype=wp.float32),
    dof_indices: wp.array(dtype=wp.int32),
    values: wp.array(dtype=wp.float32),
):
    """Scatter position targets into joint_target_pos at selected DOF indices."""
    i = wp.tid()
    joint_target_pos[dof_indices[i]] = values[i]


# ── CSV loader ────────────────────────────────────────────────────────────────


class _CsvReplay:
    """Holds pre-loaded CSV data and provides time-interpolated target queries."""

    def __init__(self, csv_path: str) -> None:
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        self.t = data[:, 0].astype(np.float64)
        # columns: t | T_target T_pos T_vel | L_target L_pos L_vel | R_target R_pos R_vel
        self._targets = data[:, [1, 4, 7]].astype(np.float32)  # (N, 3) — T, L, R

        # Pre-compute finite-difference velocities of the target signal [rad/s].
        dt = np.diff(self.t, prepend=self.t[0])
        dt[0] = dt[1] if len(dt) > 1 else 1.0
        diff = np.diff(self._targets, axis=0, prepend=self._targets[:1])
        self._vel = (diff / dt[:, None]).astype(np.float32)

        self.duration = float(self.t[-1])
        print(
            f"[INFO] Loaded {len(self.t)} rows from {csv_path!r}, "
            f"duration {self.duration:.3f} s"
        )

    def get(self, sim_time: float) -> tuple[np.ndarray, np.ndarray]:
        """Return (pos, vel) interpolated at *sim_time* [rad, rad/s], shape (3,)."""
        t = np.clip(sim_time, self.t[0], self.t[-1])
        pos = np.array(
            [np.interp(t, self.t, self._targets[:, i]) for i in range(3)],
            dtype=np.float32,
        )
        vel = np.array(
            [np.interp(t, self.t, self._vel[:, i]) for i in range(3)],
            dtype=np.float32,
        )
        return pos, vel


# ── Live plotter ──────────────────────────────────────────────────────────────


class _LivePlotter:
    """Rolling-window live plot of target vs sim joint positions."""

    _COLORS = ["tab:blue", "tab:green", "tab:red"]

    def __init__(self, joint_names: list[str], window_s: float = 10.0) -> None:
        self._names = joint_names
        n = len(joint_names)
        maxlen = max(500, int(window_s / 0.005))  # generous upper bound
        self._t: collections.deque = collections.deque(maxlen=maxlen)
        self._tgt: list[collections.deque] = [collections.deque(maxlen=maxlen) for _ in range(n)]
        self._sim: list[collections.deque] = [collections.deque(maxlen=maxlen) for _ in range(n)]

        plt.ion()
        self._fig, axes = plt.subplots(n, 1, figsize=(10, 2.5 * n), sharex=True)
        self._axes: list = [axes] if n == 1 else list(axes)

        self._lines_tgt: list = []
        self._lines_sim: list = []
        for ax, name, color in zip(self._axes, joint_names, self._COLORS):
            (l_tgt,) = ax.plot([], [], "--", color=color, alpha=0.6, linewidth=1.2, label="target")
            (l_sim,) = ax.plot([], [], "-", color=color, linewidth=1.5, label="sim")
            ax.set_ylabel(f"{name}\n[rad]", fontsize=9)
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3)
            self._lines_tgt.append(l_tgt)
            self._lines_sim.append(l_sim)

        self._axes[-1].set_xlabel("sim time [s]")
        self._fig.suptitle("Joint targets (--) vs sim positions (—)")
        self._fig.tight_layout()
        plt.show(block=False)

    def push(self, sim_time: float, tgt_pos: np.ndarray, sim_pos: np.ndarray) -> None:
        self._t.append(sim_time)
        for i in range(len(self._names)):
            self._tgt[i].append(float(tgt_pos[i]))
            self._sim[i].append(float(sim_pos[i]))

    def draw(self) -> None:
        t_arr = np.asarray(self._t)
        for i, (l_tgt, l_sim, ax) in enumerate(
            zip(self._lines_tgt, self._lines_sim, self._axes)
        ):
            l_tgt.set_data(t_arr, np.asarray(self._tgt[i]))
            l_sim.set_data(t_arr, np.asarray(self._sim[i]))
            ax.relim()
            ax.autoscale_view()
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()


# ── Newton / Kamino model ─────────────────────────────────────────────────────


def _build_model(device: wp.context.Device, ke: float, kd: float):
    """Build the Newton delta-robot model (mirrors rollout_direct.py exactly)."""
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

    base_idx = next(
        i for i, lbl in enumerate(robot_builder.body_label) if lbl.endswith("base_link")
    )
    robot_builder.add_joint_fixed(
        parent=-1,
        child=base_idx,
        parent_xform=wp.transform_identity(dtype=wp.float32),
        child_xform=wp.transform_identity(dtype=wp.float32),
    )

    builder = newton.ModelBuilder(up_axis=newton.Axis.Z)
    builder.add_world(robot_builder)
    model = builder.finalize(skip_validation_joints=True)

    qd_start_np = model.joint_qd_start.numpy()
    label_to_dof: dict[str, int] = {}
    for i, lbl in enumerate(model.joint_label):
        for name in _JOINT_NAMES:
            if name.lower() in lbl.lower() and name not in label_to_dof:
                label_to_dof[name] = int(qd_start_np[i])

    missing = [n for n in _JOINT_NAMES if n not in label_to_dof]
    if missing:
        raise RuntimeError(
            f"DOF indices not found for {missing}.\nAvailable: {model.joint_label}"
        )

    motor_dof_indices = [label_to_dof[n] for n in _JOINT_NAMES]
    print(f"[INFO] Motor DOF indices: {dict(zip(_JOINT_NAMES, motor_dof_indices))}")

    ke_np = model.joint_target_ke.numpy()
    kd_np = model.joint_target_kd.numpy()
    kd_np[:] = 0.0
    for dof in motor_dof_indices:
        ke_np[dof] = ke
        kd_np[dof] = kd
    model.joint_target_ke.assign(ke_np)
    model.joint_target_kd.assign(kd_np)
    model.rigid_contact_max = 528

    return model, motor_dof_indices


def _build_solver(model: newton.Model) -> newton.solvers.SolverKamino:
    """Configure Kamino solver (mirrors delta_pace_env_cfg.py settings)."""
    cfg = newton.solvers.SolverKamino.Config.from_model(model)
    cfg.use_collision_detector = True
    cfg.use_fk_solver = False
    cfg.collision_detector.pipeline = "primitive"
    cfg.collision_detector.max_contacts = 32
    cfg.dynamics.preconditioning = True
    cfg.padmm.primal_tolerance = 1e-4
    cfg.padmm.dual_tolerance = 1e-4
    cfg.padmm.compl_tolerance = 1e-4
    cfg.padmm.max_iterations = 200
    cfg.padmm.rho_0 = 0.1
    cfg.padmm.use_acceleration = True
    cfg.padmm.warmstart_mode = "containers"
    cfg.padmm.contact_warmstart_method = "geom_pair_net_force"
    return newton.solvers.SolverKamino(model=model, config=cfg)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay CSV joint targets into a Newton/Kamino delta-robot sim."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to the CSV file with columns: t, T_motor_target, ...",
    )
    parser.add_argument(
        "--mode",
        choices=["mirror", "track"],
        default="mirror",
        help=(
            "mirror: write target joint_q directly into Kamino state each step. "
            "track: use targets as PD targets and run free dynamics."
        ),
    )
    parser.add_argument("--loop", action="store_true", help="Loop the CSV replay indefinitely.")
    parser.add_argument("--substeps", type=int, default=1, help="Physics substeps per control step.")
    parser.add_argument("--dt", type=float, default=0.005, help="Control timestep [s].")
    parser.add_argument("--ke", type=float, default=5.0, help="PD position gain [N·m/rad].")
    parser.add_argument("--kd", type=float, default=0.2, help="PD velocity gain [N·m·s/rad].")
    parser.add_argument("--device", type=str, default="cuda", help="Warp device.")
    parser.add_argument(
        "--log_interval",
        type=int,
        default=200,
        help="Print status every N simulation steps.",
    )
    parser.add_argument("--plot", action="store_true", help="Show live joint-state plot.")
    parser.add_argument(
        "--plot_interval",
        type=int,
        default=10,
        help="Refresh the live plot every N simulation steps.",
    )
    parser.add_argument(
        "--plot_window",
        type=float,
        default=10.0,
        help="Rolling-window duration shown in the live plot [s].",
    )
    args = parser.parse_args()

    sim_dt = args.dt / args.substeps
    n_joints = len(_JOINT_NAMES)

    if args.plot and not _MPL_AVAILABLE:
        print("[WARN] matplotlib not found — --plot ignored.")
        args.plot = False

    print(f"[INFO] Mode: {args.mode}")
    replay = _CsvReplay(args.csv)

    device = wp.get_device(args.device)
    print("[INFO] Building Newton/Kamino model …")
    model, motor_dof_indices = _build_model(device, args.ke, args.kd)
    solver = _build_solver(model)

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()
    contacts = model.contacts()

    motor_dof_wp = wp.array(motor_dof_indices, dtype=wp.int32, device=device)
    pos_buf_wp = wp.zeros(n_joints, dtype=wp.float32, device=device)
    vel_buf_wp = wp.zeros(n_joints, dtype=wp.float32, device=device)

    # Seed joint_q from the first CSV row before the Kamino warm-start.
    initial_pos, _ = replay.get(0.0)
    q_np = state_0.joint_q.numpy().copy()
    for i, dof in enumerate(motor_dof_indices):
        q_np[dof] = initial_pos[i]
    state_0.joint_q.assign(q_np)

    solver.step(state_0, state_1, control, None, sim_dt)
    solver.reset(state_0)
    print("[INFO] Kamino warm-start done.")

    plotter: _LivePlotter | None = (
        _LivePlotter(_JOINT_NAMES, window_s=args.plot_window) if args.plot else None
    )

    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(0.6, -0.6, 0.4), pitch=-20.0, yaw=135.0)

    print(
        f"[INFO] Newton viewer open. Replaying {replay.duration:.1f} s of CSV data"
        + (" (looping)" if args.loop else "") + " — close the window to stop."
    )

    step = 0
    sim_time = 0.0

    while viewer.is_running():
        t_query = sim_time % replay.duration if args.loop else sim_time
        if not args.loop and sim_time > replay.duration:
            break

        tgt_pos, tgt_vel = replay.get(t_query)

        if viewer.should_step():
            pos_buf_wp.assign(tgt_pos)
            vel_buf_wp.assign(tgt_vel)

            if args.mode == "mirror":
                wp.launch(
                    _write_q,
                    dim=n_joints,
                    inputs=[state_0.joint_q, motor_dof_wp, pos_buf_wp],
                )
                wp.launch(
                    _write_qd,
                    dim=n_joints,
                    inputs=[state_0.joint_qd, motor_dof_wp, vel_buf_wp],
                )

            wp.launch(
                _write_target,
                dim=n_joints,
                inputs=[control.joint_target_pos, motor_dof_wp, pos_buf_wp],
            )

            for _ in range(args.substeps):
                state_0.clear_forces()
                viewer.apply_forces(state_0)
                solver.step(state_0, state_1, control, None, sim_dt)
                solver.update_contacts(contacts, state_0)
                state_0, state_1 = state_1, state_0

            sim_time += args.dt
            step += 1

            need_sim_pos = (plotter is not None and step % args.plot_interval == 0) or (
                step % args.log_interval == 0
            )
            if need_sim_pos:
                q_now = state_0.joint_q.numpy()
                sim_pos = np.array(
                    [q_now[motor_dof_indices[i]] for i in range(n_joints)], dtype=np.float32
                )

                if plotter is not None and step % args.plot_interval == 0:
                    plotter.push(sim_time, tgt_pos, sim_pos)
                    plotter.draw()

                if step % args.log_interval == 0:
                    print(
                        f"[INFO] step={step:6d}  t={sim_time:.3f}/{replay.duration:.3f} s"
                        f" | CSV target: {np.round(tgt_pos, 3)}"
                        f" | Sim q:      {np.round(sim_pos, 3)}"
                    )

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    viewer.close()


if __name__ == "__main__":
    main()
