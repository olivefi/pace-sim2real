# © 2026 ETH Zurich, Robotic Systems Lab
# Licensed under the Apache License 2.0

"""Mirror live hardware joint states into a running Newton/Kamino simulation.

Receives joint positions and velocities from the `joint_bridge` ROS 2 node
running on the robot (Docker container) via a UDP socket — no rclpy required.

Packet format (set by joint_bridge.py): 24 bytes, network byte order.
  struct.pack("!6f", pos_T, pos_L, pos_R, vel_T, vel_L, vel_R)

Two mirror modes
----------------
mirror (default):
    Hardware joint_q / joint_qd are written directly into the Kamino state
    each step.  Kamino propagates the motion through the delta arm's
    closed-loop constraints.

track:
    Hardware positions are set as PD targets; the sim dynamics run freely.

Prerequisites
-------------
* Newton / Kamino available (same environment as rollout_direct.py).
* joint_bridge node running on the robot side (started automatically by
  example_delta.launch.py).  Set SIM_IP in the Docker .env to this machine's
  IP before launching the container.

Usage::

    ./isaaclab.sh -p source/pace-sim2real/scripts/pace/hardware_mirror.py
    ./isaaclab.sh -p source/pace-sim2real/scripts/pace/hardware_mirror.py \\
        --mode track --ke 20 --kd 0.5
    ./isaaclab.sh -p source/pace-sim2real/scripts/pace/hardware_mirror.py \\
        --port 7654 --bind 0.0.0.0
"""

from __future__ import annotations

import argparse
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import warp as wp

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
_UDP_FMT = "!6f"  # network byte order; 6 × float32 = 24 bytes

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


# ── thread-safe hardware state buffer ─────────────────────────────────────────


class _HardwareBuffer:
    """Holds the most recently received UDP packet, thread-safe."""

    def __init__(self, n: int) -> None:
        self._pos = np.zeros(n, dtype=np.float32)
        self._vel = np.zeros(n, dtype=np.float32)
        self._lock = threading.Lock()
        self._event = threading.Event()

    def update(self, pos: np.ndarray, vel: np.ndarray) -> None:
        with self._lock:
            self._pos[:] = pos
            self._vel[:] = vel
        self._event.set()

    def get(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            return self._pos.copy(), self._vel.copy()

    def wait_first(self, timeout: float = 10.0) -> bool:
        return self._event.wait(timeout)


# ── UDP reader ────────────────────────────────────────────────────────────────


def _start_udp_reader(bind_host: str, port: int) -> _HardwareBuffer:
    """Bind a UDP socket and forward incoming packets to the shared buffer."""
    n = len(_JOINT_NAMES)
    buf = _HardwareBuffer(n)
    pkt_size = struct.calcsize(_UDP_FMT)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_host, port))
    sock.settimeout(1.0)  # lets the thread exit cleanly on shutdown

    def _loop() -> None:
        while True:
            try:
                data, _ = sock.recvfrom(64)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != pkt_size:
                continue
            vals = struct.unpack(_UDP_FMT, data)
            buf.update(
                np.array(vals[:n], dtype=np.float32),
                np.array(vals[n:], dtype=np.float32),
            )

    thread = threading.Thread(target=_loop, daemon=True)
    thread.start()
    return buf


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

    # Map joint label → velocity DOF start index (same scheme as rollout_direct.py).
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
        description="Mirror live hardware joint states into a Newton/Kamino delta-robot sim."
    )
    parser.add_argument(
        "--bind",
        type=str,
        default="0.0.0.0",
        help="Local address to bind the UDP socket on (default: all interfaces).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7654,
        help="UDP port to listen on (must match udp_port in joint_bridge / launch file).",
    )
    parser.add_argument(
        "--mode",
        choices=["mirror", "track"],
        default="mirror",
        help=(
            "mirror: write hardware joint_q directly into Kamino state each step. "
            "track: use hardware positions as PD targets and run free dynamics."
        ),
    )
    parser.add_argument("--substeps", type=int, default=4, help="Physics substeps per control step.")
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
    args = parser.parse_args()

    sim_dt = args.dt / args.substeps
    n_joints = len(_JOINT_NAMES)

    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Listening for UDP joint states on {args.bind}:{args.port} …")
    buf = _start_udp_reader(args.bind, args.port)

    print("[INFO] Waiting for first UDP packet from joint_bridge (up to 10 s) …")
    if not buf.wait_first(timeout=10.0):
        raise TimeoutError(
            f"No packet received on port {args.port} within 10 s.\n"
            "Check that joint_bridge is running and SIM_IP points to this machine."
        )
    initial_pos, initial_vel = buf.get()
    print(f"[INFO] First hardware reading — pos: {np.round(initial_pos, 4)}")

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

    # Seed joint_q from initial hardware reading before the Kamino warm-start.
    q_np = state_0.joint_q.numpy().copy()
    for i, dof in enumerate(motor_dof_indices):
        q_np[dof] = initial_pos[i]
    state_0.joint_q.assign(q_np)

    # One step + reset to initialise Kamino's internal constraint state.
    solver.step(state_0, state_1, control, None, sim_dt)
    solver.reset(state_0)
    print("[INFO] Kamino warm-start done.")

    # ── viewer ────────────────────────────────────────────────────────────────
    viewer = newton.viewer.ViewerGL()
    viewer.set_model(model)
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(0.6, -0.6, 0.4), pitch=-20.0, yaw=135.0)

    print("[INFO] Newton viewer open. Starting mirror loop — close the window to stop.")

    step = 0
    sim_time = 0.0
    t_start = time.perf_counter()

    while viewer.is_running():
        hw_pos, hw_vel = buf.get()

        if viewer.should_step():
            pos_buf_wp.assign(hw_pos)
            vel_buf_wp.assign(hw_vel)

            if args.mode == "mirror":
                # Teleport motor joints to hardware values; Kamino propagates
                # through the delta arm's closed-loop constraints.
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

            # Keep PD targets current in both modes so the solver has a reference.
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

            if step % args.log_interval == 0:
                elapsed = time.perf_counter() - t_start
                hz = step / elapsed
                q_now = state_0.joint_q.numpy()
                sim_pos = np.array(
                    [q_now[motor_dof_indices[i]] for i in range(n_joints)], dtype=np.float32
                )
                print(
                    f"[INFO] step={step:6d}  {hz:5.0f} Hz"
                    f" | HW pos: {np.round(hw_pos, 3)}"
                    f" | Sim q:  {np.round(sim_pos, 3)}"
                )

        viewer.begin_frame(sim_time)
        viewer.log_state(state_0)
        viewer.log_contacts(contacts, state_0)
        viewer.end_frame()

    viewer.close()


if __name__ == "__main__":
    main()
