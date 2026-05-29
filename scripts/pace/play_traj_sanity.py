# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Sanity-check trajectory playback: replay a saved chirp trajectory in the delta-robot sim."""

import argparse
import sys
from pathlib import Path

from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="Play back a saved trajectory in a Newton environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Delta-v0", help="Name of the task.")
parser.add_argument("--data_file", type=str, default=None, help="Path to a .pt trajectory file.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import matplotlib.axes  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import pace_sim2real.tasks  # noqa: F401, E402
from pace_sim2real.utils import project_root  # noqa: E402

_PLOT_UPDATE_INTERVAL = 10
_DEFAULT_JOINT_NAMES = ["motor1", "motor2", "motor3"]


def _setup_live_plot(time_np, target_np, real_np, joint_names):
    """Create an interactive figure with static chirp/real lines and a live sim line per joint."""
    n_joints = target_np.shape[1]
    plt.ion()
    fig, axes_arr = plt.subplots(n_joints, 1, figsize=(12, 4 * n_joints), sharex=True, squeeze=False)
    axes: list[matplotlib.axes.Axes] = list(axes_arr[:, 0])

    sim_lines = []
    for i, (ax, name) in enumerate(zip(axes, joint_names)):
        ax.plot(time_np, target_np[:, i], color="grey", linestyle="--", linewidth=1.5, alpha=0.7, label="Chirp (target)")
        if real_np is not None:
            ax.plot(time_np, real_np[:, i], color="tab:green", linewidth=1.5, label="Real")
        (sim_line,) = ax.plot([], [], color="tab:orange", linewidth=2, label="Sim")
        sim_lines.append(sim_line)
        ax.set_title(f"Joint {name}")
        ax.set_ylabel("Joint position [rad]")
        ax.legend(loc="upper right")
        ax.grid(True, linewidth=0.5)

    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("Chirp trajectory — live playback (env 0)")
    fig.tight_layout()
    fig.canvas.draw()
    plt.pause(0.001)
    return fig, axes, sim_lines


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        if args_cli.data_file is not None:
            data_file = Path(args_cli.data_file)
        else:
            data_file = project_root() / "data" / env_cfg.sim2real.data_dir
        print(f"[INFO]: Loading trajectory from {data_file}")

        data = torch.load(data_file)
        target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)  # [T, n_joints]
        num_steps = target_dof_pos.shape[0]
        n_joints = target_dof_pos.shape[1]

        # static reference signals for the plot
        time_np = (
            data["time"].cpu().numpy()
            if "time" in data
            else np.arange(num_steps) * env.unwrapped.sim.get_physics_dt()
        )
        real_np = data["dof_pos"].cpu().numpy() if "dof_pos" in data else None
        target_np = target_dof_pos.cpu().numpy()

        # joint names and sim-side joint index mapping
        has_sim2real = hasattr(env_cfg, "sim2real")
        joint_names = (
            env_cfg.sim2real.joint_order
            if has_sim2real and hasattr(env_cfg.sim2real, "joint_order")
            else _DEFAULT_JOINT_NAMES[:n_joints]
        )
        articulation = env.unwrapped.scene["robot"]
        sim_joint_ids = torch.tensor(
            [articulation.joint_names.index(name) for name in joint_names],
            device=env.unwrapped.device,
        )

        fig, axes, sim_lines = _setup_live_plot(time_np, target_np, real_np, joint_names)
        # accumulates sim positions for env 0 within the current cycle
        sim_history: list[list[float]] = [[] for _ in range(n_joints)]

        env.reset()

        counter = 0
        while True:
            with torch.inference_mode():
                step_in_cycle = counter % num_steps
                actions = target_dof_pos[step_in_cycle].unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                env.step(actions)

                # record env-0 joint positions
                sim_pos = articulation.data.joint_pos.torch[0, sim_joint_ids]
                for i in range(n_joints):
                    sim_history[i].append(sim_pos[i].item())

                if counter % _PLOT_UPDATE_INTERVAL == 0:
                    t_so_far = time_np[: len(sim_history[0])]
                    for i, sim_line in enumerate(sim_lines):
                        sim_line.set_xdata(t_so_far)
                        sim_line.set_ydata(sim_history[i])
                    for ax in axes:
                        ax.relim()
                        ax.autoscale_view()
                    fig.canvas.flush_events()
                    plt.pause(0.001)

                # reset sim history at cycle boundary so the next pass draws cleanly
                if step_in_cycle == num_steps - 1:
                    sim_history = [[] for _ in range(n_joints)]

                counter += 1
                if counter % 500 == 0:
                    elapsed = counter * env.unwrapped.sim.get_physics_dt()
                    print(f"[INFO]: {elapsed:.1f}s played ({counter} steps, looping every {num_steps} steps)")

        env.close()


if __name__ == "__main__":
    main()
