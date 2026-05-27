# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Collect chirp-excitation data from a simulated robot for PACE system identification."""

import argparse
import sys

from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="PACE data collection agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
parser.add_argument("--min_frequency", type=float, default=0.1, help="Minimum frequency for the chirp signal in Hz.")
parser.add_argument("--max_frequency", type=float, default=10.0, help="Maximum frequency for the chirp signal in Hz.")
parser.add_argument("--duration", type=float, default=20.0, help="Duration of the chirp signal in seconds.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch import pi  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import pace_sim2real.tasks  # noqa: F401, E402
from pace_sim2real.utils import project_root  # noqa: E402


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "rsl_rl_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        articulation = env.unwrapped.scene["robot"]
        joint_order = env_cfg.sim2real.joint_order
        joint_ids = torch.tensor(
            [articulation.joint_names.index(name) for name in joint_order],
            device=env.unwrapped.device,
        )

        armature = torch.tensor([0.1] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
        damping = torch.tensor([4.5] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
        friction = torch.tensor([0.05] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
        bias = torch.tensor([0.05] * len(joint_ids), device=env.unwrapped.device).unsqueeze(0)
        time_lag = torch.tensor([[5]], dtype=torch.int, device=env.unwrapped.device)
        env.reset()

        device = env.unwrapped.device
        articulation.write_joint_armature_to_sim_index(armature=armature, joint_ids=joint_ids, env_ids=torch.arange(len(armature), device=device))
        articulation.write_joint_damping_to_sim_index(damping=damping, joint_ids=joint_ids, env_ids=torch.arange(len(damping), device=device))
        articulation.write_joint_friction_coefficient_to_sim_index(joint_friction_coeff=friction, joint_ids=joint_ids, env_ids=torch.tensor([0], device=device))
        for actuator in articulation.actuators.values():
            if not hasattr(actuator, "update_encoder_bias"):
                continue
            drive_indices = actuator.joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = (joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0))
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            actuator.update_time_lags(time_lag)
            actuator.update_encoder_bias(bias[:, drive_joint_idx])
            actuator.reset(torch.arange(env.unwrapped.num_envs, device=device))

        data_dir = project_root() / "data" / env_cfg.sim2real.robot_name

        duration = args_cli.duration
        sample_rate = 1 / env.unwrapped.sim.get_physics_dt()
        num_steps = int(duration * sample_rate)
        t = torch.linspace(0, duration, steps=num_steps, device=env.unwrapped.device)
        f0 = args_cli.min_frequency
        f1 = args_cli.max_frequency

        phase = 2 * pi * (f0 * t + ((f1 - f0) / (2 * duration)) * t ** 2)
        chirp_signal = torch.sin(phase)

        trajectory = torch.zeros((num_steps, len(joint_ids)), device=env.unwrapped.device)
        trajectory[:, :] = chirp_signal.unsqueeze(-1)
        trajectory_directions = torch.tensor(
            [1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
            device=env.unwrapped.device,
        )
        trajectory_bias = torch.tensor([0.0, 0.4, 0.8] * 4, device=env.unwrapped.device)
        trajectory_scale = torch.tensor([0.25, 0.5, -2.0] * 4, device=env.unwrapped.device)
        trajectory[:, joint_ids] = (
            (trajectory[:, joint_ids] + trajectory_bias.unsqueeze(0))
            * trajectory_directions.unsqueeze(0)
            * trajectory_scale.unsqueeze(0)
        )

        articulation.write_joint_position_to_sim_index(position=trajectory[0, :].unsqueeze(0) + bias[0, joint_ids])
        articulation.write_joint_velocity_to_sim_index(velocity=torch.zeros((1, len(joint_ids)), device=env.unwrapped.device))

        counter = 0
        dof_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
        dof_target_pos_buffer = torch.zeros(num_steps, len(joint_ids), device=env.unwrapped.device)
        time_data = t

        while True:
            with torch.inference_mode():
                dof_pos_buffer[counter, :] = (
                    env.unwrapped.scene.articulations["robot"].data.joint_pos.torch[0, joint_ids] - bias[0]
                )
                actions = trajectory[counter % num_steps, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                obs, _, _, _, _ = env.step(actions)
                dof_target_pos_buffer[counter, :] = (
                    env.unwrapped.scene.articulations["robot"]._data.joint_pos_target.torch[0, joint_ids]
                )
                counter += 1
                if counter % 400 == 0:
                    print(f"[INFO]: Step {counter / sample_rate:.1f} seconds")
                if counter >= num_steps:
                    break

        env.close()

        from time import sleep
        sleep(1)

        data_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "time": time_data.cpu(),
                "dof_pos": dof_pos_buffer.cpu(),
                "des_dof_pos": dof_target_pos_buffer.cpu(),
            },
            data_dir / "chirp_data.pt",
        )

        import matplotlib.pyplot as plt

        for i in range(len(joint_ids)):
            plt.figure()
            plt.plot(t.cpu().numpy(), dof_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} pos")
            plt.plot(t.cpu().numpy(), dof_target_pos_buffer[:, i].cpu().numpy(), label=f"{joint_order[i]} target", linestyle="dashed")
            plt.title(f"Joint {joint_order[i]} Trajectory")
            plt.xlabel("Time [s]")
            plt.ylabel("Joint position [rad]")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    main()
