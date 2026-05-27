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
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Delta-v0", help="Name of the task.")
parser.add_argument("--data_file", type=str, default=None, help="Path to a .pt trajectory file.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import pace_sim2real.tasks  # noqa: F401, E402
from pace_sim2real.utils import project_root  # noqa: E402


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

        env.reset()

        counter = 0
        while True:
            with torch.inference_mode():
                actions = target_dof_pos[counter % num_steps].unsqueeze(0).expand(env.unwrapped.num_envs, -1)
                env.step(actions)
                counter += 1
                if counter % 500 == 0:
                    elapsed = counter * env.unwrapped.sim.get_physics_dt()
                    print(f"[INFO]: {elapsed:.1f}s played ({counter} steps, looping every {num_steps} steps)")

        env.close()


if __name__ == "__main__":
    main()
