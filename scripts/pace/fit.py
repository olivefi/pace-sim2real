# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Run the PACE CMA-ES system-identification fitting loop."""

import argparse
import re
import sys

from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="PACE fitting agent for Isaac Lab environments.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of environments (= CMA-ES population size).")
parser.add_argument("--task", type=str, default="Isaac-Pace-Anymal-D-v0", help="Name of the task.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import pace_sim2real.tasks  # noqa: F401, E402
from pace_sim2real import CMAESOptimizer  # noqa: E402
from pace_sim2real.utils import project_root  # noqa: E402


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "env_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

        env = gym.make(args_cli.task, cfg=env_cfg)

        print(f"[INFO]: Gym observation space: {env.observation_space}")
        print(f"[INFO]: Gym action space: {env.action_space}")

        bounds_params = env_cfg.sim2real.bounds_params.to(env.unwrapped.device)
        articulation = env.unwrapped.scene["robot"]
        joint_order = env_cfg.sim2real.joint_order
        sim_joint_ids = torch.tensor(
            [articulation.joint_names.index(name) for name in joint_order],
            device=env.unwrapped.device,
        )

        # Resolve passive (extra) joints from regex patterns and build their bounds.
        extra_joint_names: list[str] = []
        for joint_name in articulation.joint_names:
            for pattern in env_cfg.sim2real.extra_joint_order:
                if re.fullmatch(pattern, joint_name):
                    extra_joint_names.append(joint_name)
                    break

        extra_joint_ids = None
        if extra_joint_names:
            extra_joint_ids = torch.tensor(
                [articulation.joint_names.index(name) for name in extra_joint_names],
                device=env.unwrapped.device,
            )
            # extra_bounds_params shape (2, 2): row 0 = armature, row 1 = friction.
            # Tile so each matched joint gets its own row: shape (2*m, 2).
            per_type = env_cfg.sim2real.extra_bounds_params.to(env.unwrapped.device)
            m = len(extra_joint_names)
            extra_bounds = torch.cat([
                per_type[0:1].expand(m, -1),  # armature for all extra joints
                per_type[1:2].expand(m, -1),  # friction for all extra joints
            ], dim=0)
            bounds_params = torch.cat([bounds_params, extra_bounds], dim=0)
            print(f"[INFO]: Extra (passive) joints for optimisation: {extra_joint_names}")

        data_file = project_root() / "data" / env_cfg.sim2real.data_dir
        log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

        data = torch.load(data_file)
        time_data = data["time"].to(env.unwrapped.device)
        target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
        measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)
        initial_dof_pos = measured_dof_pos[0, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

        time_steps = time_data.shape[0]
        sim_dt = env.unwrapped.sim.cfg.dt

        opt = CMAESOptimizer(
            bounds=bounds_params,
            population_size=env.unwrapped.num_envs,
            log_dir=log_dir,
            joint_order=joint_order,
            max_iteration=env_cfg.sim2real.cmaes.max_iteration,
            data=data,
            device=env.unwrapped.device,
            epsilon=env_cfg.sim2real.cmaes.epsilon,
            sigma=env_cfg.sim2real.cmaes.sigma,
            save_interval=env_cfg.sim2real.cmaes.save_interval,
            save_optimization_process=env_cfg.sim2real.cmaes.save_optimization_process,
            extra_joint_order=extra_joint_names,
        )

        num_warmup_steps = 50

        def warmup(warmup_actions: torch.Tensor) -> None:
            # env.reset() sets q_j/q_i to USD defaults; warmup lets the PD pull all joints
            # to the desired start position before the trajectory replay begins.
            for _ in range(num_warmup_steps):
                env.step(warmup_actions)

        env.reset()
        opt.update_simulator(articulation, sim_joint_ids, initial_dof_pos, extra_joint_ids=extra_joint_ids)
        warmup(initial_dof_pos)

        iteration_bar = tqdm(total=env_cfg.sim2real.cmaes.max_iteration, desc="CMA-ES", unit="iter")
        rollout_bar = tqdm(total=time_steps, desc="Rollout", unit="step", leave=False)

        counter = 0
        while True:
            with torch.inference_mode():
                opt.tell(
                    env.unwrapped.scene.articulations["robot"].data.joint_pos.torch[:, sim_joint_ids],
                    measured_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1),
                )
                # Action space == joint_order, so data columns map directly to actions.
                actions = target_dof_pos[counter, :].unsqueeze(0).repeat(env.unwrapped.num_envs, 1)
                env.step(actions)
                counter += 1
                rollout_bar.update(1)
                if counter >= time_steps:
                    rollout_bar.reset()
                    counter = 0
                    opt.evolve()
                    iteration_bar.update(1)
                    if opt.finished():
                        break
                    env.reset()
                    opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, initial_dof_pos, extra_joint_ids=extra_joint_ids)
                    warmup(initial_dof_pos)

        rollout_bar.close()
        iteration_bar.close()

        opt.close()
        env.close()


if __name__ == "__main__":
    main()
