# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

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
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments (= CMA-ES population size).")
parser.add_argument("--task", type=str, default="Isaac-Pace-Delta-v1", help="Name of the task.")
parser.add_argument("--max_iteration", type=int, default=None, help="Override the configured CMA-ES iteration budget.")
parser.add_argument(
    "--data_dir",
    type=str,
    default=None,
    help="Override sim2real.data_dir (path under <project>/data). Used to point each per-leg fit at its own .pt.",
)
parser.add_argument(
    "--robot_name",
    type=str,
    default=None,
    help="Override sim2real.robot_name (the logs/pace/<robot_name> subdir). Keeps per-leg fit logs separate.",
)
parser.add_argument(
    "--fix_static_friction",
    type=float,
    default=None,
    help="If set, pin every motor's static friction to this value [Nm] (held constant, not optimised).",
)
parser.add_argument("--wandb", action="store_true", help="Log the CMA-ES fit (incl. min score) to Weights & Biases.")
parser.add_argument("--wandb_project", type=str, default="pace_sim2real", help="W&B project name.")
parser.add_argument(
    "--wandb_run_name",
    type=str,
    default=None,
    help="W&B run name (default: derived from robot_name).",
)
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import pace_sim2real.tasks  # noqa: F401, E402
import torch  # noqa: E402
from pace_sim2real import CMAESOptimizer  # noqa: E402
from pace_sim2real.utils import project_root  # noqa: E402
from tqdm import tqdm  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "env_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.max_iteration is not None:
            env_cfg.sim2real.cmaes.max_iteration = args_cli.max_iteration
        if args_cli.data_dir is not None:
            env_cfg.sim2real.data_dir = args_cli.data_dir
        if args_cli.robot_name is not None:
            env_cfg.sim2real.robot_name = args_cli.robot_name
        if args_cli.fix_static_friction is not None:
            # Pin every motor's static-friction bound to a single value so CMA-ES holds
            # it constant (denorm maps a degenerate [v, v] bound to v). Block layout:
            # [2n:3n] = static friction; passive/extra joints are appended later and untouched.
            n = len(env_cfg.sim2real.joint_order)
            env_cfg.sim2real.bounds_params[2 * n : 3 * n, 0] = args_cli.fix_static_friction
            env_cfg.sim2real.bounds_params[2 * n : 3 * n, 1] = args_cli.fix_static_friction
            print(
                f"[INFO] Static friction PINNED at {args_cli.fix_static_friction} Nm for all {n} "
                "motors (held constant, not optimised)."
            )

        # Optional Weights & Biases tracking. Entity/API key come from the
        # WANDB_ENTITY / WANDB_API_KEY environment variables (set on the cluster).
        wandb_run = None
        if args_cli.wandb:
            import wandb

            run_name = args_cli.wandb_run_name or env_cfg.sim2real.robot_name
            wandb_run = wandb.init(
                project=args_cli.wandb_project,
                name=run_name,
                config={
                    "task": args_cli.task,
                    "num_envs": args_cli.num_envs,
                    "robot_name": env_cfg.sim2real.robot_name,
                    "data_dir": env_cfg.sim2real.data_dir,
                    "max_iteration": env_cfg.sim2real.cmaes.max_iteration,
                    "sigma": env_cfg.sim2real.cmaes.sigma,
                    "joint_order": env_cfg.sim2real.joint_order,
                    "fix_static_friction": args_cli.fix_static_friction,
                },
            )

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
            extra_bounds = torch.cat(
                [
                    per_type[0:1].expand(m, -1),  # armature for all extra joints
                    per_type[1:2].expand(m, -1),  # friction for all extra joints
                ],
                dim=0,
            )
            bounds_params = torch.cat([bounds_params, extra_bounds], dim=0)
            print(f"[INFO]: Extra (passive) joints for optimisation: {extra_joint_names}")

        data_file = project_root() / "data" / env_cfg.sim2real.data_dir
        log_dir = project_root() / "logs" / "pace" / env_cfg.sim2real.robot_name

        data = torch.load(data_file)
        time_data = data["time"].to(env.unwrapped.device)
        target_dof_pos = data["des_dof_pos"].to(env.unwrapped.device)
        measured_dof_pos = data["dof_pos"].to(env.unwrapped.device)

        time_steps = time_data.shape[0]
        # Exclude the first 1.5 s (settling transient, no warmup) from the CMA-ES score.
        data_dt = (time_data[1] - time_data[0]).item()
        score_skip_steps = min(time_steps - 1, round(1.5 / data_dt))

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
            joint_groups=env_cfg.sim2real.joint_groups,
            wandb_run=wandb_run,
            score_skip_steps=score_skip_steps,
        )

        # No warmup: env.reset() places the leg at the loop-consistent USD default
        # stance and Kamino's FK solver (use_fk_solver) makes it a valid assembled
        # configuration. We deliberately skip the dynamic warmup — a high-armature
        # leg cannot be PD-driven cleanly into a start pose and that transient
        # would distort the fit — and start replay straight from this FK-valid pose.
        env.reset()
        # initial_position=None: keep the FK-valid reset stance (don't overwrite motors).
        opt.update_simulator(articulation, sim_joint_ids, extra_joint_ids=extra_joint_ids)

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
                    opt.update_simulator(env.unwrapped.scene["robot"], sim_joint_ids, extra_joint_ids=extra_joint_ids)

        rollout_bar.close()
        iteration_bar.close()

        if wandb_run is not None:
            # The pace log dir is ephemeral on the cluster, so persist the final
            # fitted parameters to the W&B run summary as the result of record.
            best = opt.get_best_sim_params()
            n = len(joint_order)
            for i, name in enumerate(joint_order):
                wandb_run.summary[f"final_armature/{name}"] = best[i].item()
                wandb_run.summary[f"final_viscous_friction/{name}"] = best[n + i].item()
                wandb_run.summary[f"final_static_friction/{name}"] = best[2 * n + i].item()
                wandb_run.summary[f"final_dynamic_friction/{name}"] = best[3 * n + i].item()
                wandb_run.summary[f"final_bias/{name}"] = best[4 * n + i].item()
            wandb_run.summary["final_delay"] = best[5 * n].item()
            wandb_run.finish()

        opt.close()
        env.close()


if __name__ == "__main__":
    main()
