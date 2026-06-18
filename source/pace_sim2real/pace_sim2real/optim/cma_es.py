# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

from __future__ import annotations

import os
from datetime import datetime

import cmaes
import torch
from torch.utils.tensorboard import SummaryWriter as TensorboardSummaryWriter


class CMAESOptimizer:
    def __init__(
        self,
        bounds,
        population_size,
        log_dir,
        joint_order,
        max_iteration,
        data,
        device,
        epsilon=None,
        sigma=0.5,
        save_interval=10,
        save_optimization_process=False,
        extra_joint_order=None,
        joint_groups=None,
        wandb_run=None,
        score_skip_steps=0,
    ):
        """Initialize the CMA-ES optimizer.

        Args:
            bounds: Parameter bounds tensor of shape ``(num_params, 2)`` for the
                actuated joints (and the global delay).  When *extra_joint_order*
                is provided the caller must append the extra-joint bounds before
                passing; this tensor therefore already includes those rows.
            population_size: Number of parallel environments (= CMA-ES population).
            log_dir: Directory for Tensorboard logs and checkpoints.
            joint_order: Names of the *actuated* joints whose positions are
                compared against real data for scoring.
            max_iteration: Maximum number of CMA-ES evolution steps.
            data: Dict with keys ``"dof_pos"``, ``"des_dof_pos"``, ``"time"``
                from the real-hardware recording.
            device: Torch device string.
            epsilon: Early-stopping threshold on the relative score spread.
            sigma: Initial CMA-ES step size.
            save_interval: Checkpoint frequency in iterations.
            save_optimization_process: If True, save the full parameter and score
                history to disk when optimisation finishes.
            extra_joint_order: Names of *passive* joints (e.g. bearings) whose
                armature and friction are optimised but which do not contribute
                to the score because no real measurements exist for them.
            joint_groups: Optional list of joint-index groups (each entry lists
                indices into *joint_order*). When given, one independent CMA-ES
                process is run per group and the population is split evenly across
                the groups: group ``g`` owns environments
                ``[g * block, (g + 1) * block)`` (``block = population_size //
                num_groups``) and only the joints in that group are scored for
                those environments. This decomposes a high-dimensional joint fit
                into several low-dimensional per-group fits that converge much
                faster, and is valid only when the groups are dynamically
                independent (e.g. the legs of a fixed-base robot). When None
                (default) a single CMA-ES optimises all joints jointly.
            wandb_run: Optional active Weights & Biases run. When provided, the
                per-iteration metrics (min/max/diff score and the best-scoring
                environment's parameters) are mirrored to W&B alongside the
                Tensorboard logs. When None, only Tensorboard is written.
            score_skip_steps: Number of initial rollout steps to exclude from the
                CMA-ES score. The leg has no warmup, so the first fraction of a
                second is a settling transient (the PD position controller has no
                gravity compensation) that the real recording does not contain;
                excluding it keeps the fit from chasing that transient. The full
                trajectory is still recorded for plotting.
        """

        self.wandb_run = wandb_run
        self.score_skip_steps = score_skip_steps
        self.joint_order = joint_order
        self.extra_joint_order = extra_joint_order or []
        self.max_iteration = max_iteration
        self.epsilon = epsilon
        self.save_interval = save_interval
        self.device = device
        self.save_optimization_process = save_optimization_process
        self._timer_start = datetime.now()  # timer for logging purposes

        # create log_dir in YY_MM_DD_hh-mm-ss format
        folder_time = datetime.now().strftime("%y_%m_%d_%H-%M-%S")
        # create string with time and date
        log_dir = os.path.join(log_dir, folder_time)
        os.makedirs(log_dir, exist_ok=True)
        self.writer = TensorboardSummaryWriter(log_dir=log_dir)
        torch.save(
            {
                "bounds": bounds,
                "joint_order": joint_order,
                "extra_joint_order": self.extra_joint_order,
                "joint_groups": joint_groups,
                "dof_pos": data["dof_pos"],
                "des_dof_pos": data["des_dof_pos"],
                "time": data["time"],
            },
            log_dir + "/config.pt",
        )

        self.bounds = bounds

        # Reference trajectories (commanded target + real measurement), kept for
        # optional per-iteration W&B trajectory plots. The per-env simulated
        # trajectories are accumulated separately in ``sim_dof_pos_buffer``.
        self.des_dof_pos = data["des_dof_pos"]
        self.real_dof_pos = data["dof_pos"]
        self.traj_time = data["time"]

        self.scores_counter = 0
        self.iteration_counter = 0

        self.scores = torch.zeros(population_size, device=device)
        self.scores_buffer = torch.zeros((max_iteration, population_size), device=device)
        self.sim_dof_pos_buffer = torch.zeros(
            (population_size, data["dof_pos"].shape[0], len(joint_order)), device=device
        )

        self.params = torch.zeros((population_size, bounds.shape[0]), device=device)
        self.sim_params = torch.zeros_like(self.params)
        if save_optimization_process:
            self.sim_params_buffer = torch.zeros((max_iteration, population_size, bounds.shape[0]), device=device)

        num_joints = len(joint_order)
        self.num_joints = num_joints
        # Parameter block layout (matches the env-cfg bounds_params and group_cols
        # below): [armature, viscous, static, dynamic, bias, delay]. The viscous
        # block is written to the simulator (joint friction coefficient); the
        # static (stiction) and dynamic (Coulomb) blocks are applied manually in
        # the PaceDCMotor actuator, since Kamino's sim-side dry friction breaks
        # closed-loop settle. The actuator applies static friction below its speed
        # threshold and dynamic friction at or above it. Keep this order in sync
        # with both sinks (`apply` below) — swapping them pushes the wrong block
        # into the sim and into the actuator.
        self.armature_idx = slice(0, num_joints)
        self.viscous_friction_idx = slice(num_joints, 2 * num_joints)
        self.static_friction_idx = slice(2 * num_joints, 3 * num_joints)
        self.dynamic_friction_idx = slice(3 * num_joints, 4 * num_joints)
        self.bias_idx = slice(4 * num_joints, 5 * num_joints)
        self.delay_idx = 5 * num_joints

        # Extra (passive) joint parameters: appended after the delay scalar.
        # Each extra joint gets an armature and a friction coefficient.
        # These params are applied to the simulator but never scored.
        num_extra = len(self.extra_joint_order)
        _base = 5 * num_joints + 1
        self.extra_armature_idx = slice(_base, _base + num_extra)
        self.extra_friction_idx = slice(_base + num_extra, _base + 2 * num_extra)

        # ----- per-group (multi-process) setup -----
        self.multi = joint_groups is not None and len(joint_groups) > 1
        if self.multi:
            if self.extra_joint_order:
                raise ValueError("joint_groups is not supported together with extra_joint_order.")
            if population_size % len(joint_groups) != 0:
                raise ValueError(
                    f"population_size ({population_size}) must be divisible by the number of "
                    f"joint groups ({len(joint_groups)})."
                )
            self.joint_groups = [list(g) for g in joint_groups]
            self.num_groups = len(self.joint_groups)
            self.block = population_size // self.num_groups

            # For each group, the columns of the global parameter vector it owns,
            # ordered [armature(group), viscous(group), static(group),
            # dynamic(group), bias(group), delay]. The delay column (index
            # ``delay_idx``) is owned by every group but only ever written within
            # that group's own (disjoint) block of environments, so there is no
            # conflict.
            n = num_joints
            self.group_cols = []
            self.joint_to_group = [None] * num_joints
            for gi, jg in enumerate(self.joint_groups):
                cols = (
                    [j for j in jg]
                    + [n + j for j in jg]
                    + [2 * n + j for j in jg]
                    + [3 * n + j for j in jg]
                    + [4 * n + j for j in jg]
                    + [self.delay_idx]
                )
                self.group_cols.append(torch.tensor(cols, device=device, dtype=torch.long))
                for j in jg:
                    self.joint_to_group[j] = gi

            # Per-environment score mask: an environment in group ``g``'s block
            # only accumulates squared error for the joints owned by group ``g``.
            self.score_mask = torch.zeros((population_size, num_joints), device=device)
            for gi, jg in enumerate(self.joint_groups):
                self.score_mask[gi * self.block : (gi + 1) * self.block, torch.tensor(jg, device=device)] = 1.0

            self.optimizers = []
            for gi in range(self.num_groups):
                gb = self.bounds[self.group_cols[gi]]  # (dim_g, 2)
                bounds_normalized = torch.ones_like(gb)
                bounds_normalized[:, 0] *= -1
                mean_normalized = torch.zeros(gb.shape[0])
                self.optimizers.append(
                    cmaes.CMA(
                        mean=mean_normalized.cpu().numpy(),
                        sigma=sigma,
                        bounds=bounds_normalized.cpu().numpy(),
                        seed=gi,
                        population_size=self.block,
                    )
                )
            print(f"CMA-ES optimizer initialized with {self.num_groups} per-group processes (block size {self.block}).")
        else:
            self.joint_groups = None
            # all joints scored in every environment
            self.score_mask = torch.ones((population_size, num_joints), device=device)

            bounds_normalized = torch.ones_like(bounds)
            bounds_normalized[:, 0] *= -1
            mean_normalized = torch.zeros_like(bounds[:, 0])
            self.optimizer = cmaes.CMA(
                mean=mean_normalized.cpu().numpy(),
                sigma=sigma,
                bounds=bounds_normalized.cpu().numpy(),
                seed=0,
                population_size=population_size,
            )
            print("CMA-ES optimizer initialized.")

        self._reset_population()
        print("Current iteration: ", self.iteration_counter)

    def ask(self):
        return self.optimizer.ask()

    def tell(self, sim_dof_pos, real_dof_pos):
        # Skip the initial settling transient (no warmup): accumulate score only
        # once past score_skip_steps. The full trajectory is still buffered for plots.
        if self.scores_counter >= self.score_skip_steps:
            err2 = torch.square(sim_dof_pos - real_dof_pos - self.sim_params[:, self.bias_idx])
            self.scores += torch.sum(err2 * self.score_mask, dim=1)
        self.sim_dof_pos_buffer[:, self.scores_counter, :] = sim_dof_pos
        self.scores_counter += 1

    def evolve(self):
        # Normalise by the number of *scored* steps (total minus the skipped transient).
        num_scored = max(1, self.scores_counter - self.score_skip_steps)
        self.scores /= num_scored
        self.scores_buffer[self.iteration_counter, :] = self.scores
        if self.save_optimization_process:
            self.sim_params_buffer[self.iteration_counter, :, :] = self.sim_params

        if self.multi:
            for gi in range(self.num_groups):
                cols = self.group_cols[gi]
                lo = gi * self.block
                solutions = []
                for i in range(self.block):
                    solutions.append((self.params[lo + i, cols].cpu().numpy(), self.scores[lo + i].item()))
                self.optimizers[gi].tell(solutions)
        else:
            solutions = []
            for i in range(self.optimizer.population_size):
                solutions.append((self.params[i].cpu().numpy(), self.scores[i].item()))
            self.optimizer.tell(solutions)

        if self.save_interval > 0 and self.iteration_counter % self.save_interval == 0:
            self.save_checkpoint(self.get_best_sim_params(), self.iteration_counter)
        self._print_iteration()

        self._reset_population()

        self.scores = torch.zeros_like(self.scores)
        self.scores_counter = 0
        self.iteration_counter += 1
        print("CMA-ES optimizer iteration: ", self.iteration_counter)

    def finished(self):
        finished = self.max_iteration <= self.iteration_counter
        last = self.scores_buffer[self.iteration_counter - 1, :]
        if self.epsilon is not None:
            if self.multi:
                # converged only once every group's block has collapsed
                all_converged = True
                for gi in range(self.num_groups):
                    block = last[gi * self.block : (gi + 1) * self.block]
                    diff = (block.max() - block.min()) / block.min()
                    if diff >= self.epsilon:
                        all_converged = False
                        break
                finished = finished or all_converged
            else:
                diff_score = (last.max() - last.min()) / last.min()
                finished = finished or (diff_score < self.epsilon)
        if finished:
            print("CMA-ES optimization finished.")
            self.save_checkpoint(self.get_best_sim_params(), self.iteration_counter - 1, finished=True)
        return finished

    def _reset_population(self):
        if self.multi:
            self.params = torch.zeros_like(self.params)  # non-owned columns: normalized 0 = bound midpoint
            for gi in range(self.num_groups):
                cols = self.group_cols[gi]
                lo = gi * self.block
                for i in range(self.block):
                    self.params[lo + i, cols] = torch.tensor(
                        self.optimizers[gi].ask(), device=self.device, dtype=self.params.dtype
                    )
        else:
            for i in range(self.optimizer.population_size):
                self.params[i, :] = torch.tensor(self.optimizer.ask(), device=self.device)
        self.sim_params = self._params_to_sim_params(self.params)

    def update_simulator(self, articulation, joint_ids, initial_position=None, extra_joint_ids=None):
        """Write the current CMA-ES population's parameters into the simulator.

        Args:
            articulation: The Isaac Lab articulation asset.
            joint_ids: Indices of the actuated joints (scored joints).
            initial_position: Optional starting positions for the scored joints,
                shape ``(num_envs, num_joints)``. When None (default), the joint
                state is left at the env-reset pose — the loop-consistent USD
                default stance — so the closed-loop leg starts in an FK-valid
                configuration and the warmup drives it to the commanded target.
                When given, the *actuated* joints are force-written to
                ``initial_position + bias``; note this leaves the passive loop
                DOFs at the stance, so the leg starts loop-inconsistent and the
                solver must snap the closures shut (legacy behaviour).
            extra_joint_ids: Indices of the passive/extra joints (not scored).
                When provided, their armature and friction are written too.
        """
        env_ids = torch.arange(len(self.sim_params[:, self.armature_idx]), device=self.device)
        articulation.write_joint_armature_to_sim_index(
            armature=self.sim_params[:, self.armature_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        articulation.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=self.sim_params[:, self.viscous_friction_idx], joint_ids=joint_ids, env_ids=env_ids
        )
        if initial_position is not None:
            articulation.write_joint_position_to_sim_index(
                position=initial_position + self.sim_params[:, self.bias_idx], joint_ids=joint_ids
            )
            articulation.write_joint_velocity_to_sim_index(
                velocity=torch.zeros_like(initial_position), joint_ids=joint_ids
            )
        for drive_type, actuator in articulation.actuators.items():
            if not hasattr(actuator, "update_encoder_bias"):
                continue
            drive_indices = actuator.joint_indices
            if isinstance(drive_indices, slice):
                all_idx = torch.arange(joint_ids.shape[0], device=joint_ids.device)
                drive_indices = all_idx[drive_indices]
            comparison_matrix = joint_ids.unsqueeze(1) == drive_indices.unsqueeze(0)
            drive_joint_idx = torch.argmax(comparison_matrix.int(), dim=0)
            actuator.update_encoder_bias(self.sim_params[:, self.bias_idx][:, drive_joint_idx])
            actuator.update_static_friction(self.sim_params[:, self.static_friction_idx][:, drive_joint_idx])
            actuator.update_dynamic_friction(self.sim_params[:, self.dynamic_friction_idx][:, drive_joint_idx])
            actuator.update_time_lags(self.sim_params[:, self.delay_idx].to(torch.int))
            actuator.reset(env_ids)

        # Apply armature and friction for passive joints (bearings, etc.).
        # These joints have no actuator and no real measurements; they affect
        # the simulation dynamics and are thus optimised indirectly.
        if extra_joint_ids is not None and len(self.extra_joint_order) > 0:
            articulation.write_joint_armature_to_sim_index(
                armature=self.sim_params[:, self.extra_armature_idx],
                joint_ids=extra_joint_ids,
                env_ids=env_ids,
            )
            articulation.write_joint_friction_coefficient_to_sim_index(
                joint_friction_coeff=self.sim_params[:, self.extra_friction_idx],
                joint_ids=extra_joint_ids,
                env_ids=env_ids,
            )

    def _print_iteration(self):
        if self.multi:
            print(f"--- iteration {self.iteration_counter} (per-leg best scores) ---")
            for gi in range(self.num_groups):
                lo = gi * self.block
                block_scores = self.scores[lo : lo + self.block]
                local_min = torch.argmin(block_scores)
                env = lo + local_min.item()
                jg = self.joint_groups[gi]
                names = [self.joint_order[j] for j in jg]
                print(f"  leg {gi} {names}: min score {block_scores[local_min].item():.6e}")
                print(f"    Armature:        {self.sim_params[env, self.armature_idx][jg].tolist()}")
                print(f"    Static Friction: {self.sim_params[env, self.static_friction_idx][jg].tolist()}")
                print(f"    Dynamic Friction:{self.sim_params[env, self.dynamic_friction_idx][jg].tolist()}")
                print(f"    Viscous Friction:{self.sim_params[env, self.viscous_friction_idx][jg].tolist()}")
                print(f"    Bias:            {self.sim_params[env, self.bias_idx][jg].tolist()}")
                print(f"    Delay:           {self.sim_params[env, self.delay_idx].item()}")
            print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
            self._timer_start = datetime.now()
            self._log()
            return

        min_score = torch.min(self.scores)
        max_score = torch.max(self.scores)
        min_index = torch.argmin(self.scores)
        print("Max score: ", max_score.item())
        print("Min score: ", min_score.item(), " at index: ", min_index.item())
        print("Armature: ", self.sim_params[min_index, self.armature_idx].tolist())
        print("Static Friction: ", self.sim_params[min_index, self.static_friction_idx].tolist())
        print("Dynamic Friction: ", self.sim_params[min_index, self.dynamic_friction_idx].tolist())
        print("Viscous Friction: ", self.sim_params[min_index, self.viscous_friction_idx].tolist())
        print("Bias: ", self.sim_params[min_index, self.bias_idx].tolist())
        print("Delay: ", self.sim_params[min_index, self.delay_idx].tolist())
        if self.extra_joint_order:
            print("Extra Armature: ", self.sim_params[min_index, self.extra_armature_idx].tolist())
            print("Extra Friction: ", self.sim_params[min_index, self.extra_friction_idx].tolist())
        print(f"Elapsed time: {(datetime.now() - self._timer_start).total_seconds():.1f} seconds")
        self._timer_start = datetime.now()
        self._log()

    def _params_to_sim_params(self, params):
        sim_params = (params + 1.0) / 2.0  # change range from 0 to 1
        sim_params = self.bounds[:, 0] + sim_params * (
            self.bounds[:, 1] - self.bounds[:, 0]
        )  # range from lower to upper bound
        return sim_params

    def _assemble_global_mean(self):
        """Assemble the per-group CMA-ES means into one global normalized vector.

        The delay column is owned by every group; the global config carries a
        single delay, so the group delay means are averaged.
        """
        gparams = torch.zeros(self.bounds.shape[0], device=self.device)
        delay_means = []
        for gi in range(self.num_groups):
            mean_g = torch.tensor(self.optimizers[gi]._mean, device=self.device)
            gparams[self.group_cols[gi]] = mean_g
            delay_means.append(mean_g[-1])  # last entry of each group vector is delay
        gparams[self.delay_idx] = torch.stack(delay_means).mean()
        return gparams

    def get_best_sim_params(self):
        if self.multi:
            return self._params_to_sim_params(self._assemble_global_mean())
        best_params = torch.tensor(self.optimizer._mean, device=self.device)
        return self._params_to_sim_params(best_params)

    def _log(self):
        if self.multi:
            self._log_multi()
            return
        min_score, min_score_index = torch.min(self.scores, dim=0)
        max_score, _ = torch.max(self.scores, dim=0)
        for i in range(len(self.joint_order)):
            self.writer.add_histogram(
                "4_Bias/distribution_" + self.joint_order[i],
                self.sim_params[:, self.bias_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "3_Viscous_Friction/distribution_" + self.joint_order[i],
                self.sim_params[:, self.viscous_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "2b_Dynamic_Friction/distribution_" + self.joint_order[i],
                self.sim_params[:, self.dynamic_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "2_Static_Friction/distribution_" + self.joint_order[i],
                self.sim_params[:, self.static_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "1_Armature/distribution_" + self.joint_order[i],
                self.sim_params[:, self.armature_idx][:, i],
                self.iteration_counter,
            )

            self.writer.add_scalar(
                "4_Bias/best_" + self.joint_order[i],
                self.sim_params[min_score_index, self.bias_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "3_Viscous_Friction/best_" + self.joint_order[i],
                self.sim_params[min_score_index, self.viscous_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "2b_Dynamic_Friction/best_" + self.joint_order[i],
                self.sim_params[min_score_index, self.dynamic_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "2_Static_Friction/best_" + self.joint_order[i],
                self.sim_params[min_score_index, self.static_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "1_Armature/best_" + self.joint_order[i],
                self.sim_params[min_score_index, self.armature_idx][i].item(),
                self.iteration_counter,
            )
        self.writer.add_histogram("0_Delay/distribution", self.sim_params[:, self.delay_idx], self.iteration_counter)
        self.writer.add_scalar(
            "0_Delay/best", self.sim_params[min_score_index, self.delay_idx].item(), self.iteration_counter
        )

        for i, name in enumerate(self.extra_joint_order):
            self.writer.add_histogram(
                "5_Extra_Armature/distribution_" + name,
                self.sim_params[:, self.extra_armature_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "5_Extra_Friction/distribution_" + name,
                self.sim_params[:, self.extra_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "5_Extra_Armature/best_" + name,
                self.sim_params[min_score_index, self.extra_armature_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "5_Extra_Friction/best_" + name,
                self.sim_params[min_score_index, self.extra_friction_idx][i].item(),
                self.iteration_counter,
            )

        self.writer.add_scalar("0_Episode/score", min_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/max_score", max_score.item(), self.iteration_counter)
        self.writer.add_scalar("0_Episode/diff_score", (max_score - min_score) / min_score, self.iteration_counter)

        if self.wandb_run is not None:
            metrics = {
                "score/min": min_score.item(),
                "score/max": max_score.item(),
                "score/diff": ((max_score - min_score) / min_score).item(),
                "delay/best": self.sim_params[min_score_index, self.delay_idx].item(),
            }
            for i, name in enumerate(self.joint_order):
                metrics[f"armature/{name}"] = self.sim_params[min_score_index, self.armature_idx][i].item()
                metrics[f"viscous_friction/{name}"] = self.sim_params[min_score_index, self.viscous_friction_idx][
                    i
                ].item()
                metrics[f"static_friction/{name}"] = self.sim_params[min_score_index, self.static_friction_idx][
                    i
                ].item()
                metrics[f"dynamic_friction/{name}"] = self.sim_params[min_score_index, self.dynamic_friction_idx][
                    i
                ].item()
                metrics[f"bias/{name}"] = self.sim_params[min_score_index, self.bias_idx][i].item()
            for i, name in enumerate(self.extra_joint_order):
                metrics[f"extra_armature/{name}"] = self.sim_params[min_score_index, self.extra_armature_idx][i].item()
                metrics[f"extra_friction/{name}"] = self.sim_params[min_score_index, self.extra_friction_idx][i].item()
            # Best rollout's command/real/sim trajectory (global best env for all joints).
            # Merge into the same log call/step as the metrics (a second same-step log is dropped).
            metrics.update(self._best_trajectory_charts([int(min_score_index)] * len(self.joint_order)))
            self.wandb_run.log(metrics, step=self.iteration_counter)

    def _log_multi(self):
        # Each joint is scored only within its group's block; histograms and the
        # logged "best" value are therefore restricted to the owning group's block.
        group_min_env = []
        group_min_score = []
        for gi in range(self.num_groups):
            lo = gi * self.block
            block_scores = self.scores[lo : lo + self.block]
            local_min = torch.argmin(block_scores)
            group_min_env.append(lo + local_min.item())
            group_min_score.append(block_scores[local_min].item())

        for i in range(len(self.joint_order)):
            gi = self.joint_to_group[i]
            lo, hi = gi * self.block, (gi + 1) * self.block
            me = group_min_env[gi]
            self.writer.add_histogram(
                "4_Bias/distribution_" + self.joint_order[i],
                self.sim_params[lo:hi, self.bias_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "3_Viscous_Friction/distribution_" + self.joint_order[i],
                self.sim_params[lo:hi, self.viscous_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "2b_Dynamic_Friction/distribution_" + self.joint_order[i],
                self.sim_params[lo:hi, self.dynamic_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "2_Static_Friction/distribution_" + self.joint_order[i],
                self.sim_params[lo:hi, self.static_friction_idx][:, i],
                self.iteration_counter,
            )
            self.writer.add_histogram(
                "1_Armature/distribution_" + self.joint_order[i],
                self.sim_params[lo:hi, self.armature_idx][:, i],
                self.iteration_counter,
            )

            self.writer.add_scalar(
                "4_Bias/best_" + self.joint_order[i],
                self.sim_params[me, self.bias_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "3_Viscous_Friction/best_" + self.joint_order[i],
                self.sim_params[me, self.viscous_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "2b_Dynamic_Friction/best_" + self.joint_order[i],
                self.sim_params[me, self.dynamic_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "2_Static_Friction/best_" + self.joint_order[i],
                self.sim_params[me, self.static_friction_idx][i].item(),
                self.iteration_counter,
            )
            self.writer.add_scalar(
                "1_Armature/best_" + self.joint_order[i],
                self.sim_params[me, self.armature_idx][i].item(),
                self.iteration_counter,
            )

        for gi in range(self.num_groups):
            lo, hi = gi * self.block, (gi + 1) * self.block
            me = group_min_env[gi]
            self.writer.add_histogram(
                f"0_Delay/distribution_leg{gi}", self.sim_params[lo:hi, self.delay_idx], self.iteration_counter
            )
            self.writer.add_scalar(
                f"0_Delay/best_leg{gi}", self.sim_params[me, self.delay_idx].item(), self.iteration_counter
            )
            self.writer.add_scalar(f"0_Episode/score_leg{gi}", group_min_score[gi], self.iteration_counter)

        self.writer.add_scalar("0_Episode/score", float(sum(group_min_score) / self.num_groups), self.iteration_counter)

        if self.wandb_run is not None:
            metrics = {"score/min": float(sum(group_min_score) / self.num_groups)}
            for gi in range(self.num_groups):
                metrics[f"score/min_leg{gi}"] = group_min_score[gi]
                metrics[f"delay/best_leg{gi}"] = self.sim_params[group_min_env[gi], self.delay_idx].item()
            for i, name in enumerate(self.joint_order):
                me = group_min_env[self.joint_to_group[i]]
                metrics[f"armature/{name}"] = self.sim_params[me, self.armature_idx][i].item()
                metrics[f"viscous_friction/{name}"] = self.sim_params[me, self.viscous_friction_idx][i].item()
                metrics[f"static_friction/{name}"] = self.sim_params[me, self.static_friction_idx][i].item()
                metrics[f"dynamic_friction/{name}"] = self.sim_params[me, self.dynamic_friction_idx][i].item()
                metrics[f"bias/{name}"] = self.sim_params[me, self.bias_idx][i].item()
            # Best rollout's command/real/sim trajectory (each joint from its group's best env).
            # Merge into the same log call/step as the metrics (a second same-step log is dropped).
            metrics.update(
                self._best_trajectory_charts(
                    [group_min_env[self.joint_to_group[i]] for i in range(len(self.joint_order))]
                )
            )
            self.wandb_run.log(metrics, step=self.iteration_counter)

    # Max points per line in the W&B trajectory plots (the raw chirp is ~6k steps;
    # downsampling keeps the logged data small while staying visually faithful).
    _TRAJ_PLOT_MAX_POINTS = 400

    def _best_trajectory_charts(self, env_for_joint):
        """Build the W&B trajectory visualisation of the best rollout (command/real/sim).

        Returns ``{"trajectory/img": wandb.Image}`` to be merged into the iteration's single
        ``wandb_run.log(...)`` call (a second ``log`` at an already-used step is dropped by W&B).

        A small, downsampled, low-dpi PNG stacking all joints. Media panels render reliably and
        scrub across iterations, and the image is kept small (~50 kB) — lighter *and* more robust
        than ``wandb.plot.line_series`` custom charts, which log a full table per joint per
        iteration (~3x larger) and render unreliably when logged every step.

        ``env_for_joint[i]`` is the environment whose sim trajectory to use for joint ``i``.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import wandb

        t = self.traj_time.cpu().numpy()
        stride = max(1, len(t) // self._TRAJ_PLOT_MAX_POINTS)
        t_ds = t[::stride]
        n = len(self.joint_order)
        skip_t = float(t[self.score_skip_steps]) if 0 < self.score_skip_steps < len(t) else 0.0

        # Three clearly distinct styles so command/real/sim never blur together when overlapping.
        styles = {
            "command": dict(color="#2ca02c", linestyle="--", lw=1.0),  # green, dashed
            "real": dict(color="#1f77b4", linestyle="-", lw=1.2),  # blue, solid
            "sim": dict(color="#d62728", linestyle="-", lw=0.9, alpha=0.85),  # red, solid
        }

        fig, axes = plt.subplots(n, 1, figsize=(7.0, 1.5 * n), sharex=True, squeeze=False, dpi=70)
        for i, name in enumerate(self.joint_order):
            ax = axes[i][0]
            if skip_t > 0.0:
                ax.axvspan(t_ds[0], skip_t, color="grey", alpha=0.15, lw=0)
            ax.plot(t_ds, self.des_dof_pos[:, i].cpu().numpy()[::stride], label="command", **styles["command"])
            ax.plot(t_ds, self.real_dof_pos[:, i].cpu().numpy()[::stride], label="real", **styles["real"])
            env = int(env_for_joint[i])
            ax.plot(t_ds, self.sim_dof_pos_buffer[env, :, i].cpu().numpy()[::stride], label="sim", **styles["sim"])
            ax.set_ylabel(name.replace("single_leg_", ""), fontsize=7)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.legend(loc="upper right", fontsize=6, ncol=3)
        axes[-1][0].set_xlabel("time [s]", fontsize=7)
        fig.suptitle(f"iter {self.iteration_counter}  (first {skip_t:.2f}s excluded)", fontsize=8)
        fig.tight_layout()
        out = {"trajectory/img": wandb.Image(fig)}
        plt.close(fig)
        return out

    def save_checkpoint(self, mean, iteration, finished=False):
        if self.multi:
            # Assemble a representative best trajectory: each leg's joint columns
            # come from that leg's own best-scoring environment.
            best_traj = torch.zeros_like(self.sim_dof_pos_buffer[0])
            for gi in range(self.num_groups):
                lo = gi * self.block
                block_scores = self.scores_buffer[iteration, lo : lo + self.block]
                env = lo + torch.argmin(block_scores).item()
                jg = self.joint_groups[gi]
                best_traj[:, jg] = self.sim_dof_pos_buffer[env][:, jg]
            best_traj = best_traj.cpu()
        else:
            min_index = torch.argmin(self.scores_buffer[iteration, :])
            best_traj = self.sim_dof_pos_buffer[min_index].detach().clone().cpu()
        torch.save(best_traj, os.path.join(self.writer.log_dir, "best_trajectory.pt"))
        torch.save(mean, os.path.join(self.writer.log_dir, "mean_" + f"{iteration:03}" + ".pt"))
        if finished and self.save_optimization_process:
            torch.save(
                {
                    "params_buffer": self.sim_params_buffer,
                    "scores_buffer": self.scores_buffer,
                },
                os.path.join(self.writer.log_dir, "progress.pt"),
            )

    def close(self):
        self.writer.close()
