# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0

"""Replay one episode with a given PACE parameter set and report the leg-wise score.

Scores a fitted (or hand-specified) parameter set on one robot without running CMA-ES. The
rollout and the score are computed exactly as in ``fit.py`` — same skip of the settling
transient, same ``(sim - real - bias)^2`` error, same per-leg joint grouping — and the
parameters are written through the same sinks (:mod:`pace_sim2real.optim.sim_params`), so a
score printed here is directly comparable to ``score/min_leg*`` in a fit's W&B history.

Parameters are resolved in increasing priority:

1. whatever the robot is configured with in the environment (so with no arguments this scores
   the shipped actuator config),
2. ``--wandb_run`` (a finished run's ``final_*`` summary, else the last logged values — which
   makes it work on a run that is still going) or ``--params`` (a ``mean_*.pt`` checkpoint),
3. explicit ``--armature`` / ``--coulomb`` / ``--viscous`` / ... overrides, applied to all joints.

Usage
-----
    # score a W&B fit on the data it was fitted to
    ./isaaclab.sh -p scripts/pace/check_params_score.py \
        --task Isaac-Pace-Delta-Quadruped-Nofoot-Backlash-v0 \
        --data_dir delta_quadruped_nofoot_15hz/chirp_data.pt \
        --wandb_run pace-nofoot-15hz-fitbacklash-simdamp --headless

    # same parameters, but step the physics differently (200 Hz actuator, 4 substeps)
    ... --sim_dt 0.005 --decimation 1 --num_substeps 4

.. note::
    The delay buffer lags per *environment*, not per joint, so a single robot carries one
    delay for all twelve motors. A per-leg fit assigns each leg its own delay, so when those
    disagree this script reports the spread and applies their mean; pass ``--delay`` to pin it.
"""

import argparse
import sys

from isaaclab_tasks.utils import (
    add_launcher_args,
    fold_preset_tokens,
    launch_simulation,
    resolve_task_config,
    setup_preset_cli,
)

parser = argparse.ArgumentParser(description="Score a PACE parameter set on one replay episode.")
parser.add_argument("--task", type=str, default="Isaac-Pace-Delta-Quadruped-Nofoot-Backlash-v0", help="Task name.")
parser.add_argument(
    "--data_dir", type=str, default=None, help="Override sim2real.data_dir (path under <project>/data)."
)
parser.add_argument("--num_envs", type=int, default=1, help="Robots to spawn; all get identical parameters.")
# -- parameter sources
parser.add_argument("--wandb_run", type=str, default=None, help="W&B run name or id to pull fitted parameters from.")
parser.add_argument("--wandb_project", type=str, default="pace_sim2real", help="W&B project.")
parser.add_argument("--wandb_entity", type=str, default=None, help="W&B entity (default: $WANDB_ENTITY).")
parser.add_argument("--params", type=str, default=None, help="Path to a mean_*.pt checkpoint written by the fit.")
# -- per-block scalar overrides, applied to every joint
parser.add_argument("--armature", type=float, default=None, help="Armature [kg·m²] for all joints.")
parser.add_argument("--coulomb", type=float, default=None, help="Coulomb friction [N·m] -> dof_frictionloss.")
parser.add_argument("--viscous", type=float, default=None, help="Viscous coefficient [N·m·s/rad].")
parser.add_argument("--bias", type=float, default=None, help="Encoder bias [rad].")
parser.add_argument("--delay", type=float, default=None, help="Action delay [sim steps]; truncated to int.")
parser.add_argument("--backlash", type=float, default=None, help="Total gear-play band [rad].")
# -- integration overrides
parser.add_argument("--sim_dt", type=float, default=None, help="Override sim.dt [s] (the actuator rate).")
parser.add_argument("--decimation", type=int, default=None, help="Override control decimation.")
parser.add_argument("--num_substeps", type=int, default=None, help="Override solver substeps per sim.dt.")
parser.add_argument(
    "--rescale_delay",
    action="store_true",
    help="Reinterpret the fitted delay as a wall-clock lag and convert it to the new sim.dt.",
)
parser.add_argument("--score_skip_s", type=float, default=1.5, help="Initial seconds excluded from the score.")
add_launcher_args(parser)
args_cli, hydra_args = setup_preset_cli(parser)
sys.argv = [sys.argv[0]] + fold_preset_tokens(hydra_args)

import gymnasium as gym  # noqa: E402
import pace_sim2real.tasks  # noqa: F401, E402
import torch  # noqa: E402
from pace_sim2real.optim.sim_params import PaceParamIndex, apply_sim_params  # noqa: E402
from pace_sim2real.utils import project_root  # noqa: E402
from tqdm import tqdm  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402

# Fitted sim.dt of the PACE environments; a delay is expressed in steps of this size.
_FIT_SIM_DT = 0.00125


def _physics_cfg(env_cfg):
    """Return the active solver cfg, whether or not the backend preset has been resolved yet.

    Before the environment is built ``sim.physics`` is a :class:`PresetCfg` holding the
    backend variants; afterwards it has been collapsed to the selected one.
    """
    physics = env_cfg.sim.physics
    return getattr(physics, "default", physics)


def _fetch_wandb_params(run_name: str, joint_order: list[str], num_groups: int) -> dict:
    """Pull a fitted parameter set from a W&B run.

    Prefers the ``final_*`` summary keys written when a fit finishes; falls back to the last
    logged value of each per-iteration metric, which W&B also keeps in the run summary, so a
    run that is still going can be scored.
    """
    import os

    import wandb

    entity = args_cli.wandb_entity or os.environ.get("WANDB_ENTITY")
    api = wandb.Api()
    path = run_name if run_name.count("/") == 2 else f"{entity}/{args_cli.wandb_project}/{run_name}"
    try:
        run = api.run(path)
    except Exception:
        # Not an id -- resolve by display name.
        matches = [r for r in api.runs(f"{entity}/{args_cli.wandb_project}") if r.name == run_name]
        if not matches:
            raise ValueError(f"No W&B run named or id'd '{run_name}' in {entity}/{args_cli.wandb_project}.") from None
        run = matches[0]
    summary = dict(run.summary)
    print(f"[INFO] W&B run '{run.name}' ({run.id}), state={run.state}")

    def block(final_key: str, live_key: str) -> list[float] | None:
        if all(f"{final_key}/{j}" in summary for j in joint_order):
            return [float(summary[f"{final_key}/{j}"]) for j in joint_order]
        if all(f"{live_key}/{j}" in summary for j in joint_order):
            print(f"[INFO]   '{final_key}/*' absent (run unfinished) -> using last logged '{live_key}/*'")
            return [float(summary[f"{live_key}/{j}"]) for j in joint_order]
        return None

    # Runs fitted before the block rename (2026-07-28) logged the Coulomb block under
    # "viscous_friction" and the viscous block under "static_friction"; accept both so
    # earlier fits stay replayable.
    out = {
        "armature": block("final_armature", "armature"),
        "coulomb": block("final_coulomb_friction", "coulomb_friction")
        or block("final_viscous_friction", "viscous_friction"),
        "viscous": block("final_viscous_damping", "viscous_damping")
        or block("final_static_friction", "static_friction"),
        "bias": block("final_bias", "bias"),
        "backlash": block("final_backlash", "backlash"),
    }
    # Per-leg delays are reported even when the CMA mean (final_delay) is used, because a
    # single robot cannot reproduce them: the delay buffer lags per environment, so in the fit
    # each leg's own delay only ever acted in that leg's block of environments.
    per_leg = [summary[f"delay/best_leg{g}"] for g in range(num_groups) if f"delay/best_leg{g}" in summary]
    if per_leg:
        applied = [int(d) for d in per_leg]
        print(f"[INFO]   fit's per-leg delays: {[round(float(d), 2) for d in per_leg]} -> applied as {applied}")
        if len(set(applied)) > 1:
            print(
                "[WARN]   the fit used different delays per leg; one robot has a single delay buffer, so legs"
                " whose delay differs from the one applied here will score worse than they did in the fit."
            )
    out["delay"] = [float(summary["final_delay"])] if "final_delay" in summary else [float(d) for d in per_leg] or None
    return out


def main():
    env_cfg, _ = resolve_task_config(args_cli.task, "env_cfg_entry_point")

    with launch_simulation(env_cfg, args_cli):
        env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if args_cli.data_dir is not None:
            env_cfg.sim2real.data_dir = args_cli.data_dir
        # -- integration overrides. sim.dt is the actuator rate; sim.dt * decimation must stay
        # equal to the data period so one recorded row is still replayed per env.step().
        if args_cli.sim_dt is not None:
            env_cfg.sim.dt = args_cli.sim_dt
        if args_cli.decimation is not None:
            env_cfg.decimation = args_cli.decimation
        if args_cli.num_substeps is not None:
            _physics_cfg(env_cfg).num_substeps = args_cli.num_substeps

        env = gym.make(args_cli.task, cfg=env_cfg)
        articulation = env.unwrapped.scene["robot"]
        device = env.unwrapped.device
        joint_order = env_cfg.sim2real.joint_order
        n = len(joint_order)
        groups = env_cfg.sim2real.joint_groups or [list(range(n))]
        fit_backlash = env_cfg.sim2real.fit_backlash
        joint_ids = torch.tensor([articulation.joint_names.index(j) for j in joint_order], device=device)
        backlash_joint_ids = None
        if fit_backlash:
            backlash_joint_ids = torch.tensor(
                [articulation.joint_names.index(j.replace("_motor", "_backlash")) for j in joint_order], device=device
            )

        data = torch.load(project_root() / "data" / env_cfg.sim2real.data_dir)
        time_data = data["time"].to(device)
        target_dof_pos = data["des_dof_pos"].to(device)
        measured_dof_pos = data["dof_pos"].to(device)
        time_steps = time_data.shape[0]
        # First-interval dt, matching fit.py, so score_skip_steps is identical to the fit's.
        data_dt = (time_data[1] - time_data[0]).item()
        score_skip_steps = min(time_steps - 1, round(args_cli.score_skip_s / data_dt))
        # Recorded timestamps jitter, so the *mean* period is what the control rate must match.
        mean_data_dt = ((time_data[-1] - time_data[0]) / (time_steps - 1)).item()

        control_dt = env_cfg.sim.dt * env_cfg.decimation
        substeps = _physics_cfg(env_cfg).num_substeps
        print(
            f"[INFO] sim.dt={env_cfg.sim.dt * 1e3:.3f} ms (actuator @ {1 / env_cfg.sim.dt:.0f} Hz), "
            f"decimation={env_cfg.decimation}, substeps={substeps} "
            f"(solver @ {substeps / env_cfg.sim.dt:.0f} Hz), control={1 / control_dt:.0f} Hz"
        )
        if abs(control_dt - mean_data_dt) > 0.02 * mean_data_dt:
            print(
                f"[WARN] control period {control_dt * 1e3:.3f} ms differs from the recording's mean period"
                f" {mean_data_dt * 1e3:.3f} ms; one row is still replayed per step, so the trajectory is played"
                " back at the wrong speed and the score is NOT comparable to the fit."
            )

        env.reset()

        # ---- assemble the parameter vector -------------------------------------------
        idx = PaceParamIndex(n, num_extra=0, fit_backlash=fit_backlash)
        sim_params = torch.zeros((args_cli.num_envs, idx.num_params), device=device)

        # 1. defaults: whatever the robot currently carries.
        sim_params[:, idx.armature] = articulation.data.joint_armature.torch[:, joint_ids]
        sim_params[:, idx.coulomb_friction] = articulation.data.joint_friction_coeff.torch[:, joint_ids]
        sim_params[:, idx.viscous_damping] = articulation.data.joint_passive_damping.torch[:, joint_ids]
        motor_act = next(a for a in articulation.actuators.values() if hasattr(a, "update_encoder_bias"))
        sim_params[:, idx.bias] = motor_act.encoder_bias.to(device)
        sim_params[:, idx.delay] = motor_act.position_targets_delay_buffer.time_lags.float().to(device)
        if fit_backlash:
            upper = articulation.data.joint_pos_limits_upper.torch[:, backlash_joint_ids]
            sim_params[:, idx.backlash] = 2.0 * upper

        # 2. a fitted set from W&B or a checkpoint.
        source = None
        if args_cli.wandb_run is not None:
            source = _fetch_wandb_params(args_cli.wandb_run, joint_order, len(groups))
        elif args_cli.params is not None:
            loaded = torch.load(args_cli.params, map_location=device)
            vec = loaded["params"] if isinstance(loaded, dict) else loaded
            if vec.numel() < idx.num_params:
                raise ValueError(
                    f"{args_cli.params} holds {vec.numel()} parameters but this task's layout needs"
                    f" {idx.num_params}. Was it fitted with a different fit_backlash setting?"
                )
            sim_params[:] = vec[: idx.num_params].to(device).unsqueeze(0)
        if source is not None:
            for key, sl in (
                ("armature", idx.armature),
                ("coulomb", idx.coulomb_friction),
                ("viscous", idx.viscous_damping),
                ("bias", idx.bias),
            ):
                if source.get(key) is not None:
                    sim_params[:, sl] = torch.tensor(source[key], device=device)
            if fit_backlash and source.get("backlash") is not None:
                sim_params[:, idx.backlash] = torch.tensor(source["backlash"], device=device)
            if source.get("delay") is not None:
                delays = source["delay"]
                if len(delays) > 1 and max(delays) - min(delays) > 0.5:
                    print(
                        f"[WARN] per-leg delays disagree ({', '.join(f'{d:.2f}' for d in delays)} sim steps). One"
                        " robot has a single delay buffer, so their mean is used; pass --delay to pin it."
                    )
                sim_params[:, idx.delay] = sum(delays) / len(delays)

        # 3. explicit overrides.
        for value, sl in (
            (args_cli.armature, idx.armature),
            (args_cli.coulomb, idx.coulomb_friction),
            (args_cli.viscous, idx.viscous_damping),
            (args_cli.bias, idx.bias),
        ):
            if value is not None:
                sim_params[:, sl] = value
        if args_cli.backlash is not None and fit_backlash:
            sim_params[:, idx.backlash] = args_cli.backlash
        if args_cli.delay is not None:
            sim_params[:, idx.delay] = args_cli.delay
        elif args_cli.rescale_delay and abs(env_cfg.sim.dt - _FIT_SIM_DT) > 1e-12:
            lag_s = sim_params[:, idx.delay] * _FIT_SIM_DT
            sim_params[:, idx.delay] = lag_s / env_cfg.sim.dt
            print(f"[INFO] Delay rescaled to the new sim.dt: {lag_s[0].item() * 1e3:.2f} ms preserved.")

        applied_delay = int(sim_params[0, idx.delay].item())
        print(
            f"[INFO] Delay: {sim_params[0, idx.delay].item():.3f} steps -> applied as {applied_delay}"
            f" = {applied_delay * env_cfg.sim.dt * 1e3:.2f} ms"
        )

        apply_sim_params(
            articulation,
            sim_params,
            idx,
            joint_ids,
            env_ids=torch.arange(args_cli.num_envs, device=device),
            backlash_joint_ids=backlash_joint_ids,
        )

        # ---- replay, scoring exactly as fit.py does ----------------------------------
        bias = sim_params[:, idx.bias]
        sq_err = torch.zeros((args_cli.num_envs, n), device=device)
        scored_steps = 0
        for counter in tqdm(range(time_steps), desc="Replay", unit="step"):
            with torch.inference_mode():
                sim_pos = articulation.data.joint_pos.torch[:, joint_ids]
                real = measured_dof_pos[counter, :].unsqueeze(0).repeat(args_cli.num_envs, 1)
                if counter >= score_skip_steps:
                    sq_err += torch.square(sim_pos - real - bias)
                    scored_steps += 1
                env.step(target_dof_pos[counter, :].unsqueeze(0).repeat(args_cli.num_envs, 1))

        mse = (sq_err / max(1, scored_steps))[0]  # env 0; all envs are identical
        leg_names = ["LF", "RF", "LH", "RH"]
        print(
            f"\n=== leg-wise score ({scored_steps}/{time_steps} steps scored, first {args_cli.score_skip_s} s skipped)"
        )
        print(f"{'leg':<5} {'score':>12}   per-joint RMSE [deg]")
        for gi, jg in enumerate(groups):
            name = leg_names[gi] if gi < len(leg_names) and len(groups) == 4 else f"g{gi}"
            rmse = [f"{joint_order[j].split('_')[-2]}={torch.rad2deg(mse[j].sqrt()).item():.3f}" for j in jg]
            print(f"{name:<5} {mse[jg].sum().item():>12.6e}   {'  '.join(rmse)}")
        print(f"{'mean':<5} {sum(mse[jg].sum().item() for jg in groups) / len(groups):>12.6e}")
        print(f"{'all':<5} {mse.sum().item():>12.6e}   (sum over all {n} joints)")

        if not torch.isfinite(mse).all():
            print("[WARN] non-finite score: the replay diverged.")

        env.close()


if __name__ == "__main__":
    main()
