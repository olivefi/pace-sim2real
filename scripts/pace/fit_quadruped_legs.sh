#!/usr/bin/env bash
# © 2025 ETH Zurich, Robotic Systems Lab
# Author: Filip Bjelonic
# Licensed under the Apache License 2.0
#
# Fit all four delta legs independently on the single-leg aluminium PACE env,
# one process per leg. Each process replays that leg's three motor columns
# (split out of the quadruped HW recording by prepare_delta_quadruped_leg_data.py)
# on the cheap 3-DOF sim. Run them in parallel across GPUs, or sequentially on one.
#
# Prerequisite:
#   ./isaaclab.sh -p scripts/pace/prepare_delta_quadruped_leg_data.py
# produces data/delta_quadruped_leg/{LF,RF,LH,RH}.pt
#
# Usage:
#   scripts/pace/fit_quadruped_legs.sh                 # 4 legs in parallel, GPU per leg (cuda:0..3)
#   PARALLEL=0 scripts/pace/fit_quadruped_legs.sh      # 4 legs one after another on cuda:0
#   NUM_ENVS=512 PARALLEL=0 WANDB=1 scripts/pace/fit_quadruped_legs.sh   # sequential, single GPU, W&B
#
# Env vars:
#   PARALLEL  1 (default) = launch all legs at once; 0 = sequential.
#   NUM_ENVS  CMA-ES population per leg (default 512).
#   MAX_ITER  optional CMA-ES iteration override (default: env cfg value).
#   GPUS      space/comma-separated device list, one per leg when parallel
#             (default "cuda:0 cuda:1 cuda:2 cuda:3"; fewer entries are cycled).
#   WANDB     1 = enable Weights & Biases logging (per-leg run "pace-leg-<LEG>").
#   WANDB_PROJECT  W&B project (default "pace_sim2real").
#   WANDB_ENV_FILE optional file to source for WANDB_API_KEY / WANDB_ENTITY
#             (e.g. docker/cluster/.env.cluster).

set -euo pipefail

# Resolve repo root (this script lives at <root>/source/pace-sim2real/scripts/pace).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
ISAACLAB="${REPO_ROOT}/isaaclab.sh"
FIT="${SCRIPT_DIR}/fit.py"

TASK="Isaac-Pace-Delta-Quadruped-Leg-v0"
LEGS=(LF RF LH RH)
PARALLEL="${PARALLEL:-1}"
NUM_ENVS="${NUM_ENVS:-512}"
MAX_ITER="${MAX_ITER:-}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-pace_sim2real}"
read -r -a GPU_LIST <<< "${GPUS:-cuda:0 cuda:1 cuda:2 cuda:3}"

# Optionally source W&B credentials (WANDB_API_KEY / WANDB_ENTITY).
if [[ -n "${WANDB_ENV_FILE:-}" && -f "${WANDB_ENV_FILE}" ]]; then
  set -a; source "${WANDB_ENV_FILE}"; set +a
fi

extra_args=()
[[ -n "${MAX_ITER}" ]] && extra_args+=(--max_iteration "${MAX_ITER}")

run_leg() {
  local leg="$1" device="$2"
  echo "[fit_quadruped_legs] leg=${leg} device=${device} num_envs=${NUM_ENVS} wandb=${WANDB}"
  local wandb_args=()
  [[ "${WANDB}" == "1" ]] && wandb_args=(--wandb --wandb_project "${WANDB_PROJECT}" --wandb_run_name "pace-leg-${leg}")
  "${ISAACLAB}" -p "${FIT}" \
    --task "${TASK}" \
    --num_envs "${NUM_ENVS}" \
    --device "${device}" \
    --data_dir "delta_quadruped_leg/${leg}.pt" \
    --robot_name "delta_quadruped_leg/${leg}" \
    --headless \
    "${wandb_args[@]}" \
    "${extra_args[@]}"
}

pids=()
for i in "${!LEGS[@]}"; do
  leg="${LEGS[$i]}"
  device="${GPU_LIST[$(( i % ${#GPU_LIST[@]} ))]}"
  if [[ "${PARALLEL}" == "1" ]]; then
    run_leg "${leg}" "${device}" &
    pids+=("$!")
  else
    run_leg "${leg}" "${GPU_LIST[0]}"
  fi
done

if [[ "${PARALLEL}" == "1" ]]; then
  status=0
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  exit "${status}"
fi
