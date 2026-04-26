#!/usr/bin/env bash
set -euo pipefail

REPO_BASE="/home/sporter/synergistic_Y90/SIRF-SIMIND-Connection"
DATA_ROOT="/home/sporter/synergistic_Y90/prepared_data"
CLUSTER_ENV_SETUP_SCRIPT="${CLUSTER_ENV_SETUP_SCRIPT:-$REPO_BASE/scripts/cluster/simind_env_sge.sh}"

cd "$REPO_BASE"

cmd=(
  python scripts/run_iterative_penetrate_grid.py
  --execution cluster
  --base-config project/config/cluster_sge.yaml
  --output-root output/grid_additive_modes
  --mpi-procs 1
  --cluster-memory 48G
  --skip-completed
  --dataset "$DATA_ROOT/phantom_data/nema_phantom_data/SPECT"
  --dataset "$DATA_ROOT/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140"
  --dataset "$DATA_ROOT/phantom_data/manc_nema_phantom_data/SPECT"
  --dataset "$DATA_ROOT/oxford_patient_data/sirt3/SPECT"
)

if [ -n "$CLUSTER_ENV_SETUP_SCRIPT" ]; then
  cmd+=(--cluster-env-setup-script "$CLUSTER_ENV_SETUP_SCRIPT")
fi

if [ "${DRY_RUN:-0}" = "1" ]; then
  cmd+=(--dry-run)
else
  cmd+=(--submit)
fi

cmd+=("$@")

printf 'Running:'
printf ' %q' "${cmd[@]}"
printf '\n'

"${cmd[@]}"
