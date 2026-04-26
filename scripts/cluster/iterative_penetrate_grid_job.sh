#!/usr/bin/env bash
#$ -S /bin/bash
#$ -cwd
#$ -j y
#$ -r y
set -euo pipefail

TASK_ID=${SGE_TASK_ID:-1}
GRID_SCRIPT=${GRID_SCRIPT:?GRID_SCRIPT environment variable must be set}
TASK_MANIFEST=${TASK_MANIFEST:?TASK_MANIFEST environment variable must be set}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-python}

if [ -n "${RUN_WORKDIR:-}" ]; then
  cd "$RUN_WORKDIR"
fi

if [ -n "${ENV_SETUP_SCRIPT:-}" ]; then
  if [ ! -f "$ENV_SETUP_SCRIPT" ]; then
    echo "Environment setup script not found: $ENV_SETUP_SCRIPT" >&2
    exit 2
  fi
  # shellcheck disable=SC1090
  source "$ENV_SETUP_SCRIPT"
fi

if [ ! -f "$TASK_MANIFEST" ]; then
  echo "Task manifest not found: $TASK_MANIFEST" >&2
  exit 3
fi

CMD=(
  "$PYTHON_EXECUTABLE"
  "$GRID_SCRIPT"
  "--execution" "task"
  "--task-manifest" "$TASK_MANIFEST"
  "--task-index" "$TASK_ID"
)

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting iterative penetrate sweep task ${TASK_ID}"
echo "Workdir: ${PWD}"
echo "Manifest: ${TASK_MANIFEST}"
echo "Command: ${CMD[*]}"

"${CMD[@]}"
