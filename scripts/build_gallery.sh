#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"

BACKENDS=()
SIMIND_VERSION=8

usage() {
    cat <<'USAGE'
Usage: bash scripts/build_gallery.sh [--backend B]... [--simind-version 8|9]

Builds the gallery once per backend in its container, executing only that
backend's examples, then checks that every example of the built backends has current outputs.
Backends: python, stir, sirf, pytomography (default: all, in that order).
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backend)
            BACKENDS+=("${2:?--backend requires an argument}")
            shift
            ;;
        --simind-version)
            SIMIND_VERSION="${2:?--simind-version requires an argument}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 2
            ;;
    esac
    shift
done

if [[ ${#BACKENDS[@]} -eq 0 ]]; then
    BACKENDS=(python stir sirf pytomography)
fi

case "$SIMIND_VERSION" in
    8) SIMIND_DIR=/workspace/simind ;;
    9) SIMIND_DIR=/workspace/tmp/engines/v9 ;;
    *) echo "--simind-version must be 8 or 9" >&2; exit 2 ;;
esac

for backend in "${BACKENDS[@]}"; do
    case "$backend" in
        python|stir|sirf|pytomography) ;;
        *) echo "Unknown backend: $backend" >&2; exit 2 ;;
    esac
    user_args=()
    if [[ "$backend" == python ]]; then
        user_args=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
    fi
    echo "[gallery] $backend (SIMIND v$SIMIND_VERSION)"
    DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose -f "$COMPOSE_FILE" run --rm \
        "${user_args[@]}" \
        -e SIMIND_GALLERY_BACKEND="$backend" \
        -e SIMIND_EXAMPLES_OUTPUT=/workspace/output \
        -e SIMIND_CONNECTOR_Y90_DATA=/workspace/tmp/y90_data \
        -e SIMIND_VERSION="$SIMIND_VERSION" \
        -e SIMIND_ROOT="$SIMIND_DIR" \
        -e SIMIND_BIN="$SIMIND_DIR/simind" \
        -e SIMIND_MPI_BIN="$SIMIND_DIR/simind_mpi" \
        -e SIMIND_SMC_DIR="$SIMIND_DIR/smc_dir/" \
        -e SIMIND_DATA_DIR="$SIMIND_DIR/smc_dir/" \
        "$backend" bash -lc "python -m sphinx -b html docs docs/_build/html"
done

DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose -f "$COMPOSE_FILE" run --rm \
    --user "$(id -u):$(id -g)" -e HOME=/tmp \
    python bash -lc "python scripts/check_gallery_current.py ${BACKENDS[*]/#/--backend }"
