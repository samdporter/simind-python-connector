#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"
MACOS_V9_SOURCE="${SIMIND_MACOS_V9_DIR:-/Users/samd-work/Projects/simind-python-connector/tmp/simind_macos_v90}"
# the macOS v9 install ships no phantom.zub; the library phantoms take it from the v9 Linux tree
LIBRARY_TABLE_ROOT="${SIMIND_MACOS_V9_TABLE_ROOT:-$ROOT_DIR/tmp/engines/v9}"
# SIMIND rejects an SMC_DIR of 100 characters or more, so it is reached through a short link
MACOS_V9_DIR="${SIMIND_MACOS_V9_LINK:-/tmp/simind-v9-macos}"
OUT_DIR="$ROOT_DIR/output/validation"

ENGINE=""
FEATURES=()

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_validation.sh --engine v8|v9|v9-macos [--feature NAME]...

Runs the SIMIND validation tests and records the numbers in
output/validation/<engine>.jsonl. v8 and v9 run in Docker (python image,
then sirf image, then pytomography image); v9-macos runs on the host with
the native macOS binary.
--feature NAME runs tests/validation/test_NAME.py and replaces only that
feature's records; without it the engine file is rebuilt from scratch.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --engine)
            ENGINE="${2:?--engine requires an argument}"
            shift
            ;;
        --feature)
            FEATURES+=("${2:?--feature requires an argument}")
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

case "$ENGINE" in
    v8|v9|v9-macos) ;;
    *) echo "--engine must be v8, v9 or v9-macos" >&2; usage; exit 2 ;;
esac

if [[ ${#FEATURES[@]} -eq 0 ]]; then
    TARGETS=(tests/validation)
else
    TARGETS=()
    for feature in "${FEATURES[@]}"; do
        TARGETS+=("tests/validation/test_${feature}.py")
    done
fi

if [[ "$ENGINE" == v9-macos ]]; then
    RECORD_FILE="$OUT_DIR/v9-darwin.jsonl"
else
    RECORD_FILE="$OUT_DIR/$ENGINE-linux.jsonl"
fi
mkdir -p "$OUT_DIR"
if [[ ${#FEATURES[@]} -eq 0 ]]; then
    : > "$RECORD_FILE"
elif [[ -f "$RECORD_FILE" ]]; then
    for feature in "${FEATURES[@]}"; do
        grep -v "\"feature\": \"$feature\"" "$RECORD_FILE" > "$RECORD_FILE.tmp" || true
        mv "$RECORD_FILE.tmp" "$RECORD_FILE"
    done
fi

PYTHON="${PYTHON:-$ROOT_DIR/../../../.venv/bin/python}"
STATUS=0

# pytest exits 5 when nothing was selected, which is not a failure here
collect_status() {
    if [[ "$1" -ne 5 ]]; then
        STATUS=1
    fi
}

if [[ "$ENGINE" == v9-macos ]]; then
    [[ -e "$MACOS_V9_DIR" ]] || ln -s "$MACOS_V9_SOURCE" "$MACOS_V9_DIR"
    SIMIND_VALIDATION_ENGINE=v9-darwin \
    SIMIND_VALIDATION_OUT="$OUT_DIR" \
    SIMIND_VERSION=9 \
    SIMIND_ROOT="$LIBRARY_TABLE_ROOT" \
    SIMIND_BIN="$MACOS_V9_DIR/simind" \
    SMC_DIR="$MACOS_V9_DIR/smc_dir/" \
    PATH="$MACOS_V9_DIR:$PATH" \
    PYTHONPATH="$ROOT_DIR" \
        "$PYTHON" -m pytest -m "validation and macos_v9" "${TARGETS[@]}" || collect_status $?
    exit "$STATUS"
fi

case "$ENGINE" in
    v8) SIMIND_DIR=/workspace/simind; SIMIND_VERSION=8 ;;
    v9) SIMIND_DIR=/workspace/tmp/engines/v9; SIMIND_VERSION=9 ;;
esac

run_in_docker() {
    local service="$1" marker="$2"
    shift 2
    local user_args=()
    if [[ "$service" == python ]]; then
        user_args=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
    fi
    DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose -f "$COMPOSE_FILE" run --rm \
        ${user_args[@]+"${user_args[@]}"} \
        -e SIMIND_VALIDATION_ENGINE="$ENGINE-linux" \
        -e SIMIND_VALIDATION_OUT=/workspace/output/validation \
        -e SIMIND_VERSION="$SIMIND_VERSION" \
        -e SIMIND_ROOT="$SIMIND_DIR" \
        -e SIMIND_BIN="$SIMIND_DIR/simind" \
        -e SIMIND_MPI_BIN="$SIMIND_DIR/simind_mpi" \
        -e SIMIND_SMC_DIR="$SIMIND_DIR/smc_dir/" \
        -e SIMIND_DATA_DIR="$SIMIND_DIR/smc_dir/" \
        -e SMC_DIR="$SIMIND_DIR/smc_dir/" \
        "$@" \
        "$service" bash -lc "export PATH=$SIMIND_DIR:\$PATH; python -m pytest -m '$marker' ${TARGETS[*]}" || collect_status $?
}

run_in_docker python "validation and not requires_sirf and not requires_pytomography"
run_in_docker sirf "validation and requires_sirf"
run_in_docker pytomography "validation and requires_pytomography"

exit "$STATUS"
