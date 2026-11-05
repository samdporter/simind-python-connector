#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/docker/compose.yaml"
OUT_DIR="$ROOT_DIR/output/validation"

ENGINE=""
FEATURES=()

usage() {
    cat <<'USAGE'
Usage: bash scripts/run_validation.sh --engine v8 [--feature NAME]...

Runs the SIMIND v8 release validation tests and records the numbers in
output/validation/v8-linux.jsonl. The cases run in Docker (python image,
then sirf image, then pytomography image).
--feature NAME runs tests/validation/test_NAME.py and replaces only that
feature's records; without it the engine file is rebuilt from scratch.
SIMIND v9 is experimental and unvalidated in this release; it is planned for
2.1.0, so the release runner accepts only v8.
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
    v8) ;;
    v9|v9-macos)
        echo "SIMIND v9 is experimental and unvalidated in this release;" \
            "release validation runs v8 only. v9 is planned for 2.1.0." >&2
        exit 2
        ;;
    *)
        echo "--engine must be v8 (SIMIND v9 is experimental/unvalidated)" >&2
        usage
        exit 2
        ;;
esac

if [[ ${#FEATURES[@]} -eq 0 ]]; then
    TARGETS=(tests/validation)
else
    TARGETS=()
    for feature in "${FEATURES[@]}"; do
        TARGETS+=("tests/validation/test_${feature}.py")
    done
fi

RECORD_FILE="$OUT_DIR/v8-linux.jsonl"
mkdir -p "$OUT_DIR"
if [[ ${#FEATURES[@]} -eq 0 ]]; then
    : > "$RECORD_FILE"
elif [[ -f "$RECORD_FILE" ]]; then
    for feature in "${FEATURES[@]}"; do
        grep -v "\"feature\": \"$feature\"" "$RECORD_FILE" > "$RECORD_FILE.tmp" || true
        mv "$RECORD_FILE.tmp" "$RECORD_FILE"
    done
fi

STATUS=0

# pytest exits 5 when nothing was selected, which is not a failure here
collect_status() {
    if [[ "$1" -ne 5 ]]; then
        STATUS=1
    fi
}

SIMIND_DIR=/workspace/simind
SIMIND_VERSION=8

run_in_docker() {
    local service="$1" marker="$2"
    shift 2
    local user_args=()
    if [[ "$service" == python ]]; then
        user_args=(--user "$(id -u):$(id -g)" -e HOME=/tmp)
    fi
    DOCKER_DEFAULT_PLATFORM=linux/amd64 docker compose -f "$COMPOSE_FILE" run --rm \
        ${user_args[@]+"${user_args[@]}"} \
        -e SIMIND_VALIDATION_ENGINE="v8-linux" \
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
