#!/usr/bin/env python3
"""
Batch sweep runner for iterative_penetrate_scatter.py.

Supports three execution modes:
- local: run the prepared sweep directly (serially or with limited parallelism)
- cluster: stage configs/manifest and optionally submit an SGE array job
- task: execute a single staged task from a manifest (used by the cluster job script)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import yaml


DEFAULT_DATASETS = [
    "/home/storage/prepared_data/phantom_data/nema_phantom_data/SPECT",
    "/home/storage/prepared_data/phantom_data/anthropomorphic_phantom_data/SPECT/phantom_140",
    "/home/storage/prepared_data/phantom_data/manc_nema_phantom_data/SPECT",
]

DEFAULT_ADDITIVE_MODES = [
    "initial_scatter",
    "update_scatter",
    "initial_scatter_plus_residual",
    "update_scatter_plus_residual",
]

DEFAULT_BETAS = [0.0, 0.001, 0.01, 0.1]
DEFAULT_GAUSS_OPTIONS = [[6.9, 6.9, 6.9], [0.0, 0.0, 0.0]]
DEFAULT_SCATTER_ALPHAS = [0.2]
DEFAULT_RESIDUAL_ALPHAS = [0.2, 1.0]

DEFAULT_INTEL_MPI_ROOT = Path("/opt/intel/oneapi/mpi")
DEFAULT_LOCAL_WORKERS = 1
DEFAULT_CLUSTER_JOB_NAME = "iter_pen_grid"
DEFAULT_CLUSTER_RUNTIME = "48:00:00"
DEFAULT_CLUSTER_MEMORY = "16G"


@dataclass(frozen=True)
class RunSpec:
    data_dir: Path
    dataset_name: str
    additive_mode: str
    scatter_alpha: float
    residual_alpha: float
    beta: float
    gauss_fwhm: list[float]
    output_dir: Path
    config_path: Path
    log_path: Path
    scatter_init_path: Path


@dataclass(frozen=True)
class PreparedTask:
    task_number: int
    data_dir: Path
    dataset_name: str
    additive_mode: str
    scatter_alpha: float
    residual_alpha: float
    beta: float
    gauss_fwhm: list[float]
    output_dir: Path
    config_path: Path
    log_path: Path
    scatter_init_path: Path
    cmd: list[str]


@dataclass(frozen=True)
class TaskResult:
    task: PreparedTask
    status: str
    message: str


def set_nested(cfg: dict, keys: list[str], value):
    cur = cfg
    for key in keys[:-1]:
        if key not in cur or not isinstance(cur[key], dict):
            cur[key] = {}
        cur = cur[key]
    cur[keys[-1]] = value


def slug_float(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def parse_gauss_fwhm(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--gauss-fwhm expects exactly three comma-separated values, "
            "e.g. 6.9,6.9,6.9"
        )
    try:
        return [float(part) for part in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid --gauss-fwhm value {value!r}: {exc}"
        ) from exc


def dataset_name_for(path: Path) -> str:
    if path.name.lower() == "spect" and path.parent.name:
        return path.parent.name
    return path.name


def ensure_unique_dataset_names(paths: Iterable[Path]) -> dict[Path, str]:
    counts: dict[str, int] = {}
    out: dict[Path, str] = {}
    for path in paths:
        base = dataset_name_for(path)
        idx = counts.get(base, 0)
        counts[base] = idx + 1
        out[path] = base if idx == 0 else f"{base}_{idx + 1}"
    return out


def mode_uses_scatter(additive_mode: str) -> bool:
    return additive_mode in ("update_scatter", "update_scatter_plus_residual")


def mode_uses_residual(additive_mode: str) -> bool:
    return additive_mode in (
        "initial_scatter_plus_residual",
        "update_scatter_plus_residual",
    )


def _version_key(path: Path) -> tuple[int, ...]:
    key = []
    for token in path.name.split("."):
        if token.isdigit():
            key.append(int(token))
        else:
            key.append(-1)
    return tuple(key)


def find_intel_mpi_install(root: Path = DEFAULT_INTEL_MPI_ROOT) -> Path | None:
    if not root.exists():
        return None

    candidates: list[Path] = []
    for version_dir in root.glob("2021.*"):
        lib_dir = version_dir / "lib"
        bin_dir = version_dir / "bin"
        if not lib_dir.exists() or not bin_dir.exists():
            continue
        if not (bin_dir / "mpirun").exists():
            continue
        if not (lib_dir / "libmpi.so.12").exists():
            continue
        if not (lib_dir / "libmpifort.so.12").exists():
            continue
        candidates.append(version_dir)

    if not candidates:
        return None

    candidates.sort(key=_version_key, reverse=True)
    return candidates[0]


def build_run_env_prepends(
    args: argparse.Namespace,
) -> tuple[dict[str, list[str]], list[str]]:
    prepends: dict[str, list[str]] = {}
    notes: list[str] = []

    if int(args.mpi_procs) <= 1:
        return prepends, notes

    mpi_bin_dir = args.mpi_bin_dir
    mpi_lib_dir = args.mpi_lib_dir

    if not args.disable_auto_intel_mpi_env and (
        mpi_bin_dir is None or mpi_lib_dir is None
    ):
        auto_install = find_intel_mpi_install()
        if auto_install is not None:
            if mpi_bin_dir is None:
                mpi_bin_dir = auto_install / "bin"
            if mpi_lib_dir is None:
                mpi_lib_dir = auto_install / "lib"

    if mpi_bin_dir is not None:
        mpi_bin_dir = mpi_bin_dir.expanduser().resolve()
        if not mpi_bin_dir.exists():
            raise FileNotFoundError(f"MPI bin dir not found: {mpi_bin_dir}")
        prepends.setdefault("PATH", []).append(str(mpi_bin_dir))
        notes.append(f"PATH+={mpi_bin_dir}")

    if mpi_lib_dir is not None:
        mpi_lib_dir = mpi_lib_dir.expanduser().resolve()
        if not mpi_lib_dir.exists():
            raise FileNotFoundError(f"MPI lib dir not found: {mpi_lib_dir}")
        prepends.setdefault("LD_LIBRARY_PATH", []).append(str(mpi_lib_dir))
        notes.append(f"LD_LIBRARY_PATH+={mpi_lib_dir}")

    return prepends, notes


def apply_env_prepends(env: dict[str, str], prepends: dict[str, list[str]]) -> None:
    for var_name, paths in prepends.items():
        cleaned = [path for path in paths if path]
        if not cleaned:
            continue
        existing = env.get(var_name, "")
        env[var_name] = (
            ":".join([*cleaned, existing]) if existing else ":".join(cleaned)
        )


def build_runs(
    output_root: Path,
    datasets: list[Path],
    additive_modes: list[str],
    scatter_alphas: list[float],
    residual_alphas: list[float],
    betas: list[float],
    gauss_options: list[list[float]],
    initial_additive_name: str,
) -> list[RunSpec]:
    names = ensure_unique_dataset_names(datasets)
    runs: list[RunSpec] = []
    for data_dir in datasets:
        dataset_name = names[data_dir]
        scatter_init_path = data_dir / initial_additive_name
        for additive_mode in additive_modes:
            scatter_candidates = (
                scatter_alphas
                if mode_uses_scatter(additive_mode)
                else [scatter_alphas[0]]
            )
            residual_candidates = (
                residual_alphas
                if mode_uses_residual(additive_mode)
                else [residual_alphas[0]]
            )
            for scatter_alpha in scatter_candidates:
                scatter_alpha_label = (
                    f"scatter_alpha_{slug_float(scatter_alpha)}"
                    if mode_uses_scatter(additive_mode)
                    else "scatter_alpha_na"
                )
                for residual_alpha in residual_candidates:
                    residual_alpha_label = (
                        f"residual_alpha_{slug_float(residual_alpha)}"
                        if mode_uses_residual(additive_mode)
                        else "residual_alpha_na"
                    )
                    for beta in betas:
                        beta_label = f"beta_{slug_float(beta)}"
                        for gauss in gauss_options:
                            gauss_label = "gauss_" + "_".join(
                                slug_float(x) for x in gauss
                            )
                            run_output = (
                                output_root
                                / dataset_name
                                / additive_mode
                                / scatter_alpha_label
                                / residual_alpha_label
                                / beta_label
                                / gauss_label
                            )
                            runs.append(
                                RunSpec(
                                    data_dir=data_dir,
                                    dataset_name=dataset_name,
                                    additive_mode=additive_mode,
                                    scatter_alpha=float(scatter_alpha),
                                    residual_alpha=float(residual_alpha),
                                    beta=float(beta),
                                    gauss_fwhm=[float(x) for x in gauss],
                                    output_dir=run_output,
                                    config_path=run_output / "run_config.yaml",
                                    log_path=run_output / "run.log",
                                    scatter_init_path=scatter_init_path,
                                )
                            )
    return runs


def validate_run_inputs(
    run: RunSpec,
    measured_name: str,
) -> None:
    if not run.scatter_init_path.exists():
        raise FileNotFoundError(f"missing initial additive: {run.scatter_init_path}")

    measured_path = run.data_dir / measured_name
    if not measured_path.exists():
        raise FileNotFoundError(f"missing measured data: {measured_path}")

    has_initial = (run.data_dir / "initial_image.hv").exists()
    has_template = (run.data_dir / "template_image.hv").exists()
    if not (has_initial or has_template):
        raise FileNotFoundError(
            f"missing both initial_image.hv and template_image.hv in {run.data_dir}"
        )


def build_task_command(
    run: RunSpec,
    args: argparse.Namespace,
    iterative_script: Path,
) -> list[str]:
    python_executable = (
        str(args.cluster_python_executable)
        if args.execution == "cluster"
        else sys.executable
    )
    cmd = [
        python_executable,
        str(iterative_script),
        "--config",
        str(run.config_path),
        "--data-dir",
        str(run.data_dir),
        "--output-dir",
        str(run.output_dir),
        "--additive-mode",
        run.additive_mode,
        "--additive-alpha",
        str(float(run.scatter_alpha)),
        "--residual-alpha",
        str(float(run.residual_alpha)),
        "--initial-additive-path",
        str(run.scatter_init_path),
        "--mpi-procs",
        str(int(args.mpi_procs)),
    ]
    if args.num_subsets is not None:
        cmd.extend(["--num-subsets", str(int(args.num_subsets))])
    if args.num_epochs is not None:
        cmd.extend(["--num-epochs", str(int(args.num_epochs))])
    if args.num_iterations is not None:
        cmd.extend(["--num-iterations", str(int(args.num_iterations))])
    if args.save_components:
        cmd.append("--save-components")
    return cmd


def prepare_tasks(
    runs: Sequence[RunSpec],
    base_cfg: dict,
    measured_name: str,
    args: argparse.Namespace,
    iterative_script: Path,
    *,
    write_files: bool,
) -> tuple[list[PreparedTask], list[tuple[RunSpec, str]], int]:
    tasks: list[PreparedTask] = []
    failures: list[tuple[RunSpec, str]] = []
    skipped = 0

    for task_number, run in enumerate(runs, start=1):
        run_ok_marker = run.output_dir / "RUN_OK"
        if args.skip_completed and run_ok_marker.exists():
            skipped += 1
            print(f"[{task_number}/{len(runs)}] SKIP {run.output_dir} (RUN_OK exists)")
            continue

        try:
            validate_run_inputs(run, measured_name)

            cfg = copy.deepcopy(base_cfg)
            set_nested(cfg, ["project", "data_dir"], str(run.data_dir))
            set_nested(cfg, ["project", "output_dir"], str(run.output_dir))
            set_nested(
                cfg, ["osem", "gaussian_image_processor_fwhm"], list(run.gauss_fwhm)
            )
            set_nested(cfg, ["osem", "relative_difference_prior", "weight"], run.beta)
            set_nested(cfg, ["scatter_estimation", "additive_mode"], run.additive_mode)
            set_nested(
                cfg, ["scatter_estimation", "additive_alpha"], float(run.scatter_alpha)
            )
            set_nested(
                cfg, ["scatter_estimation", "residual_alpha"], float(run.residual_alpha)
            )
            if args.num_iterations is not None:
                set_nested(
                    cfg,
                    ["pipeline", "num_iterations"],
                    int(args.num_iterations),
                )

            cmd = build_task_command(run, args, iterative_script)
            task = PreparedTask(
                task_number=len(tasks) + 1,
                data_dir=run.data_dir,
                dataset_name=run.dataset_name,
                additive_mode=run.additive_mode,
                scatter_alpha=run.scatter_alpha,
                residual_alpha=run.residual_alpha,
                beta=run.beta,
                gauss_fwhm=list(run.gauss_fwhm),
                output_dir=run.output_dir,
                config_path=run.config_path,
                log_path=run.log_path,
                scatter_init_path=run.scatter_init_path,
                cmd=cmd,
            )

            scatter_alpha_str = (
                f"{run.scatter_alpha:g}"
                if mode_uses_scatter(run.additive_mode)
                else "na"
            )
            residual_alpha_str = (
                f"{run.residual_alpha:g}"
                if mode_uses_residual(run.additive_mode)
                else "na"
            )
            print(
                f"[{task_number}/{len(runs)}] READY dataset={run.dataset_name} "
                f"mode={run.additive_mode} scatter_alpha={scatter_alpha_str} "
                f"residual_alpha={residual_alpha_str} beta={run.beta:g} "
                f"gauss={run.gauss_fwhm}"
            )
            print(f"  output: {run.output_dir}")
            print(f"  cmd:    {shlex.join(cmd)}")

            if write_files:
                run.output_dir.mkdir(parents=True, exist_ok=True)
                with open(run.config_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)

            tasks.append(task)
        except Exception as exc:
            reason = str(exc)
            failures.append((run, reason))
            print(
                f"[{task_number}/{len(runs)}] FAIL_PRECHECK {run.output_dir} ({reason})"
            )

    return tasks, failures, skipped


def prepared_task_to_dict(task: PreparedTask) -> dict:
    return {
        "task_number": int(task.task_number),
        "data_dir": str(task.data_dir),
        "dataset_name": task.dataset_name,
        "additive_mode": task.additive_mode,
        "scatter_alpha": float(task.scatter_alpha),
        "residual_alpha": float(task.residual_alpha),
        "beta": float(task.beta),
        "gauss_fwhm": [float(x) for x in task.gauss_fwhm],
        "output_dir": str(task.output_dir),
        "config_path": str(task.config_path),
        "log_path": str(task.log_path),
        "scatter_init_path": str(task.scatter_init_path),
        "cmd": list(task.cmd),
    }


def prepared_task_from_dict(data: dict) -> PreparedTask:
    return PreparedTask(
        task_number=int(data["task_number"]),
        data_dir=Path(data["data_dir"]),
        dataset_name=str(data["dataset_name"]),
        additive_mode=str(data["additive_mode"]),
        scatter_alpha=float(data["scatter_alpha"]),
        residual_alpha=float(data["residual_alpha"]),
        beta=float(data["beta"]),
        gauss_fwhm=[float(x) for x in data["gauss_fwhm"]],
        output_dir=Path(data["output_dir"]),
        config_path=Path(data["config_path"]),
        log_path=Path(data["log_path"]),
        scatter_init_path=Path(data["scatter_init_path"]),
        cmd=[str(x) for x in data["cmd"]],
    )


def write_manifest(
    manifest_path: Path,
    tasks: Sequence[PreparedTask],
    env_prepends: dict[str, list[str]],
    mpi_env_notes: Sequence[str],
    args: argparse.Namespace,
    base_config: Path,
    iterative_script: Path,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_config": str(base_config),
        "iterative_script": str(iterative_script),
        "skip_completed": bool(args.skip_completed),
        "env_prepends": env_prepends,
        "mpi_env_notes": list(mpi_env_notes),
        "tasks": [prepared_task_to_dict(task) for task in tasks],
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(manifest_path: Path) -> dict:
    with open(manifest_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    payload["tasks"] = [prepared_task_from_dict(task) for task in payload["tasks"]]
    payload["env_prepends"] = {
        str(key): [str(value) for value in values]
        for key, values in payload.get("env_prepends", {}).items()
    }
    payload["mpi_env_notes"] = [str(note) for note in payload.get("mpi_env_notes", [])]
    payload["skip_completed"] = bool(payload.get("skip_completed", False))
    return payload


def execute_prepared_task(
    task: PreparedTask,
    *,
    env_prepends: dict[str, list[str]],
    mpi_env_notes: Sequence[str],
    skip_completed: bool,
) -> TaskResult:
    run_ok_marker = task.output_dir / "RUN_OK"
    fail_marker = task.output_dir / "RUN_FAILED"

    if skip_completed and run_ok_marker.exists():
        return TaskResult(task=task, status="skipped", message="RUN_OK exists")

    task.output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    apply_env_prepends(env, env_prepends)

    try:
        with open(task.log_path, "w", encoding="utf-8") as log_file:
            log_file.write(f"COMMAND: {shlex.join(task.cmd)}\n")
            if mpi_env_notes:
                log_file.write("\n")
                for note in mpi_env_notes:
                    log_file.write(f"ENV: {note}\n")
            log_file.write("\n")
            proc = subprocess.run(
                task.cmd,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
                env=env,
            )
    except Exception as exc:
        fail_text = (
            f"unexpected_error={type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"
        )
        fail_marker.write_text(fail_text, encoding="utf-8")
        return TaskResult(
            task=task,
            status="failed",
            message=f"unexpected_error={type(exc).__name__}: {exc}",
        )

    if proc.returncode == 0:
        run_ok_marker.write_text("ok\n", encoding="utf-8")
        if fail_marker.exists():
            fail_marker.unlink()
        return TaskResult(task=task, status="ok", message="exit_code=0")

    message = f"exit_code={proc.returncode}"
    fail_marker.write_text(message + "\n", encoding="utf-8")
    return TaskResult(task=task, status="failed", message=message)


def run_tasks_local(
    tasks: Sequence[PreparedTask],
    *,
    env_prepends: dict[str, list[str]],
    mpi_env_notes: Sequence[str],
    skip_completed: bool,
    max_local_jobs: int,
) -> list[TaskResult]:
    if int(max_local_jobs) <= 1:
        results: list[TaskResult] = []
        for idx, task in enumerate(tasks, start=1):
            print(
                f"[{idx}/{len(tasks)}] RUN task={task.task_number} "
                f"output={task.output_dir}"
            )
            result = execute_prepared_task(
                task,
                env_prepends=env_prepends,
                mpi_env_notes=mpi_env_notes,
                skip_completed=skip_completed,
            )
            print(f"  status: {result.status.upper()} ({result.message})")
            results.append(result)
        return results

    print(
        f"Running {len(tasks)} prepared tasks locally "
        f"with max_local_jobs={max_local_jobs}"
    )
    results: list[TaskResult] = []
    with ThreadPoolExecutor(max_workers=int(max_local_jobs)) as executor:
        futures = {
            executor.submit(
                execute_prepared_task,
                task,
                env_prepends=env_prepends,
                mpi_env_notes=mpi_env_notes,
                skip_completed=skip_completed,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            print(
                f"[{completed}/{len(tasks)}] TASK {result.task.task_number} "
                f"{result.status.upper()} ({result.message})"
            )
            results.append(result)

    results.sort(key=lambda item: item.task.task_number)
    return results


def default_job_script_path() -> Path:
    return (
        Path(__file__).resolve().parent / "cluster" / "iterative_penetrate_grid_job.sh"
    )


def staging_root_for(output_root: Path) -> Path:
    return output_root / "_grid"


def manifest_path_for(output_root: Path, job_name: str) -> Path:
    return staging_root_for(output_root) / f"{job_name}_manifest.json"


def cluster_logs_dir_for(output_root: Path, job_name: str) -> Path:
    return staging_root_for(output_root) / f"{job_name}_logs"


def resolve_cluster_parallel_env(
    args: argparse.Namespace,
) -> tuple[str | None, int | None]:
    pe_mode = str(args.cluster_parallel_env).lower()
    slots = args.cluster_slots

    if pe_mode == "none":
        return None, None

    if pe_mode == "mpi":
        resolved_slots = int(slots) if slots is not None else int(args.mpi_procs)
        if resolved_slots <= 1:
            raise ValueError("--cluster-parallel-env mpi requires slots > 1")
        if resolved_slots != int(args.mpi_procs):
            raise ValueError(
                "MPI cluster slots must match --mpi-procs so the scheduler allocation "
                "matches the task runtime."
            )
        return "mpi", resolved_slots

    if pe_mode == "smp":
        resolved_slots = int(slots) if slots is not None else 1
        if resolved_slots <= 1:
            raise ValueError("--cluster-parallel-env smp requires --cluster-slots > 1")
        return "smp", resolved_slots

    if pe_mode != "auto":
        raise ValueError(f"Unknown cluster parallel environment mode: {pe_mode}")

    if int(args.mpi_procs) > 1:
        if slots is not None and int(slots) != int(args.mpi_procs):
            raise ValueError(
                "When --mpi-procs > 1, --cluster-slots must match --mpi-procs "
                "or be omitted."
            )
        return "mpi", int(args.mpi_procs)

    if slots is not None and int(slots) > 1:
        return "smp", int(slots)

    return None, None


def expand_qsub_extra_args(values: Sequence[str] | None) -> list[str]:
    args_out: list[str] = []
    for value in values or []:
        args_out.extend(shlex.split(str(value)))
    return args_out


def build_cluster_qsub_command(
    args: argparse.Namespace,
    *,
    grid_script: Path,
    manifest_path: Path,
    logs_dir: Path,
    job_script: Path,
    task_count: int,
) -> list[str]:
    if task_count <= 0:
        raise ValueError("task_count must be > 0 for cluster submission")

    workdir = (
        args.cluster_workdir.expanduser().resolve()
        if args.cluster_workdir is not None
        else Path.cwd().resolve()
    )
    python_exec = str(args.cluster_python_executable or sys.executable)
    pe_name, pe_slots = resolve_cluster_parallel_env(args)

    env_exports = {
        "GRID_SCRIPT": str(grid_script),
        "TASK_MANIFEST": str(manifest_path),
        "PYTHON_EXECUTABLE": python_exec,
        "RUN_WORKDIR": str(workdir),
    }
    if args.cluster_env_setup_script is not None:
        env_exports["ENV_SETUP_SCRIPT"] = str(
            args.cluster_env_setup_script.expanduser().resolve()
        )

    def _escape_env_value(value: str) -> str:
        text = str(value)
        text = text.replace("\\", "\\\\")
        text = text.replace(",", "\\,")
        text = text.replace(" ", "\\ ")
        return text

    env_export_str = ",".join(
        f"{key}={_escape_env_value(value)}" for key, value in env_exports.items()
    )

    qsub_cmd = [
        "qsub",
        "-N",
        str(args.cluster_job_name),
        "-t",
        f"1-{task_count}",
        "-l",
        f"tmem={args.cluster_memory}",
        "-l",
        f"h_vmem={args.cluster_memory}",
        "-l",
        f"h_rt={args.cluster_runtime}",
        "-wd",
        str(workdir),
        "-o",
        str(logs_dir),
        "-e",
        str(logs_dir),
    ]

    if args.cluster_max_concurrent is not None:
        qsub_cmd.extend(["-tc", str(int(args.cluster_max_concurrent))])

    if args.cluster_queue:
        qsub_cmd.extend(["-q", str(args.cluster_queue)])

    if pe_name is not None and pe_slots is not None:
        qsub_cmd.extend(["-pe", pe_name, str(int(pe_slots)), "-R", "y"])

    qsub_cmd.extend(expand_qsub_extra_args(args.cluster_qsub_extra_arg))
    qsub_cmd.extend(["-v", env_export_str, str(job_script)])
    return qsub_cmd


def print_failure_summary(failures: Sequence[tuple[RunSpec, str]]) -> None:
    if not failures:
        return
    print("\nPrecheck failures:")
    for run, reason in failures:
        print(f"- {run.output_dir}: {reason}")


def print_result_summary(results: Sequence[TaskResult]) -> None:
    ok_count = sum(result.status == "ok" for result in results)
    skipped_count = sum(result.status == "skipped" for result in results)
    failed_count = sum(result.status == "failed" for result in results)
    print("\nExecution summary")
    print(f"- OK:       {ok_count}")
    print(f"- Skipped:  {skipped_count}")
    print(f"- Failed:   {failed_count}")
    if failed_count:
        print("\nFailed tasks:")
        for result in results:
            if result.status == "failed":
                print(f"- {result.task.output_dir}: {result.message}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run iterative_penetrate_scatter.py over a parameter grid."
    )
    parser.add_argument(
        "--execution",
        choices=["local", "cluster", "task"],
        default="local",
        help=(
            "Execution mode: local runs the sweep directly, cluster stages/submits an "
            "SGE array job, task executes a single staged manifest entry."
        ),
    )
    parser.add_argument(
        "--task-manifest",
        type=Path,
        default=None,
        help="Manifest written during staging. Required for --execution task.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="1-based manifest task index. Required for --execution task.",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=Path("project/config/local.yaml"),
        help="Base YAML config used as template for each run.",
    )
    parser.add_argument(
        "--iterative-script",
        type=Path,
        default=Path("scripts/iterative_penetrate_scatter.py"),
        help="Path to iterative penetrate scatter script.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/grid_additive_modes"),
        help="Root output directory for all grid runs.",
    )
    parser.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "Dataset SPECT directory. Repeat to add multiple datasets. "
            "If omitted, defaults are used."
        ),
    )
    parser.add_argument(
        "--initial-additive-name",
        type=str,
        default="scatter_dl.hs",
        help=(
            "Filename inside each dataset directory used as the initial additive "
            "estimate. Historical datasets often use scatter-like names even when "
            "the file contains the full additive term."
        ),
    )
    parser.add_argument(
        "--additive-mode",
        dest="additive_modes",
        action="append",
        default=None,
        choices=DEFAULT_ADDITIVE_MODES,
        help="Additive mode to include in the sweep. Repeatable.",
    )
    parser.add_argument(
        "--scatter-alpha",
        type=float,
        action="append",
        default=None,
        help=("Scatter damping alpha values to sweep (repeatable). Defaults to 0.2."),
    )
    parser.add_argument(
        "--residual-alpha",
        type=float,
        action="append",
        default=None,
        help=(
            "Residual damping alpha values to sweep (repeatable). "
            "Defaults to 0.2 and 1.0."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        action="append",
        default=None,
        help="RDP beta values to sweep (repeatable).",
    )
    parser.add_argument(
        "--gauss-fwhm",
        type=parse_gauss_fwhm,
        action="append",
        default=None,
        help=(
            "Gaussian image-processor FWHM triplet for the sweep, formatted as "
            "x,y,z. Repeat to add multiple options."
        ),
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=None,
        help="Optional override for pipeline.num_iterations.",
    )
    parser.add_argument(
        "--num-subsets",
        type=int,
        default=None,
        help="Optional CLI override passed to iterative script.",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Optional CLI override passed to iterative script.",
    )
    parser.add_argument(
        "--mpi-procs",
        type=int,
        default=1,
        help="MPI processes passed to iterative script.",
    )
    parser.add_argument(
        "--mpi-bin-dir",
        type=Path,
        default=None,
        help=(
            "Optional MPI bin directory to prepend to PATH "
            "(e.g. /opt/intel/oneapi/mpi/2021.17/bin)."
        ),
    )
    parser.add_argument(
        "--mpi-lib-dir",
        type=Path,
        default=None,
        help=(
            "Optional MPI lib directory to prepend to LD_LIBRARY_PATH "
            "(e.g. /opt/intel/oneapi/mpi/2021.17/lib)."
        ),
    )
    parser.add_argument(
        "--disable-auto-intel-mpi-env",
        action="store_true",
        help=(
            "Disable auto-detection of Intel MPI under /opt/intel/oneapi/mpi/2021.* "
            "when --mpi-procs > 1."
        ),
    )
    parser.add_argument(
        "--max-local-jobs",
        type=int,
        default=DEFAULT_LOCAL_WORKERS,
        help=(
            "Maximum number of local sweep tasks to run in parallel. Keep this at 1 "
            "unless you know the per-task SIMIND/SIRF load."
        ),
    )
    parser.add_argument(
        "--skip-completed",
        action="store_true",
        help="Skip runs with an existing RUN_OK marker.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned runs and commands without writing files or executing.",
    )
    parser.add_argument(
        "--save-components",
        action="store_true",
        help="Pass --save-components to iterative script.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit the prepared SGE array job when --execution cluster is used.",
    )
    parser.add_argument(
        "--cluster-job-name",
        type=str,
        default=DEFAULT_CLUSTER_JOB_NAME,
        help="SGE job name for cluster execution.",
    )
    parser.add_argument(
        "--cluster-runtime",
        type=str,
        default=DEFAULT_CLUSTER_RUNTIME,
        help="SGE wall-time request passed as -l h_rt=... for cluster execution.",
    )
    parser.add_argument(
        "--cluster-memory",
        type=str,
        default=DEFAULT_CLUSTER_MEMORY,
        help=(
            "SGE memory request passed as both -l tmem=... and -l h_vmem=... "
            "for cluster execution."
        ),
    )
    parser.add_argument(
        "--cluster-queue",
        type=str,
        default=None,
        help="Optional SGE queue passed as -q <queue>.",
    )
    parser.add_argument(
        "--cluster-max-concurrent",
        type=int,
        default=None,
        help="Optional SGE array concurrency limit passed as -tc <n>.",
    )
    parser.add_argument(
        "--cluster-parallel-env",
        choices=["auto", "mpi", "smp", "none"],
        default="auto",
        help=(
            "Cluster parallel environment selection. auto uses -pe mpi when "
            "--mpi-procs > 1, otherwise no parallel environment unless "
            "--cluster-slots > 1."
        ),
    )
    parser.add_argument(
        "--cluster-slots",
        type=int,
        default=None,
        help=(
            "Cluster slot count for the selected parallel environment. For MPI jobs "
            "this must match --mpi-procs."
        ),
    )
    parser.add_argument(
        "--cluster-env-setup-script",
        type=Path,
        default=None,
        help=(
            "Optional shell script sourced by the SGE job before running a task. "
            "Use this for SIMIND/MPI environment exports on the cluster."
        ),
    )
    parser.add_argument(
        "--cluster-python-executable",
        type=str,
        default=sys.executable,
        help="Python executable used inside the SGE job script.",
    )
    parser.add_argument(
        "--cluster-workdir",
        type=Path,
        default=None,
        help=(
            "Working directory passed to qsub via -wd. "
            "Defaults to the current directory."
        ),
    )
    parser.add_argument(
        "--cluster-job-script",
        type=Path,
        default=default_job_script_path(),
        help="Shell script executed by qsub for each array task.",
    )
    parser.add_argument(
        "--cluster-qsub-extra-arg",
        action="append",
        default=None,
        help=(
            "Additional qsub argument fragment. Repeatable; each fragment is "
            "shell-split."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.max_local_jobs) <= 0:
        raise ValueError("--max-local-jobs must be >= 1")
    if (
        args.cluster_max_concurrent is not None
        and int(args.cluster_max_concurrent) <= 0
    ):
        raise ValueError("--cluster-max-concurrent must be >= 1")
    if int(args.mpi_procs) <= 0:
        raise ValueError("--mpi-procs must be >= 1")

    if args.execution == "task":
        if args.task_manifest is None or args.task_index is None:
            raise ValueError(
                "--execution task requires both --task-manifest and --task-index"
            )
        manifest = load_manifest(args.task_manifest.expanduser().resolve())
        tasks = manifest["tasks"]
        if int(args.task_index) < 1 or int(args.task_index) > len(tasks):
            raise IndexError(
                f"task index {args.task_index} out of range 1..{len(tasks)}"
            )
        task = tasks[int(args.task_index) - 1]
        print(
            f"Running staged task {args.task_index}/{len(tasks)} "
            f"dataset={task.dataset_name} mode={task.additive_mode}"
        )
        print(f"Output: {task.output_dir}")
        print(f"Command: {shlex.join(task.cmd)}")
        result = execute_prepared_task(
            task,
            env_prepends=manifest.get("env_prepends", {}),
            mpi_env_notes=manifest.get("mpi_env_notes", []),
            skip_completed=manifest.get("skip_completed", False),
        )
        print(f"Status: {result.status.upper()} ({result.message})")
        if result.status == "failed":
            raise SystemExit(1)
        return

    base_config = args.base_config.expanduser().resolve()
    iterative_script = args.iterative_script.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not base_config.exists():
        raise FileNotFoundError(f"Base config not found: {base_config}")
    if not iterative_script.exists():
        raise FileNotFoundError(f"Iterative script not found: {iterative_script}")

    scatter_alphas = (
        [float(x) for x in args.scatter_alpha]
        if args.scatter_alpha
        else list(DEFAULT_SCATTER_ALPHAS)
    )
    for alpha in scatter_alphas:
        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(
                "--scatter-alpha values must be in (0, 1) to enforce "
                "damped scatter updates"
            )

    residual_alphas = (
        [float(x) for x in args.residual_alpha]
        if args.residual_alpha
        else list(DEFAULT_RESIDUAL_ALPHAS)
    )
    for alpha in residual_alphas:
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("--residual-alpha values must be in [0, 1]")

    additive_modes = (
        list(args.additive_modes)
        if args.additive_modes
        else list(DEFAULT_ADDITIVE_MODES)
    )
    betas = [float(x) for x in args.beta] if args.beta else list(DEFAULT_BETAS)
    gauss_options = (
        [[float(x) for x in values] for values in args.gauss_fwhm]
        if args.gauss_fwhm
        else [list(values) for values in DEFAULT_GAUSS_OPTIONS]
    )

    dataset_inputs = args.dataset if args.dataset else DEFAULT_DATASETS
    datasets = [Path(p).expanduser().resolve() for p in dataset_inputs]
    for data_dir in datasets:
        if not data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    with open(base_config, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    measured_name = str(
        base_cfg.get("data", {}).get("measured_data_filename", "peak.hs")
    )

    runs = build_runs(
        output_root=output_root,
        datasets=datasets,
        additive_modes=additive_modes,
        scatter_alphas=scatter_alphas,
        residual_alphas=residual_alphas,
        betas=betas,
        gauss_options=gauss_options,
        initial_additive_name=args.initial_additive_name,
    )

    print(f"Execution mode: {args.execution}")
    print(f"Planned candidate runs: {len(runs)}")
    print(f"Output root: {output_root}")
    print(f"Base config: {base_config}")

    env_prepends, mpi_env_notes = build_run_env_prepends(args)
    if mpi_env_notes:
        print("MPI environment injection:")
        for note in mpi_env_notes:
            print(f"- {note}")
    elif int(args.mpi_procs) > 1:
        print(
            "MPI environment injection: none (using current PATH/LD_LIBRARY_PATH as-is)"
        )

    tasks, failures, skipped = prepare_tasks(
        runs,
        base_cfg,
        measured_name,
        args,
        iterative_script,
        write_files=not args.dry_run,
    )

    print("\nPreparation summary")
    print(f"- Candidate runs:  {len(runs)}")
    print(f"- Ready tasks:     {len(tasks)}")
    print(f"- Skipped tasks:   {skipped}")
    print(f"- Precheck failed: {len(failures)}")
    print_failure_summary(failures)

    if not tasks:
        raise SystemExit(1 if failures else 0)

    manifest_path = manifest_path_for(output_root, args.cluster_job_name)
    if not args.dry_run:
        write_manifest(
            manifest_path,
            tasks,
            env_prepends,
            mpi_env_notes,
            args,
            base_config,
            iterative_script,
        )
        print(f"\nManifest: {manifest_path}")

    if args.dry_run:
        if args.execution == "cluster":
            job_script = args.cluster_job_script.expanduser().resolve()
            logs_dir = cluster_logs_dir_for(output_root, args.cluster_job_name)
            qsub_cmd = build_cluster_qsub_command(
                args,
                grid_script=Path(__file__).resolve(),
                manifest_path=manifest_path,
                logs_dir=logs_dir,
                job_script=job_script,
                task_count=len(tasks),
            )
            print("Cluster qsub command:")
            print(shlex.join(qsub_cmd))
        return

    if args.execution == "local":
        results = run_tasks_local(
            tasks,
            env_prepends=env_prepends,
            mpi_env_notes=mpi_env_notes,
            skip_completed=bool(args.skip_completed),
            max_local_jobs=int(args.max_local_jobs),
        )
        print_result_summary(results)
        if any(result.status == "failed" for result in results):
            raise SystemExit(1)
        return

    if args.execution != "cluster":
        raise ValueError(f"Unknown execution mode: {args.execution}")

    job_script = args.cluster_job_script.expanduser().resolve()
    if not job_script.exists():
        raise FileNotFoundError(f"Cluster job script not found: {job_script}")

    logs_dir = cluster_logs_dir_for(output_root, args.cluster_job_name)
    logs_dir.mkdir(parents=True, exist_ok=True)

    qsub_cmd = build_cluster_qsub_command(
        args,
        grid_script=Path(__file__).resolve(),
        manifest_path=manifest_path,
        logs_dir=logs_dir,
        job_script=job_script,
        task_count=len(tasks),
    )
    print(f"Cluster logs dir: {logs_dir}")
    print("Cluster qsub command:")
    print(shlex.join(qsub_cmd))

    if not args.submit:
        print(
            "Cluster sweep staged only. Re-run with --submit to submit "
            "the SGE array job."
        )
        return

    if shutil.which("qsub") is None:
        raise FileNotFoundError("qsub not found in PATH; cannot submit jobs.")

    print("Submitting SGE array job...")
    subprocess.run(qsub_cmd, check=True)
    print("SGE submission successful. Monitor with qstat -u $USER")


if __name__ == "__main__":
    main()
