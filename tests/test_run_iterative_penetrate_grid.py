import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_iterative_penetrate_grid.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_iterative_penetrate_grid",
    SCRIPT_PATH,
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_args(**overrides):
    defaults = {
        "execution": "local",
        "skip_completed": False,
        "num_iterations": None,
        "num_subsets": None,
        "num_epochs": None,
        "mpi_procs": 1,
        "save_components": False,
        "cluster_parallel_env": "auto",
        "cluster_slots": None,
        "cluster_workdir": None,
        "cluster_python_executable": sys.executable,
        "cluster_env_setup_script": None,
        "cluster_job_name": "iter_pen_grid",
        "cluster_memory": "16G",
        "cluster_runtime": "48:00:00",
        "cluster_max_concurrent": None,
        "cluster_queue": None,
        "cluster_qsub_extra_arg": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.mark.unit
def test_build_runs_respects_mode_specific_sweeps(tmp_path):
    runs = MODULE.build_runs(
        output_root=tmp_path / "out",
        datasets=[tmp_path / "dataset"],
        additive_modes=["initial_scatter", "update_scatter_plus_residual"],
        scatter_alphas=[0.2, 0.4],
        residual_alphas=[0.2, 1.0],
        betas=[0.0, 0.1],
        gauss_options=[[6.9, 6.9, 6.9], [0.0, 0.0, 0.0]],
        initial_additive_name="scatter_dl.hs",
    )
    assert len(runs) == 20


@pytest.mark.unit
def test_prepare_tasks_writes_configs_and_skips_completed(tmp_path):
    data_dir = tmp_path / "dataset" / "SPECT"
    data_dir.mkdir(parents=True)
    (data_dir / "peak.hs").write_text("measured\n", encoding="utf-8")
    (data_dir / "initial_image.hv").write_text("initial\n", encoding="utf-8")
    (data_dir / "scatter_dl.hs").write_text("scatter\n", encoding="utf-8")

    runs = MODULE.build_runs(
        output_root=tmp_path / "out",
        datasets=[data_dir],
        additive_modes=["initial_scatter", "update_scatter"],
        scatter_alphas=[0.2],
        residual_alphas=[0.2],
        betas=[0.0],
        gauss_options=[[6.9, 6.9, 6.9]],
        initial_additive_name="scatter_dl.hs",
    )
    skipped_run = next(run for run in runs if run.additive_mode == "initial_scatter")
    kept_run = next(run for run in runs if run.additive_mode == "update_scatter")
    skipped_run.output_dir.mkdir(parents=True, exist_ok=True)
    (skipped_run.output_dir / "RUN_OK").write_text("ok\n", encoding="utf-8")

    base_cfg = {
        "project": {"data_dir": "", "output_dir": ""},
        "data": {"measured_data_filename": "peak.hs"},
        "osem": {"relative_difference_prior": {}},
        "scatter_estimation": {},
        "pipeline": {},
    }
    args = make_args(skip_completed=True)

    tasks, failures, skipped = MODULE.prepare_tasks(
        [skipped_run, kept_run],
        base_cfg,
        "peak.hs",
        args,
        SCRIPT_PATH,
        write_files=True,
    )

    assert skipped == 1
    assert len(failures) == 0
    assert len(tasks) == 1
    assert tasks[0].additive_mode == "update_scatter"
    assert tasks[0].config_path.exists()
    assert "--additive-mode" in tasks[0].cmd


@pytest.mark.unit
def test_manifest_roundtrip_preserves_tasks(tmp_path):
    task = MODULE.PreparedTask(
        task_number=1,
        data_dir=tmp_path / "data",
        dataset_name="dataset",
        additive_mode="update_scatter",
        scatter_alpha=0.2,
        residual_alpha=0.2,
        beta=0.1,
        gauss_fwhm=[6.9, 6.9, 6.9],
        output_dir=tmp_path / "out",
        config_path=tmp_path / "out" / "run_config.yaml",
        log_path=tmp_path / "out" / "run.log",
        scatter_init_path=tmp_path / "data" / "scatter_dl.hs",
        cmd=[sys.executable, "scripts/iterative_penetrate_scatter.py"],
    )
    manifest_path = tmp_path / "manifest.json"
    args = make_args(skip_completed=True)

    MODULE.write_manifest(
        manifest_path,
        [task],
        {"PATH": ["/opt/intel/bin"]},
        ["PATH+=/opt/intel/bin"],
        args,
        tmp_path / "base.yaml",
        SCRIPT_PATH,
    )
    manifest = MODULE.load_manifest(manifest_path)

    loaded_task = manifest["tasks"][0]
    assert loaded_task.output_dir == task.output_dir
    assert loaded_task.cmd == task.cmd
    assert manifest["env_prepends"]["PATH"] == ["/opt/intel/bin"]
    assert manifest["skip_completed"] is True


@pytest.mark.unit
def test_build_cluster_qsub_command_uses_parallel_env_and_tc(tmp_path):
    args = make_args(
        mpi_procs=8,
        cluster_workdir=tmp_path,
        cluster_queue="all.q",
        cluster_max_concurrent=5,
        cluster_qsub_extra_arg=["-P myproject"],
    )

    qsub_cmd = MODULE.build_cluster_qsub_command(
        args,
        grid_script=SCRIPT_PATH,
        manifest_path=tmp_path / "manifest.json",
        logs_dir=tmp_path / "logs",
        job_script=tmp_path / "job.sh",
        task_count=12,
    )

    assert qsub_cmd[:2] == ["qsub", "-N"]
    pe_index = qsub_cmd.index("-pe")
    assert ["-pe", "mpi", "8"] == qsub_cmd[pe_index : pe_index + 3]
    assert ["-R", "y"] == qsub_cmd[qsub_cmd.index("-R") : qsub_cmd.index("-R") + 2]
    assert ["-tc", "5"] == qsub_cmd[qsub_cmd.index("-tc") : qsub_cmd.index("-tc") + 2]
    assert ["-q", "all.q"] == qsub_cmd[qsub_cmd.index("-q") : qsub_cmd.index("-q") + 2]
    assert "-P" in qsub_cmd
    assert "myproject" in qsub_cmd
    assert "tmem=16G" in qsub_cmd
    assert "h_vmem=16G" in qsub_cmd
