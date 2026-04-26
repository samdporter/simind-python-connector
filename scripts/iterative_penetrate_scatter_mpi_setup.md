# MPI Setup Note for `iterative_penetrate_scatter.py`

Use this when running `scripts/iterative_penetrate_scatter.py` with `--mpi-procs > 1`.

## Environment Setup

```bash
export PATH=/opt/intel/oneapi/mpi/2021.17/bin:/home/sam/devel/simind_v8/simind:$PATH
export LD_LIBRARY_PATH=/opt/intel/oneapi/mpi/2021.17/lib:$LD_LIBRARY_PATH
export SMC_DIR=/home/sam/devel/simind_v8/simind/smc_dir/
```

Important: keep the trailing slash in `SMC_DIR`.

## Verify

```bash
which mpirun
which simind
which simind_mpi
ldd /home/sam/devel/simind_v8/simind/simind_mpi | grep -E "libmpi|libmpifort"
```

Expected:
- `mpirun` points to Intel MPI (`/opt/intel/oneapi/mpi/.../bin/mpirun`)
- `simind` and `simind_mpi` are found
- `libmpi.so.12` and `libmpifort.so.12` are resolved

## Run

```bash
python scripts/iterative_penetrate_scatter.py --config project/config/local.yaml --mpi-procs 8
```

## Sweep Runs

Local sweep, staged and executed directly:

```bash
python scripts/run_iterative_penetrate_grid.py \
  --execution local \
  --base-config project/config/local.yaml \
  --mpi-procs 8 \
  --skip-completed
```

Local sweep with limited task-level parallelism:

```bash
python scripts/run_iterative_penetrate_grid.py \
  --execution local \
  --base-config project/config/local.yaml \
  --mpi-procs 1 \
  --max-local-jobs 2 \
  --skip-completed
```

Cluster sweep, stage only:

```bash
python scripts/run_iterative_penetrate_grid.py \
  --execution cluster \
  --base-config project/config/local.yaml \
  --mpi-procs 1 \
  --cluster-env-setup-script /path/to/simind_env.sh \
  --cluster-runtime 48:00:00 \
  --cluster-memory 48G
```

Cluster sweep, submit immediately:

```bash
python scripts/run_iterative_penetrate_grid.py \
  --execution cluster \
  --base-config project/config/local.yaml \
  --mpi-procs 1 \
  --cluster-env-setup-script /path/to/simind_env.sh \
  --cluster-runtime 48:00:00 \
  --cluster-memory 48G \
  --submit
```

Notes:
- The cluster mode writes a manifest under `output/.../_grid/` and submits an SGE array job using `SGE_TASK_ID`.
- When `--mpi-procs > 1`, the runner requests `-pe mpi <mpi-procs>` and `-R y` to match the UCL SGE guidance for parallel jobs.
- `--cluster-max-concurrent` is optional. If set, it passes `qsub -tc <n>` to cap how many array tasks run at once; omit it to let SGE schedule as many tasks as policy/resources allow.
- Use `--dry-run` to inspect the prepared commands and the final `qsub` command without writing configs or submitting.

## If MPI fails

Run serial temporarily:

```bash
python scripts/iterative_penetrate_scatter.py --config project/config/local.yaml --mpi-procs 1
```

## Make Permanent

Add the three `export` lines to `~/.bashrc` (or `~/.bash_profile`).
