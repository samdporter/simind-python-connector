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

## If MPI fails

Run serial temporarily:

```bash
python scripts/iterative_penetrate_scatter.py --config project/config/local.yaml --mpi-procs 1
```

## Make Permanent

Add the three `export` lines to `~/.bashrc` (or `~/.bash_profile`).
