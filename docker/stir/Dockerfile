ARG STIR_BASE_IMAGE=synerbi/sirf:latest
FROM ${STIR_BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SIRF_INSTALL_PATH=/opt/SIRF-SuperBuild/INSTALL
ENV SIMIND_ROOT=/workspace/simind
ENV SIMIND_BIN=${SIMIND_ROOT}/simind
ENV SIMIND_MPI_BIN=${SIMIND_ROOT}/simind_mpi
ENV SIMIND_SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIMIND_DATA_DIR=${SIMIND_ROOT}/smc_dir/
ENV SMC_DIR=${SIMIND_ROOT}/smc_dir/
ENV SIMIND_PYTHON_CONNECTOR_BLOCK_IMPORTS=sirf,pytomography
ENV PYTHONPATH=/workspace/docker/import_guard:${SIRF_INSTALL_PATH}/python:${PYTHONPATH}
ENV LD_LIBRARY_PATH=/opt/conda/lib:${SIRF_INSTALL_PATH}/lib:${SIRF_INSTALL_PATH}/lib64:${LD_LIBRARY_PATH}
ENV PATH=${SIMIND_ROOT}/bin:${SIMIND_ROOT}:${SIRF_INSTALL_PATH}/bin:${PATH}

# MPICH runtime and launcher for SIMIND's MPI binary; the compat script exposes
# the library names that binary was built against and selects MPICH's mpirun.
# See docker/python/install_mpich_compat.sh.
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends mpich \
    && rm -rf /var/lib/apt/lists/*
COPY docker/python/install_mpich_compat.sh /usr/local/bin/install_mpich_compat.sh
RUN chmod +x /usr/local/bin/install_mpich_compat.sh \
    && install_mpich_compat.sh
USER 1000

WORKDIR /workspace

COPY --chown=1000:100 . /workspace

RUN python -m pip install --upgrade pip \
    && pip install --constraint /workspace/docker/constraints.txt -e ".[dev,examples]" \
    && pip install "phantomgen @ git+https://github.com/varzakis/phantomgen@d9196118d5caef3f1bd2c27e5732bc03b48aed2a" \
    && python - <<'PY'
import stir  # noqa: F401
import pytest  # noqa: F401
import phantomgen  # noqa: F401
import pooch  # noqa: F401
import sphinx_gallery  # noqa: F401
from simind_python_connector.utils import _import_stir_extra

_import_stir_extra()
print("STIR Python modules available")
PY

CMD ["python", "examples/backends/plot_07a_stir_adaptor_osem.py"]
