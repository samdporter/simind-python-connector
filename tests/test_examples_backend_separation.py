"""Keep core examples backend-free and adaptor examples isolated."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gallery", ROOT / "docs" / "_docs.py")
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)
SCRIPTS = ROOT / "examples"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        "getting_started/plot_01_basic_simulation.py",
        "getting_started/plot_03_multi_window.py",
        "getting_started/plot_04_custom_config.py",
        "getting_started/plot_05_scattwin_vs_penetrate_comparison.py",
        "getting_started/plot_06_schneider_density_conversion.py",
        "phantoms/plot_09_phantom_library.py",
    ],
)
def test_core_examples_do_not_import_optional_backends(name: str):
    assert gallery.example_backend(SCRIPTS / name) == "python"


@pytest.mark.unit
@pytest.mark.parametrize(
    "name,backend",
    [
        ("backends/plot_07a_stir_adaptor_osem.py", "stir"),
        ("backends/plot_07b_sirf_adaptor_osem.py", "sirf"),
        ("backends/plot_07c_pytomography_adaptor_osem.py", "pytomography"),
    ],
)
def test_adaptor_examples_import_only_their_backend(name: str, backend: str):
    assert gallery.example_backend(SCRIPTS / name) == backend


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_path,backend",
    [
        ("mc_scatter/plot_11_mc_scatter_osem.py", "sirf"),
        ("mc_scatter/plot_12_scatter_normalisation.py", "sirf"),
        ("mc_scatter/plot_13_residual_correction.py", "sirf"),
        ("cil/plot_14_cil_scatter.py", "sirf"),
        ("cil/plot_15_cil_rdp_svrg.py", "sirf"),
        ("real_data/plot_17_y90_effective_energy.py", "sirf"),
    ],
)
def test_phantom_and_reconstruction_examples_import_only_their_backend(
    relative_path: str, backend: str
):
    assert gallery.example_backend(ROOT / "examples" / relative_path) == backend
