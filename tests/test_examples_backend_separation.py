"""Keep core examples backend-free and adaptor examples isolated."""

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gallery", ROOT / "docs" / "_gallery.py")
gallery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gallery)


@pytest.mark.unit
@pytest.mark.parametrize(
    "name",
    [
        f"{number:02d}_{suffix}.py"
        for number, suffix in enumerate(
            [
                "basic_simulation",
                "runtime_switch_comparison",
                "multi_window",
                "custom_config",
                "scattwin_vs_penetrate_comparison",
                "schneider_density_conversion",
            ],
            start=1,
        )
    ],
)
def test_core_examples_do_not_import_optional_backends(name: str):
    assert (
        gallery.example_backend(ROOT / "examples" / "getting_started" / f"plot_{name}")
        == "python"
    )


@pytest.mark.parametrize(
    "name,backend",
    [
        ("plot_07a_stir_adaptor_osem.py", "stir"),
        ("plot_07b_sirf_adaptor_osem.py", "sirf"),
        ("plot_07c_pytomography_adaptor_osem.py", "pytomography"),
        ("plot_08a_stir_adaptor_from_dicom.py", "stir"),
        ("plot_08b_sirf_adaptor_from_dicom.py", "sirf"),
        ("plot_08c_pytomography_adaptor_from_dicom.py", "pytomography"),
    ],
)
@pytest.mark.unit
def test_adaptor_examples_import_only_their_backend(name: str, backend: str):
    assert gallery.example_backend(ROOT / "examples" / "backends" / name) == backend


@pytest.mark.unit
@pytest.mark.parametrize(
    "relative_path,backend",
    [
        ("phantoms/plot_09_phantom_library.py", "python"),
        ("phantoms/plot_10_nema_osem.py", "sirf"),
        ("mc_scatter/plot_11_mc_scatter_osem.py", "sirf"),
        ("mc_scatter/plot_12_scatter_normalisation.py", "sirf"),
        ("mc_scatter/plot_13_residual_correction.py", "sirf"),
        ("cil/plot_14_cil_scatter.py", "sirf"),
        ("cil/plot_15_cil_rdp_svrg.py", "sirf"),
    ],
)
def test_phantom_and_reconstruction_examples_import_only_their_backend(
    relative_path: str, backend: str
):
    assert gallery.example_backend(ROOT / "examples" / relative_path) == backend
