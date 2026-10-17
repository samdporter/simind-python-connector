"""Runtime switches CA, FW, OU, DP and X1-X6 against the SIMIND manuals.

v8 manual (simind_manual.pdf): CA p.27-28, FW p.29, list mode and OU p.29-30,
forced collimation and DP p.30, Xn p.39. v9 manual (Simind_v90-1.pdf): CA
p.27, FW p.28, list mode and OU p.29-30, forced collimation and DP p.30,
Xn p.40.
"""

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.core.config import RuntimeSwitches
from simind_python_connector.core.types import (
    DEFAULT_QUANTIZATION_SCALE,
    ScoringRoutine,
    SimulationError,
)
from tests import test_release_regressions_simind as regressions


pytestmark = [pytest.mark.validation, pytest.mark.requires_simind]

GRID = 32
VIEWS = 4
VOXEL_MM = 4.0
SEED = 7
WATER_SPHERE_RADIUS_MM = 60.0

V9_LINUX_DEFECT = "v9 Linux SCATTWIN engine defect (heap corruption)"
V9_LINUX_REPORT = (
    ".superpowers/sdd/2026-10-17-p3-feature-validation/v9-path-crash-report.md"
)
TWO_WINDOWS = "126.0,154.0,0\n92.0,125.0,0\n"


def _connector(
    directory: Path,
    prefix: str,
    routine: ScoringRoutine = ScoringRoutine.SCATTWIN,
    *,
    radius_mm: float = WATER_SPHERE_RADIUS_MM,
    source_voxels: tuple[tuple[int, int, int], ...] = ((GRID // 2,) * 3,),
    quantization_scale: float = DEFAULT_QUANTIZATION_SCALE,
    seed: int = SEED,
) -> SimindPythonConnector:
    """Point source in a density-1.0 sphere, Tc-99m, one 126-154 keV window."""
    connector = SimindPythonConnector(
        get("Example.yaml"), directory, prefix, quantization_scale=quantization_scale
    )
    connector.get_config().set_flag(11, True)
    connector.add_config_value(29, VIEWS)
    connector.add_config_value(53, 0)
    connector.add_config_value(28, VOXEL_MM / 10)
    connector.add_config_value(76, GRID)
    connector.add_config_value(77, GRID)
    source = np.zeros((GRID, GRID, GRID), np.float32)
    for voxel in source_voxels:
        source[voxel] = 1.0
    grid = np.indices((GRID, GRID, GRID)) - GRID // 2
    inside = (grid**2).sum(axis=0) <= (radius_mm / VOXEL_MM) ** 2
    connector.configure_voxel_phantom(
        source,
        inside.astype(np.float32),
        VOXEL_MM,
        scoring_routine=routine,
        mu_map_type="density",
    )
    connector.set_energy_windows([126], [154], [0])
    for switch, value in (("FI", "tc99m"), ("CC", "ma-lehr"), ("RR", seed)):
        connector.add_runtime_switch(switch, value)
    return connector


def _raw_components(directory: Path, prefix: str, suffix: str) -> set[str]:
    return {
        path.name[len(prefix) + 1 : path.name.rindex("_w")]
        for path in directory.glob(f"{prefix}_*_w*{suffix}")
    }


def _expected_raw_components(engine, ca: int) -> set[str]:
    if engine.version == 8:
        return [{"tot"}, {"air", "sca", "tot"}, {"air", "pri", "sca", "tot"}][ca]
    return [{"tot"}, {"air", "tot"}, {"ai2", "air", "pri", "sca", "tot"}][ca]


@pytest.fixture
def run_scattwin(engine, not_run):
    def _run(connector, case: str):
        try:
            return connector.run(timeout=600)
        except SimulationError:
            if engine.label != "v9-linux":
                raise
            not_run(case, V9_LINUX_DEFECT)

    return _run


@pytest.mark.macos_v9
@pytest.mark.parametrize("ca", [0, 1, 2])
def test_ca_raw_files(validate, run_scattwin, engine, tmp_path, ca):
    """Manual: CA selects the SCATTWIN files written (v8 p.27-28, v9 p.27).

    The expected sets come from observed runs: the v8 manual lists air, tot and
    pri for CA=2 but the engine also writes sca; the v9 CA=2 set is the one the
    native macOS build and a completing Linux build write.
    """
    case = f"ca_raw_files_{ca}"
    connector = _connector(tmp_path, "ca")
    connector.add_runtime_switch("CA", ca)

    outputs = run_scattwin(connector, case)
    derived = {
        key.rsplit("_w", 1)[0]
        for key, result in outputs.items()
        if "derived" in result.metadata
    }
    expected = _expected_raw_components(engine, ca)
    data = _raw_components(tmp_path, "ca", ".a00") - derived
    headers = _raw_components(tmp_path, "ca", ".h00")

    validate(
        case,
        float(data == expected and headers == expected),
        expected=1.0,
        abs_tol=0,
        note=(
            f"expected {sorted(expected)}; .a00 {sorted(data)}; .h00 {sorted(headers)}"
        ),
    )


@pytest.mark.macos_v9
def test_ca1_derived_primary(validate, not_run, run_scattwin, engine, tmp_path):
    if engine.version != 8:
        not_run("ca1_derived_primary", "v9 CA=1 writes no scatter image to derive from")
    connector = _connector(tmp_path, "pri")
    connector.add_runtime_switch("CA", 1)
    outputs = run_scattwin(connector, "ca1_derived_primary")
    if "pri_w1" not in outputs:
        not_run("ca1_derived_primary", "the connector does not derive pri_w1")

    validate(
        "ca1_derived_primary",
        float(
            np.max(
                np.abs(
                    outputs["pri_w1"].projection
                    - (outputs["tot_w1"].projection - outputs["sca_w1"].projection)
                )
            )
        ),
        expected=0,
        abs_tol=1e-6,
    )


def _run_with_window_file(run_scattwin, directory, case, name):
    (directory / f"{name}.win").write_text(TWO_WINDOWS)
    connector = _connector(directory, "fw")
    connector.add_runtime_switch("CA", 0)
    connector.add_runtime_switch("FW", name)
    return run_scattwin(connector, case)


@pytest.mark.macos_v9
def test_fw_two_windows(validate, run_scattwin, tmp_path):
    """Manual: FW names the window file, without extension (v8 p.29, v9 p.28)."""
    outputs = _run_with_window_file(run_scattwin, tmp_path, "fw_two_windows", "twowin")
    assert {"tot_w1", "tot_w2"} <= set(outputs)

    validate(
        "fw_two_windows",
        float(outputs["tot_w2"].projection.sum() / outputs["tot_w1"].projection.sum()),
        lower=0.05,
        upper=1.0,
        smoke=True,
    )


@pytest.mark.macos_v9
def test_fw_sixteen_character_name(validate, run_scattwin, tmp_path):
    name = "windowfile_sixtn"
    outputs = _run_with_window_file(
        run_scattwin, tmp_path, "fw_sixteen_character_name", name
    )

    validate(
        "fw_sixteen_character_name",
        float(sum(key.startswith("tot_w") for key in outputs)),
        expected=2,
        abs_tol=0,
    )


@pytest.mark.macos_v9
@pytest.mark.parametrize("switch", ["OU", "DP"])
def test_switch_of_unsupported_routine_is_refused(validate, switch):
    """OU (list-mode format, v8 p.29-30, v9 p.29-30) and DP (forced-collimation
    plane spacing, v8 p.30, v9 p.30) only act with the list-mode and
    forced-collimation scoring routines, which the connector does not run, so
    the switches are refused before SIMIND is started."""
    try:
        RuntimeSwitches().set_switch(switch, 1)
        refused = 0.0
    except ValueError:
        refused = 1.0

    validate(f"{switch.lower()}_refused", refused, expected=1.0, abs_tol=0)


def _penetrate_outputs(directory: Path, prefix: str, **kwargs) -> dict[str, np.ndarray]:
    switches = kwargs.pop("switches", {})
    connector = _connector(directory, prefix, ScoringRoutine.PENETRATE, **kwargs)
    for switch, value in switches.items():
        connector.add_runtime_switch(switch, value)
    return {
        key: result.projection for key, result in connector.run(timeout=600).items()
    }


def _max_difference(first: dict[str, np.ndarray], second: dict[str, np.ndarray]):
    assert first.keys() == second.keys()
    return float(max(np.max(np.abs(first[key] - second[key])) for key in first))


def test_x1_default_is_identity(validate, tmp_path):
    """Manual: X1 is the soft-tissue cross-section file, h2o by default."""
    default = _penetrate_outputs(tmp_path, "x1d")
    explicit = _penetrate_outputs(tmp_path, "x1e", switches={"X1": "h2o"})

    validate(
        "x1_default_is_identity",
        _max_difference(default, explicit),
        expected=0,
        abs_tol=0,
    )


def test_x1_copper_transmission(validate, tmp_path):
    """XCOM at 140 keV: Cu 0.251 and water 0.154 cm^2/g give 0.747 over 3 cm."""
    options = {"radius_mm": 30.0, "quantization_scale": 1.0}
    water = _penetrate_outputs(
        tmp_path, "x1w", switches={"X1": "h2o", "NN": 5}, **options
    )
    copper = _penetrate_outputs(
        tmp_path, "x1c", switches={"X1": "cu", "NN": 5}, **options
    )

    validate(
        "x1_copper_transmission",
        float(copper["geom_coll_primary"].sum() / water["geom_coll_primary"].sum()),
        lower=0.70,
        upper=0.80,
        note="histories raised 325x: the default leaves ~10% noise in the ratio",
    )


def _simind_ini_collimator_material() -> str | None:
    smc_dir = os.environ.get("SMC_DIR") or os.environ.get("SIMIND_SMC_DIR")
    ini = Path(smc_dir or "") / "simind.ini"
    if not ini.is_file():
        return None
    for line in ini.read_text().splitlines():
        value, _, description = line.partition("!")
        if "Default material for the collimator material" in description:
            return value.strip()
    return None


@pytest.mark.parametrize(
    "switch, material",
    [
        ("X2", "bone"),
        ("X3", "al"),
        ("X4", "nai"),
        ("X5", None),
        ("X6", "simind.ini"),
    ],
)
def test_default_cross_section_is_identity(
    validate, not_run, tmp_path, switch, material
):
    """Manual: the main menu names bone (X2), al (X3), nai (X4); X6 comes from
    simind.ini; no file is named for the backscatter material (X5)."""
    case = f"{switch.lower()}_default_is_identity"
    if material is None:
        not_run(case, "the manual names no default file for the backscatter material")
    if material == "simind.ini":
        material = _simind_ini_collimator_material()
        if material is None:
            not_run(case, "simind.ini is not readable through SMC_DIR")

    default = _penetrate_outputs(tmp_path, "xd")
    explicit = _penetrate_outputs(tmp_path, "xe", switches={switch: material})

    validate(case, _max_difference(default, explicit), expected=0, abs_tol=0)


@pytest.mark.macos_v9
def test_v9_linux_scattwin_defect(validate, not_run, engine):
    """Records the characterised v9 Linux SCATTWIN abort (heap corruption).

    The abort depends on the working-directory length and the source content;
    this is the crash report's case: 16^3 uniform body, three random voxels, a
    20-character working directory.
    """
    case = "v9_linux_scattwin_defect"
    if engine.label != "v9-linux":
        not_run(case, "the SCATTWIN heap corruption is specific to the v9 Linux build")
    directory = Path("/tmp/v9defect_ab12cd")
    directory.mkdir(exist_ok=True)
    try:
        connector = regressions._connector(
            directory, "defect", scoring_routine=ScoringRoutine.SCATTWIN
        )
        grid = regressions.GRID
        source = np.zeros((grid,) * 3, np.float32)
        source.flat[np.random.default_rng(3).choice(grid**3, 3, replace=False)] = 1
        connector.configure_voxel_phantom(
            source,
            np.full(source.shape, 0.15, np.float32),
            regressions.VOXEL_MM,
            scoring_routine=ScoringRoutine.SCATTWIN,
        )
        connector.add_runtime_switch("CA", 0)
        connector.set_activity(10, 1)
        try:
            connector.run(timeout=600)
            completed = 1.0
        except SimulationError:
            completed = 0.0
    finally:
        shutil.rmtree(directory)

    validate(
        case,
        completed,
        expected=1.0,
        abs_tol=0,
        note=f"working directory length {len(str(directory))}; see {V9_LINUX_REPORT}",
    )


@pytest.mark.macos_v9
def test_scattwin_default_ca(validate, tmp_path):
    """The manual defines no CA default for v8; v9 states CA:0 (p.27)."""
    connector = _connector(tmp_path, "dca")
    try:
        connector.run(timeout=600)
    except SimulationError as exc:
        validate(
            "scattwin_default_ca",
            0.0,
            lower=0,
            note=f"engine aborted without CA: {str(exc).splitlines()[0]}",
        )
        return

    components = sorted(_raw_components(tmp_path, "dca", ".a00"))
    validate(
        "scattwin_default_ca",
        1.0,
        lower=0,
        note=f"completed; components written: {components}",
    )
