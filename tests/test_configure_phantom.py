from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.configs import get
from simind_python_connector.connectors.python_connector import SimindPythonConnector
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    Box,
    CardiacDefect,
    CardiacSource,
    Ellipsoid,
    HorizontalCylinder,
    Insert,
    LibraryPhantom,
    MultipleInserts,
    PointSource,
    VerticalCylinder,
    VoxelPhantom,
)


pytestmark = pytest.mark.unit

_ANALYTIC_SWITCHES = {"BG", "HO", "CO", "FZ"} | {
    f"{letter}{n}"
    for letter, count in (("A", 3), ("L", 6), ("M", 4))
    for n in range(1, count + 1)
}


def _connector(tmp_path: Path) -> SimindPythonConnector:
    return SimindPythonConnector(
        config_source=get("Example.yaml"), output_dir=tmp_path, output_prefix="case01"
    )


def _values(connector, indices):
    return [connector.get_config().get_value(index) for index in indices]


def test_ellipsoid_in_a_horizontal_cylinder(tmp_path):
    connector = _connector(tmp_path)
    connector.configure_phantom(
        AnalyticPhantom(
            Ellipsoid((3.0, 2.0, 1.0)),
            HorizontalCylinder(10.0, (8.0, 6.0)),
            source_shift_cm=(2.0, 1.0, -1.0),
        )
    )
    config = connector.get_config()
    assert _values(connector, (15, 2, 3, 4)) == [1, 3.0, 2.0, 1.0]
    assert _values(connector, (16, 17, 18)) == [2.0, 1.0, -1.0]
    assert _values(connector, (14, 5, 6, 7)) == [4, 10.0, 8.0, 6.0]
    assert config.get_flag(11)
    assert (config.get_data_file(5), config.get_data_file(6)) == ("none", "none")
    assert connector._phantom_mode == "analytic"


@pytest.mark.parametrize(
    "source, code, dims",
    [
        (Box((1.0, 2.0, 3.0)), 2, [1.0, 2.0, 3.0]),
        (VerticalCylinder(4.0, (1.0, 2.0)), 3, [1.0, 2.0, 4.0]),
        (HorizontalCylinder(4.0, (1.0, 2.0)), 4, [4.0, 1.0, 2.0]),
        (PointSource(), 5, [0.0, 0.0, 0.0]),
        (CardiacSource((4.0, 3.0, 2.0)), 6, [4.0, 3.0, 2.0]),
    ],
)
def test_source_codes_and_dimensions(tmp_path, source, code, dims):
    connector = _connector(tmp_path)
    connector.configure_phantom(AnalyticPhantom(source, Box((20.0, 20.0, 20.0))))
    assert _values(connector, (15, 2, 3, 4)) == [code, *dims]


def test_no_attenuator_means_no_interactions_and_a_box_phantom(tmp_path):
    connector = _connector(tmp_path)
    source = np.ones((4, 4, 4), dtype=np.float32)
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)
    connector.configure_phantom(AnalyticPhantom(Ellipsoid((3.0, 2.0, 1.0))))
    assert not connector.get_config().get_flag(11)
    assert _values(connector, (14, 5, 6, 7)) == [2, 3.0, 2.0, 1.0]


@pytest.mark.parametrize(
    "mode, background, expected",
    [
        ("background", 0.5, {"BG": 0.5}),
        ("hot", None, {"HO": True}),
        ("cold", None, {"CO": True}),
    ],
)
def test_multiple_inserts_write_the_inp_file_and_switches(
    tmp_path, mode, background, expected
):
    connector = _connector(tmp_path)
    inserts = MultipleInserts(
        HorizontalCylinder(10.0, (10.0, 10.0)),
        (
            Insert((2.0, 2.0, 2.0), (3.0, 0.0, 0.0), 4.0),
            Insert((1.0, 1.0, 5.0), (-3.0, 1.5, 0.0), -0.5, shape="vertical_rod"),
        ),
        background=background,
        mode=mode,
    )
    connector.configure_phantom(
        AnalyticPhantom(inserts, HorizontalCylinder(10.0, (10.0, 10.0)))
    )

    assert (tmp_path / "case01.inp").read_text() == (
        "2,2,2,3,0,0,4,0\n1,1,5,-3,1.5,0,-0.5,4\n"
    )
    assert _values(connector, (15, 2, 3, 4)) == [7, 10.0, 10.0, 10.0]
    switches = connector.runtime_switches.switches
    assert {key: switches[key] for key in expected} == expected
    assert not ({"BG", "HO", "CO"} - set(expected)) & set(switches)


def test_cardiac_source_sets_the_myocardial_switches(tmp_path):
    connector = _connector(tmp_path)
    defect = CardiacDefect(location_deg=180.0, activity_ratio=0.2)
    connector.configure_phantom(
        AnalyticPhantom(
            CardiacSource((4.0, 4.0, 4.0), defect=defect), Box((20.0, 20.0, 20.0))
        )
    )
    switches = connector.runtime_switches.switches
    assert switches["A1"] == 122.0 and switches["M4"] == 6.1
    assert switches["L1"] == 180.0 and switches["L6"] == 0.2


def test_nema_library_phantom(tmp_path):
    connector = _connector(tmp_path)
    connector.configure_phantom(LibraryPhantom.NEMA_IQ)
    config = connector.get_config()
    assert _values(connector, (14, 15, 45, 31, 2, 5, 33, 34)) == [
        -5,
        -5,
        4,
        0.1,
        11.0,
        11.0,
        1,
        110,
    ]
    assert _values(connector, (78, 79, 81, 82)) == [364, 364, 364, 364]
    assert (config.get_data_file(5), config.get_data_file(6)) == ("nema", "nema")
    assert config.get_flag(11) and config.get_flag(15)
    assert connector.runtime_switches.switches["FZ"] == "phantom"
    assert connector._phantom_mode == "library"


def test_zubal_library_phantom_sets_only_codes_and_files(tmp_path):
    connector = _connector(tmp_path)
    before_45 = connector.get_config().get_value(45)
    connector.configure_phantom(LibraryPhantom.ZUBAL_TORSO)
    config = connector.get_config()
    assert _values(connector, (14, 15)) == [-2, -2]
    assert (config.get_data_file(5), config.get_data_file(6)) == (
        "vox_man1",
        "vox_man1",
    )
    assert config.get_value(45) == before_45
    assert "FZ" not in connector.runtime_switches.switches


def test_voxel_phantom_uses_its_density_and_known_activity(tmp_path):
    connector = _connector(tmp_path)
    connector.get_config().set_flag(11, False)
    activity = np.full((4, 4, 4), 0.25, dtype=np.float32)
    density = np.full((4, 4, 4), 1.5, dtype=np.float32)

    connector.configure_phantom(
        VoxelPhantom(activity, density, (4.0, 4.0, 4.0)), time_per_projection_s=20.0
    )

    written = np.fromfile(tmp_path / "case01_dns.dmi", dtype=np.uint16)
    assert np.all(written == 1500)
    assert connector.get_config().get_value(25) == pytest.approx(0.25 * 64 * 20.0)
    assert connector.get_config().get_flag(11)
    assert connector._phantom_mode == "voxel"


def test_switching_from_voxel_to_analytic_clears_voxel_settings(tmp_path):
    connector = _connector(tmp_path)
    source = np.ones((4, 4, 4), dtype=np.float32)
    connector.configure_voxel_phantom(source, np.zeros_like(source), (8.0, 4.0, 4.0))
    assert {"PX", "TH"} <= set(connector.runtime_switches.switches)

    connector.configure_phantom(AnalyticPhantom(PointSource(), Box((5.0, 5.0, 5.0))))

    assert not {"PX", "TH"} & set(connector.runtime_switches.switches)
    config = connector.get_config()
    assert (config.get_data_file(5), config.get_data_file(6)) == ("none", "none")


@pytest.mark.parametrize(
    "previous",
    [
        AnalyticPhantom(
            MultipleInserts(
                HorizontalCylinder(5.0, (5.0, 5.0)), (), background=1.0, mode="hot"
            ),
            Box((5.0, 5.0, 5.0)),
        ),
        AnalyticPhantom(CardiacSource((4.0, 4.0, 4.0)), Box((5.0, 5.0, 5.0))),
        LibraryPhantom.NEMA_IQ,
    ],
)
def test_switching_to_voxel_clears_analytic_and_library_switches(tmp_path, previous):
    connector = _connector(tmp_path)
    connector.configure_phantom(previous)
    source = np.ones((4, 4, 4), dtype=np.float32)
    connector.configure_voxel_phantom(source, np.zeros_like(source), 4.0)

    assert not _ANALYTIC_SWITCHES & set(connector.runtime_switches.switches)
    assert connector.get_config().get_value(15) == -1
    assert connector._phantom_mode == "voxel"


def test_unsupported_phantom_type(tmp_path):
    with pytest.raises(
        TypeError, match="AnalyticPhantom, LibraryPhantom or VoxelPhantom"
    ):
        _connector(tmp_path).configure_phantom(Ellipsoid((1.0, 1.0, 1.0)))


def test_run_removes_stale_aligned_density_files(tmp_path):
    connector = _connector(tmp_path)
    connector.configure_phantom(LibraryPhantom.NEMA_IQ)
    for name in ("case01.hct", "case01.ict"):
        (tmp_path / name).write_text("old")
    seen = {}

    def fake_run_simulation(
        output_prefix, orbit_file=None, runtime_switches=None, cwd=None
    ):
        seen["files"] = sorted(path.name for path in tmp_path.iterdir())

    connector.executor.run_simulation = fake_run_simulation  # type: ignore[assignment]
    connector._ensure_interfile_headers = lambda: []  # type: ignore[method-assign]
    connector._load_projection_outputs = lambda headers: {}  # type: ignore[method-assign]
    connector.run()

    assert "case01.hct" not in seen["files"] and "case01.ict" not in seen["files"]


@pytest.mark.parametrize(
    "phantom",
    [
        AnalyticPhantom(PointSource()),
        AnalyticPhantom(PointSource(), Box((5.0, 5.0, 5.0))),
        *list(LibraryPhantom),
        VoxelPhantom(
            np.ones((2, 2, 2), dtype=np.float32),
            np.ones((2, 2, 2), dtype=np.float32),
            (4.0, 4.0, 4.0),
        ),
    ],
)
def test_configure_phantom_enables_interfile_headers(tmp_path, phantom):
    connector = _connector(tmp_path)
    connector.get_config().set_flag(14, False)

    connector.configure_phantom(phantom)

    assert connector.get_config().get_flag(14)
