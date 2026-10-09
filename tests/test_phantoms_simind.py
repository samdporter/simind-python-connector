import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get
from simind_python_connector.phantoms import (
    AnalyticPhantom,
    CardiacDefect,
    CardiacSource,
    Ellipsoid,
    HorizontalCylinder,
    Insert,
    LibraryPhantom,
    MultipleInserts,
    VoxelPhantom,
    voxelise,
)
from simind_python_connector.utils.interfile import (
    ProjectionGeometry,
    read_simind_density_image,
)


pytestmark = [pytest.mark.integration, pytest.mark.ci_skip, pytest.mark.requires_simind]

_N = 64
_VOXEL = 4.0
_WATER = HorizontalCylinder(12.0, (10.0, 10.0))


def _connector(tmp_path, prefix, num_projections=32):
    connector = SimindPythonConnector(get("Example.yaml"), tmp_path / prefix, prefix)
    connector.configure_acquisition(
        ProjectionGeometry(
            num_projections=num_projections,
            extent_deg=360.0,
            direction="CW",
            start_angle_deg=180.0,
            radius_mm=250.0,
            radii_mm=None,
            num_bins=_N,
            num_axial=_N,
            bin_size_mm=_VOXEL,
            axial_size_mm=_VOXEL,
            image_duration_s=None,
        )
    )
    connector.set_energy_windows([126], [154], [0])
    connector.add_runtime_switch("CC", "ma-lehr")
    connector.add_runtime_switch("RR", 7)
    connector.add_config_value(53, 0)  # analytical collimator
    connector.add_config_value(19, 2)
    return connector


def _centres(projection):
    """Centre of mass (row, column) of every view of a (view, row, column) array."""
    rows = np.arange(projection.shape[1])[:, None]
    cols = np.arange(projection.shape[2])[None, :]
    totals = projection.sum(axis=(1, 2))
    return np.stack(
        [
            (projection * rows).sum(axis=(1, 2)) / totals,
            (projection * cols).sum(axis=(1, 2)) / totals,
        ],
        axis=1,
    )


def _voxel_projection(tmp_path, prefix, phantom):
    activity, attenuator = voxelise(phantom, (_N, _N, _N), (_VOXEL,) * 3, supersample=2)
    connector = _connector(tmp_path, prefix)
    connector.get_config().set_flag(11, True)
    connector.configure_voxel_phantom(
        activity, attenuator, (_VOXEL,) * 3, mu_map_type="density"
    )
    return connector.run()["tot_w1"].projection


def _compare(label, simind, voxel):
    simind, voxel = simind / simind.sum(), voxel / voxel.sum()
    correlation = np.corrcoef(simind.ravel(), voxel.ravel())[0, 1]
    centre_gap = np.abs(_centres(simind) - _centres(voxel)).max()
    first_simind, first_voxel = _centres(simind)[0], _centres(voxel)[0]
    print(
        f"{label}: correlation={correlation:.3f} max centre gap={centre_gap:.2f} bins"
    )
    print(f"{label}: view-0 centres simind={first_simind} voxel={first_voxel}")
    return correlation, centre_gap


def test_1_analytic_and_voxel_paths_agree(tmp_path):
    phantom = AnalyticPhantom(
        Ellipsoid((4.0, 3.0, 2.0)), _WATER, source_shift_cm=(2.0, 1.0, -1.0)
    )
    connector = _connector(tmp_path, "analytic")
    connector.configure_phantom(phantom)
    analytic = connector.run()["tot_w1"].projection
    correlation, centre_gap = _compare(
        "analytic vs voxel", analytic, _voxel_projection(tmp_path, "voxel", phantom)
    )
    assert correlation > 0.95
    assert centre_gap <= 1.0


def test_2_insert_position_is_the_centre(tmp_path):
    inserts = MultipleInserts(
        _WATER, (Insert((1.5, 1.5, 1.5), (4.0, 3.0, 0.0), 1.0),), mode="hot"
    )
    phantom = AnalyticPhantom(inserts, _WATER)
    connector = _connector(tmp_path, "insert")
    connector.configure_phantom(phantom)
    analytic = connector.run()["tot_w1"].projection
    _, centre_gap = _compare(
        "insert", analytic, _voxel_projection(tmp_path, "insert_voxel", phantom)
    )
    assert centre_gap <= 1.0


def test_3_cardiac_with_and_without_a_defect(tmp_path):
    def projection(prefix, defect):
        connector = _connector(tmp_path, prefix)
        heart = CardiacSource((4.0, 4.0, 4.0), defect=defect)
        connector.configure_phantom(AnalyticPhantom(heart, _WATER))
        return connector.run()["tot_w1"].projection

    healthy = projection("heart", None)
    defect = projection(
        "defect",
        CardiacDefect(location_deg=0.0, angular_size_deg=45.0, activity_ratio=0.0),
    )
    healthy, defect = healthy / healthy.sum(), defect / defect.sum()
    difference = np.linalg.norm(defect - healthy) / np.linalg.norm(healthy)
    print(f"cardiac: relative difference with a defect {difference:.3f}")
    np.save(tmp_path / "heart.npy", healthy)
    np.save(tmp_path / "defect.npy", defect)
    assert healthy.sum() > 0 and defect.sum() > 0
    assert difference > 0.01


@pytest.mark.parametrize("library", list(LibraryPhantom))
def test_4_library_phantoms_run_and_write_their_density(tmp_path, library):
    prefix = library.name.lower()
    connector = _connector(tmp_path, prefix, num_projections=8)
    connector.configure_phantom(library)
    outputs = connector.run()
    projection = outputs["tot_w1"].projection
    density, voxel_sizes = read_simind_density_image(
        tmp_path / prefix / f"{prefix}.hct"
    )
    print(f"{library.name}: {projection.sum():.4g} counts")
    print(f"{library.name}: density {density.shape} at {voxel_sizes} mm")
    assert projection.sum() > 0
    assert density.max() > 0


@pytest.mark.requires_sirf
@pytest.mark.requires_phantomgen
def test_5_phantomgen_nema_by_route_1_matches_spectub(tmp_path):
    import sirf.STIR as sirf

    from simind_python_connector import SirfSimindAdaptor
    from simind_python_connector.builders import (
        STIRSPECTAcquisitionDataBuilder,
        STIRSPECTImageDataBuilder,
    )
    from simind_python_connector.phantoms import nema_iec_phantom

    n, voxel = 72, 4.42
    phantom = nema_iec_phantom((n, n, n), (voxel,) * 3, preset="pet")
    template = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(n),
            "!matrix size [2]": str(n),
            "!number of projections": "32",
            "scaling factor (mm/pixel) [1]": str(voxel),
            "scaling factor (mm/pixel) [2]": str(voxel),
            "Radius": "300",
        },
        backend="sirf",
    ).build(output_path=tmp_path / "template")
    image_builder = STIRSPECTImageDataBuilder(
        {
            **{f"!matrix size [{axis}]": str(n) for axis in (1, 2, 3)},
            **{f"scaling factor (mm/pixel) [{axis}]": str(voxel) for axis in (1, 2, 3)},
        },
        backend="sirf",
    )
    image_template = image_builder.build(output_path=tmp_path / "image")

    adaptor = SirfSimindAdaptor(get("Example.yaml"), str(tmp_path / "sim"), "nema")
    adaptor.set_template(template)
    adaptor.set_phantom(phantom, time_per_projection_s=10.0)
    adaptor.add_runtime_switch("CC", "ma-lehr")
    adaptor.add_runtime_switch("RR", 7)
    adaptor.add_config_value(53, 0)
    adaptor.add_config_value(19, 2)
    adaptor.set_energy_windows([126], [154], [0])
    simind = adaptor.run()["tot_w1"]
    activity, mu = adaptor.get_ground_truth(image_template)

    matrix = sirf.SPECTUBMatrix()
    matrix.set_attenuation_image(mu)
    matrix.set_keep_all_views_in_cache(True)
    matrix.set_resolution_model(0.0, 0.0, False)
    model = sirf.AcquisitionModelUsingMatrix(matrix)
    model.set_up(template, activity)
    reference = model.forward(activity)

    # SIRF arrays are (axial, view, bin); make them (view, axial, bin).
    sim = np.squeeze(simind.as_array()).transpose(1, 0, 2)
    ref = np.squeeze(reference.as_array()).transpose(1, 0, 2)
    for name, mask in sorted(phantom.masks.items()):
        print(f"{name}: {phantom.activity_mbq[mask > 0].sum():.3f} MBq")
    print(f"SIMIND counts {sim.sum():.4g}")
    _, centre_gap = _compare("NEMA vs SPECTUB", sim, ref)
    assert centre_gap <= 1.0


def test_shifted_analytic_to_voxel_matches_fresh_connector(tmp_path):
    analytic = AnalyticPhantom(
        Ellipsoid((1.0, 1.0, 1.0)), _WATER, source_shift_cm=(2.0, 1.0, -1.0)
    )
    activity, density = voxelise(analytic, (_N,) * 3, (_VOXEL,) * 3, supersample=2)
    voxel = VoxelPhantom(activity * 0.001, density, (_VOXEL,) * 3)

    reused = _connector(tmp_path, "reused", num_projections=4)
    reused.add_config_value(26, 0.01)
    reused.configure_phantom(analytic)
    assert reused.run()["tot_w1"].projection.sum() > 0

    reused.configure_phantom(voxel)
    actual = reused.run()["tot_w1"].projection

    fresh = _connector(tmp_path, "fresh", num_projections=4)
    fresh.add_config_value(26, 0.01)
    fresh.configure_phantom(voxel)
    expected = fresh.run()["tot_w1"].projection

    for projection in (actual, expected):
        assert np.isfinite(projection).all()
        assert np.all(projection.sum(axis=(1, 2)) > 0)

    correlation, centre_gap = _compare("reused vs fresh voxel", actual, expected)
    assert correlation > 0.95
    assert centre_gap <= 1.0
