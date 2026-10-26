"""Tests for the Y-90 effective-energy tutorial."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.converters.attenuation import hu_to_attenuation


pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "examples" / "real_data" / "plot_17_y90_effective_energy.py"
    spec = importlib.util.spec_from_file_location("plot_17_y90_effective_energy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load()


def _write_records(path, rows):
    records = np.zeros(len(rows), dtype=script.LIST_MODE_DTYPE)
    for index, row in enumerate(rows):
        records[index] = row
    records.tofile(path)
    return path


def test_read_isotope_spectrum_keeps_headers_and_skips_comments(tmp_path):
    path = tmp_path / "source.isd"
    path.write_text(
        "# a comment\n"
        "CONTINUOUS=10.0\n"
        "BREMSSTRAHLUNG=y90.cpd\n"
        "\n"
        "3.0000E+01  7.1617E-04\n"
        "3.1000E+01  6.9503E-04\n"
    )

    spectrum = script.read_isotope_spectrum(path)

    assert spectrum.kind == "continuum"
    assert spectrum.continuous_scale == 10.0
    assert spectrum.bremsstrahlung == "y90.cpd"
    np.testing.assert_allclose(spectrum.energies_kev, [30.0, 31.0])
    np.testing.assert_allclose(spectrum.yields, [7.1617e-04, 6.9503e-04])


def test_read_isotope_spectrum_reads_discrete_lines(tmp_path):
    path = tmp_path / "nudat.isd"
    path.write_text("#Data from nudat3\nhalflife:64.05\n2186.2         1.4000E-08\n")

    spectrum = script.read_isotope_spectrum(path)

    assert spectrum.kind == "discrete"
    assert not spectrum.continuous
    assert spectrum.half_life == pytest.approx(64.05)
    np.testing.assert_allclose(spectrum.energies_kev, [2186.2])


def test_read_isotope_spectrum_accepts_unequal_energy_spacing(tmp_path):
    path = tmp_path / "uneven.isd"
    path.write_text("CONTINUOUS=1.0\n1.0 1.0\n2.0 2.0\n4.0 3.0\n")

    spectrum = script.read_isotope_spectrum(path)

    np.testing.assert_allclose(np.diff(spectrum.energies_kev), [1.0, 2.0])


@pytest.mark.parametrize(
    "text",
    [
        "# only comments\n",
        "CONTINUOUS=1.0\n30.0\n",
        "CONTINUOUS=1.0\n30.0 1.0 2.0\n",
        "CONTINUOUS=1.0\n30.0 not-a-number\n",
        "CONTINUOUS=1.0\n30.0 1.0\n20.0 1.0\n",
        "CONTINUOUS=1.0\n-30.0 1.0\n",
        "CONTINUOUS=not-a-number\n30.0 1.0\n",
        "CONTINUOUS=0.0\n30.0 1.0\n",
    ],
)
def test_read_isotope_spectrum_rejects_invalid_files(tmp_path, text):
    path = tmp_path / "invalid.isd"
    path.write_text(text)

    with pytest.raises(ValueError):
        script.read_isotope_spectrum(path)


def test_read_list_mode_decodes_layout_units_and_absolute_weights(tmp_path):
    path = _write_records(
        tmp_path / "events.lmf",
        [
            (261, 88, 3320, 1474, 0.80405435),
            (332, 708, 3294, 532, 2.60415695e-03),
            (-1, 2, 3, 75, -0.5),
        ],
    )

    events = script.read_list_mode(path)

    np.testing.assert_allclose(events.energies_kev, [147.4, 53.2, 7.5])
    np.testing.assert_allclose(events.weights, [0.80405435, 2.60415695e-03, 0.5])
    np.testing.assert_allclose(
        events.crystal_cm,
        [[2.61, 0.88, 33.20], [3.32, 7.08, 32.94], [-0.01, 0.02, 0.03]],
    )


def test_read_list_mode_marks_the_first_event_of_each_projection(tmp_path):
    path = _write_records(
        tmp_path / "events.lmf",
        [(0, 0, 0, 100, 1.0), (0, 0, 0, 100, -2.0), (0, 0, 0, 100, 3.0)],
    )

    events = script.read_list_mode(path)

    np.testing.assert_array_equal(events.projections, [0, 1, 1])
    np.testing.assert_allclose(events.weights, [1.0, 2.0, 3.0])


def test_read_list_mode_keeps_a_marker_on_the_first_record_in_projection_zero(
    tmp_path,
):
    path = _write_records(
        tmp_path / "events.lmf",
        [(0, 0, 0, 100, -1.0), (0, 0, 0, 100, 1.0)],
    )

    events = script.read_list_mode(path)

    np.testing.assert_array_equal(events.projections, [0, 0])


def test_read_list_mode_rejects_truncated_records(tmp_path):
    path = tmp_path / "truncated.lmf"
    _write_records(path, [(0, 0, 0, 100, 1.0)])
    with open(path, "ab") as stream:
        stream.write(b"\x00" * 8)

    with pytest.raises(ValueError, match="whole number"):
        script.read_list_mode(path)


def test_histogram_conserves_weight_and_applies_the_window():
    energies = np.array([10.0, 75.0, 100.0, 225.0, 400.0])
    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    counts, edges = script.histogram_spectrum(energies, weights)
    inside = script.window_weight(energies, weights)

    assert counts.sum() == pytest.approx(weights.sum())
    assert inside == pytest.approx(2.0 + 3.0 + 4.0)
    assert edges[0] == 0.0
    assert edges[-1] >= 2048.0


def test_window_channel_weights_split_channels_that_cross_the_window():
    weights = np.array([10.0, 20.0, 30.0])
    edges = np.array([70.0, 74.0, 78.0, 82.0])

    energies, clipped = script.window_channel_weights(
        weights, edges, lower=73.0, upper=79.0
    )

    np.testing.assert_allclose(energies, [73.5, 76.0, 78.5])
    np.testing.assert_allclose(clipped, [2.5, 20.0, 7.5])


def test_delta_weight_reproduces_its_monoenergetic_map():
    hu = np.array([-1000.0, -500.0, 0.0, 500.0, 1000.0])
    energies, weights = script.window_channel_weights(
        np.array([7.0]), np.array([139.0, 141.0])
    )

    weighted = script.weighted_attenuation(hu, energies, weights)

    np.testing.assert_allclose(weighted, hu_to_attenuation(hu, 140.0))


def test_constant_attenuation_is_independent_of_energy():
    hu = np.array([-1000.0, -200.0, 0.0, 400.0])
    energies = np.array([75.0, 120.0, 225.0])
    weights = np.array([1.0, 4.0, 2.0])

    weighted = script.weighted_attenuation(
        hu,
        energies,
        weights,
        converter=lambda values, energy: np.full_like(values, 0.15),
    )

    np.testing.assert_allclose(weighted, 0.15)


def test_weighted_attenuation_rejects_zero_weight():
    with pytest.raises(ValueError, match="zero total weight"):
        script.weighted_attenuation(np.zeros(4), np.array([100.0]), np.array([0.0]))


def test_weighted_mean_energy_uses_the_channel_weights():
    assert script.weighted_mean_energy(
        np.array([100.0, 200.0]), np.array([1.0, 3.0])
    ) == pytest.approx(175.0)


@pytest.mark.unit
def test_subiteration_count_matches_full_iterations():
    assert script.subiteration_count(12, 12) == 144
    assert script.subiteration_count(3, 4) == 12


def _y90_data():
    directory = os.environ.get("SIMIND_CONNECTOR_Y90_DATA")
    if not directory or not (Path(directory) / "peak.hs").is_file():
        pytest.skip("SIMIND_CONNECTOR_Y90_DATA is not set to the Y-90 phantom")
    return Path(directory)


@pytest.mark.integration
@pytest.mark.requires_sirf
def test_paired_reconstruction_shares_grid_and_initialisation(tmp_path):
    data_directory = _y90_data()
    ct_series = script.read_ct_series(data_directory / "ctac")
    affine = script.patient_to_spect_affine(
        data_directory / "peak.dcm", ct_series.table_height, ct_series.frame_uid
    )
    hu_grid, grid = script.resample_ct(
        ct_series, affine, data_directory / "umap_zoomed.hv", tmp_path
    )
    attenuation_150 = hu_to_attenuation(hu_grid, 150.0).astype(np.float32)
    attenuation_weighted = script.weighted_attenuation(
        hu_grid, np.array([100.0, 150.0, 200.0]), np.array([1.0, 1.0, 1.0])
    ).astype(np.float32)

    flipped = script.spectub_image(grid, attenuation_150)
    np.testing.assert_allclose(flipped.as_array(), attenuation_150[..., ::-1])

    import sirf.STIR as sirf

    sirf.AcquisitionData.set_storage_scheme("memory")
    acquisition = sirf.AcquisitionData(str(data_directory / "peak.hs"))
    images = script.paired_reconstruction(
        acquisition, grid, hu_grid, attenuation_150, attenuation_weighted
    )

    first, second = images.values()
    assert first.dimensions() == second.dimensions()
    assert first.voxel_sizes() == second.voxel_sizes()
    assert np.isfinite(first.as_array()).all()
    assert np.isfinite(second.as_array()).all()
    assert np.any(np.asarray(first.as_array()) != np.asarray(second.as_array()))


@pytest.mark.integration
@pytest.mark.requires_simind
def test_live_y90_source_and_list_mode_spectrum(tmp_path):
    try:
        isotope_file = script.find_isotope_file("y90.isd")
    except FileNotFoundError:
        pytest.skip("y90.isd is not installed with SIMIND")
    source_spectrum = script.read_isotope_spectrum(isotope_file)
    assert source_spectrum.kind == "continuum"
    assert source_spectrum.energies_kev.size > 100
    assert source_spectrum.energies_kev.max() - source_spectrum.energies_kev.min() > 500

    k, j, i = np.indices((32, 32, 32))
    sphere = ((i - 16) ** 2 + (j - 16) ** 2 + (k - 16) ** 2) <= 8**2
    source = sphere.astype(np.float32)
    density = sphere.astype(np.float32)
    prefix = script.acquire_detected_spectrum(
        source,
        density,
        tmp_path,
        views=1,
        quantization_scale=0.005,
        num_bins=32,
        num_axial=32,
        prefix="spectrum_smoke",
        timeout=900,
    )
    events = script.read_list_mode(prefix.with_suffix(".lmf"))
    check = script.check_window_counts(prefix, events)

    assert events.energies_kev.size > 0
    assert events.projections.max() == 0
    assert check["list_mode_window"] > 0
    assert check["ratio_to_projection"] == pytest.approx(1.0, abs=0.05)
    below = events.weights[events.energies_kev < script.WINDOW_LOW_KEV].sum()
    above = events.weights[events.energies_kev > script.WINDOW_HIGH_KEV].sum()
    assert below > 0
    assert above > 0
