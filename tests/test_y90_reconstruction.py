"""Tests for the real Y-90 Monte Carlo reconstruction tutorial."""

import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.utils.interfile import (
    read_header,
    read_projection_geometry,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "examples" / "real_data"
VOXEL_ML = 4.2578**3 / 1000.0


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


script = _load("plot_18_y90_mc_scatter")


def _mask_with_volume(shape, volume_ml, start=0):
    """Unit-weight mask whose represented volume is volume_ml."""
    weights = np.zeros(shape, dtype=float)
    count = int(round(volume_ml / VOXEL_ML))
    flat = weights.ravel()
    if start + count > flat.size:
        raise ValueError("mask is too small for the requested volume")
    flat[start : start + count] = 1.0
    return weights.reshape(shape)


SHAPE = (16, 16, 65)


def _spect_affine():
    v = script.VOXEL_MM
    centre = (script.MATRIX_SIZE - 1) / 2.0
    affine = np.eye(4)
    affine[0, 0] = v
    affine[1, 1] = v
    affine[2, 2] = -v
    affine[0, 3] = -1.999236 - centre * v
    affine[1, 3] = -140.0 - centre * v
    affine[2, 3] = -571.874206 + centre * v
    return affine


def _raw(shape=(1, 4, 6, 4), total=100.0, primary=60.0):
    return {
        "all_interactions": np.full(shape, total, dtype=np.float32),
        "geom_coll_primary": np.full(shape, primary, dtype=np.float32),
    }


# %%
# Metrics


@pytest.mark.unit
def test_mass_based_ratio_is_scale_invariant():
    shape = SHAPE
    sphere = _mask_with_volume(shape, 8.082)
    liver = _mask_with_volume(shape, 978.17, start=200)
    liver = script.liver_excluding_sphere(liver, sphere)
    image = np.ones(shape)
    image[sphere > 0] = 4.0

    reference = script.voi_ratio(image, sphere, liver)
    scaled = script.voi_ratio(image * 37.0, sphere, liver)
    per_ml = script.voi_ratio(image * script.VOXEL_MM**3, sphere, liver)

    assert reference == pytest.approx(4.0)
    assert scaled == pytest.approx(reference)
    assert per_ml == pytest.approx(reference)


@pytest.mark.unit
def test_total_calibration_sets_the_support_total():
    support = np.ones((4, 4, 4), dtype=bool)
    support[0, 0, 0] = False
    image = np.ones((4, 4, 4))

    calibrated, scale = script.calibrate_to_total(image, support, 160.8)

    total = calibrated[support].sum() * VOXEL_ML
    assert total == pytest.approx(160.8)
    assert scale == pytest.approx(160.8 / (support.sum() * VOXEL_ML))


@pytest.mark.unit
def test_fractional_masks_are_used_as_weights():
    weights = np.array([0.25, 0.75, 0.0])
    values = np.array([4.0, 0.0, 100.0])

    assert script.weighted_mean(values, weights) == pytest.approx(1.0)


@pytest.mark.unit
def test_sphere_is_excluded_from_the_liver_weights():
    shape = (4, 4, 4)
    sphere = np.zeros(shape)
    sphere[1:3, 1:3, 1:3] = 1.0
    liver = np.ones(shape)

    excluded = script.liver_excluding_sphere(liver, sphere)

    assert np.all(excluded[sphere >= 1.0] == 0.0)
    assert np.all((excluded * sphere) == 0.0)
    assert np.all(excluded <= liver)


@pytest.mark.unit
def test_fixed_sphere_reproduces_the_8_08_ml_volume():
    sphere = script.sphere_fraction(
        script.SPHERE_CENTRE_MM,
        script.SPHERE_RADIUS_MM,
        _spect_affine(),
    )

    assert script.mask_volume_ml(sphere) == pytest.approx(8.08, abs=0.02)
    assert script.mask_centroid_mm(sphere, _spect_affine()) == pytest.approx(
        script.SPHERE_CENTRE_MM, abs=1.0
    )


@pytest.mark.unit
def test_nonpositive_denominators_are_rejected():
    shape = (3, 3, 3)
    sphere = np.zeros(shape)
    sphere[1, 1, 1] = 1.0
    liver = np.ones(shape)
    liver[1, 1, 1] = 0.0
    image = np.zeros(shape)
    image[1, 1, 1] = 4.0

    with pytest.raises(ValueError, match="nonpositive"):
        script.voi_ratio(image, sphere, liver)
    with pytest.raises(ValueError, match="nonpositive"):
        script.calibrate_to_total(np.zeros(shape), np.ones(shape, dtype=bool))
    with pytest.raises(ValueError, match="weights sum to zero"):
        script.weighted_mean(np.ones(shape), np.zeros(shape))


@pytest.mark.unit
def test_image_units_cancel_in_the_ratio_and_are_per_voxel_in_calibration():
    shape = SHAPE
    sphere = _mask_with_volume(shape, 8.082)
    liver = _mask_with_volume(shape, 978.17, start=200)
    liver = script.liver_excluding_sphere(liver, sphere)
    image = np.ones(shape)
    image[sphere > 0] = 4.0

    assert script.voi_ratio(image * 4.2578**3, sphere, liver) == pytest.approx(
        script.voi_ratio(image, sphere, liver)
    )

    support = np.ones(shape, dtype=bool)
    metrics = script.compartment_metrics(image, sphere, liver, support)
    expected = 160.8 / script.mask_volume_ml(support) * 1000.0
    # The liver is not uniform here, so only the total is a closed-form check.
    assert metrics["body_total_mbq"] == pytest.approx(160.8)
    assert expected > 0.0


@pytest.mark.unit
def test_old_truth_mass_denominator_reproduces_the_4_70_bias():
    shape = SHAPE
    sphere = _mask_with_volume(shape, 8.082)
    liver = _mask_with_volume(shape, 978.17, start=200)
    liver = script.liver_excluding_sphere(liver, sphere)
    image = np.ones(shape)
    image[sphere > 0] = 4.0

    new_ratio = script.voi_ratio(image, sphere, liver)
    old_ratio = script.ratio_with_truth_mass(image, sphere, liver)

    assert new_ratio == pytest.approx(4.0)
    assert old_ratio == pytest.approx(4.70, abs=0.02)
    assert old_ratio / new_ratio == pytest.approx(
        script.LIVER_MASS_G / script.mask_volume_ml(liver), rel=0.01
    )


# %%
# Schedule and sharing


def _schedule_fakes():
    calls = []
    images = []
    raw = _raw()

    def simulate(image, label):
        calls.append(("simulate", label, float(np.asarray(image).sum())))
        return {key: value.copy() for key, value in raw.items()}

    def simulate_many(jobs):
        recorded = []
        for name, image in jobs:
            recorded.append((name, np.asarray(image).copy()))
            calls.append(("simulate_many", name, float(np.asarray(image).sum())))
        return {
            name: {key: value.copy() for key, value in raw.items()} for name, _ in jobs
        }

    def reconstruct(image, additive, subiterations):
        calls.append(
            ("reconstruct", int(subiterations), float(np.asarray(additive).sum()))
        )
        images.append(np.asarray(image).copy())
        return np.asarray(image, float) + 0.001 * np.asarray(additive).sum()

    def estimate(name, raw_components, image, index):
        values = (
            np.asarray(raw_components["all_interactions"], float)
            - np.asarray(raw_components["geom_coll_primary"], float)
        ) * (index + 1)
        return script.Additive(
            values=values,
            scale=1.0,
            zero_denominator_fraction=0.0,
        )

    def ratio(image):
        return float(np.asarray(image).sum())

    return simulate, simulate_many, reconstruct, estimate, ratio, calls, images


@pytest.mark.unit
def test_damping_halves_the_first_estimate_and_averages_the_second():
    estimate_one = np.full((1, 2, 3, 2), 10.0)
    estimate_two = np.full((1, 2, 3, 2), 30.0)

    first = script.damped_additive(np.zeros_like(estimate_one), estimate_one)
    second = script.damped_additive(first, estimate_two)

    assert np.allclose(first, 5.0)
    assert np.allclose(second, 0.5 * first + 0.5 * estimate_two)
    assert np.all(second >= 1e-5)


@pytest.mark.unit
def test_damping_applies_the_floor():
    estimate = np.full((1, 2, 3, 2), 1e-7)

    damped = script.damped_additive(np.zeros_like(estimate), estimate)

    assert np.all(damped == 1e-5)


@pytest.mark.unit
def test_schedule_has_six_runs_two_updates_and_equal_budgets():
    simulate, simulate_many, reconstruct, estimate, ratio, calls, _ = _schedule_fakes()

    result = script.run_schedule(
        np.ones((2, 2, 2)),
        np.zeros((1, 4, 6, 4)),
        simulate,
        simulate_many,
        reconstruct,
        estimate,
        ratio,
    )

    assert result.simulate_calls == ["shared"] + [name for name, _ in script.STRATEGIES]
    assert len(result.simulate_calls) == 6
    assert all(result.update_counts[name] == 2 for name, _ in script.STRATEGIES)
    assert result.update_counts[script.BASELINE] == 0
    assert all(value == 144 for value in result.budgets.values())
    assert result.settings["num_subsets"] == 12
    assert result.settings["damping"] == 0.5


@pytest.mark.requires_sirf
def test_sirf_schedule_refresh_count_and_identical_settings(y90_data):
    sirf = pytest.importorskip("sirf.STIR")
    measured = sirf.AcquisitionData(str(y90_data / "peak.hs"))
    simulate, simulate_many, reconstruct, estimate, ratio, _, _ = _schedule_fakes()

    result = script.run_schedule(
        np.ones((2, 2, 2)),
        np.zeros((1, 4, 6, 4)),
        simulate,
        simulate_many,
        reconstruct,
        estimate,
        ratio,
    )

    assert len(result.simulate_calls) == 6
    assert all(result.update_counts[name] == 2 for name, _ in script.STRATEGIES)
    assert result.update_counts[script.BASELINE] == 0
    assert all(result.budgets[name] == 144 for name, _ in script.STRATEGIES)
    assert all(result.budgets[name] == 144 for name in [script.BASELINE])
    attenuation = script._image_from_array(np.zeros((128, 128, 128), np.float32))
    model = script._make_model(measured, attenuation)
    assert type(model).__name__ == "AcquisitionModelUsingMatrix"


@pytest.mark.unit
def test_divergent_second_refresh_uses_each_strategys_own_first_image():
    simulate, simulate_many, reconstruct, estimate, ratio, _, _ = _schedule_fakes()
    recorded = []

    def recording_many(jobs):
        for name, image in jobs:
            recorded.append((name, np.asarray(image).copy()))
        return simulate_many(jobs)

    result = script.run_schedule(
        np.ones((2, 2, 2)),
        np.zeros((1, 4, 6, 4)),
        simulate,
        recording_many,
        reconstruct,
        estimate,
        ratio,
    )

    second_sources = dict(recorded)
    for name, _ in script.STRATEGIES:
        assert np.allclose(second_sources[name], result.first_images[name])
        assert not np.allclose(second_sources[name], result.common_image)


@pytest.mark.unit
def test_shared_raw_components_are_not_mutated_by_normalisation():
    raw = _raw()
    before = {key: value.copy() for key, value in raw.items()}
    reference = np.full((1, 4, 6, 4), 90.0)
    measured = np.full((1, 4, 6, 4), 100.0)

    known = script.build_additive(raw, measured, "known")
    reference_sum = script.build_additive(
        raw, measured, "reference_sum", reference_projection=reference
    )
    pixelwise = script.build_additive(raw, measured, "pixelwise")

    for key, value in raw.items():
        assert np.array_equal(value, before[key])
    known.values[0, 0, 0, 0] += 1.0
    assert pixelwise.values[0, 0, 0, 0] == pytest.approx(40.0)
    assert reference_sum.values[0, 0, 0, 0] != known.values[0, 0, 0, 0]


@pytest.mark.unit
def test_shared_output_normalisation_equals_separate_normalisation():
    raw = _raw()
    measured = np.full((1, 4, 6, 4), 100.0)
    reference = np.full((1, 4, 6, 4), 90.0)

    shared = script.build_additive(
        raw, measured, "reference_sum", reference_projection=reference
    )
    separate = script.build_additive(
        {key: value.copy() for key, value in raw.items()},
        measured,
        "reference_sum",
        reference_projection=reference,
    )

    assert np.allclose(shared.values, separate.values)
    assert shared.scale == pytest.approx(separate.scale)
    assert shared.scale == pytest.approx(90.0 / 60.0)


@pytest.mark.unit
def test_pixelwise_zero_denominator_coverage_is_reported():
    raw = _raw(shape=(1, 2, 3, 2))
    raw["all_interactions"][0, 0, 0, 0] = 0.0
    raw["geom_coll_primary"][0, 0, 0, 0] = 0.0
    measured = np.full((1, 2, 3, 2), 10.0)

    additive = script.build_additive(raw, measured, "pixelwise")

    assert additive.zero_denominator_fraction == pytest.approx(1.0 / 12.0)
    assert additive.values.shape == raw["all_interactions"].shape


# %%
# Presentation and completion


def _failed_summary():
    return {
        "status": "quality_not_established",
        "gates": {
            "registration": True,
            "seed_passed": False,
            "seed_scatter_l1": 0.0624,
            "stability_passed": False,
            "stability_max_ratio_change": 0.0622,
            "nn_passed": False,
            "nn_max_ratio_change": 0.10,
        },
    }


@pytest.mark.unit
def test_quality_failure_summary_preserves_false_gates(tmp_path):
    script.complete_run(_failed_summary(), tmp_path, strict=False)

    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["status"] == "quality_not_established"
    assert written["gates"]["seed_passed"] is False
    assert written["gates"]["stability_max_ratio_change"] == pytest.approx(0.0622)
    marker = (tmp_path / "UNACCEPTED.txt").read_text()
    assert "seed_passed" in marker
    assert "stability_passed" in marker


@pytest.mark.unit
def test_diagnostic_completion_prints_warning_without_raising(tmp_path, capsys):
    script.complete_run(_failed_summary(), tmp_path, strict=False)

    output = capsys.readouterr().out
    assert "WARNING: Statistical adequacy was not established" in output
    assert "UNACCEPTED.txt" in output


@pytest.mark.unit
def test_strict_completion_raises(tmp_path):
    with pytest.raises(RuntimeError, match="Statistical adequacy"):
        script.complete_run(_failed_summary(), tmp_path, strict=True)

    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "UNACCEPTED.txt").exists()


@pytest.mark.unit
def test_precondition_and_simulation_errors_still_propagate(tmp_path):
    with pytest.raises(RuntimeError, match="registration failed"):
        script.accept_registration(
            {"passed": False, "reason": "bad offset", "status": "fail"}
        )

    source = np.full((32, 32, 32), 1e-6, dtype=np.float32)
    source[0, 0, 0] = 1.0
    with pytest.raises(RuntimeError, match="lost to quantization"):
        script.simulate_single(
            "bad",
            source,
            tmp_path / "template.hs",
            tmp_path / "density.npy",
            tmp_path,
        )


@pytest.mark.unit
def test_global_arm_is_diagnostic_only_in_the_tables(capsys):
    valid = [
        "Known activity",
        "Reference, sum",
        "Reference, trimmed",
        "Dewaraja, pixel-wise",
    ]
    metrics = {
        name: {
            "sphere_mbq": 0.4,
            "sphere_kbq_g": 50.0,
            "liver_mbq": 25.0,
            "liver_kbq_g": 25.0,
            "body_total_mbq": 160.8,
        }
        for name in valid + [script.GLOBAL_STRATEGY, script.BASELINE]
    }
    ratios_first = {name: 1.5 for name in valid}
    ratios_first[script.GLOBAL_STRATEGY] = 0.00174987
    ratios_final = {name: 1.7 for name in valid}
    ratios_final[script.GLOBAL_STRATEGY] = 0.0713
    ratios_final[script.BASELINE] = 1.579

    script.print_metric_tables(metrics, ratios_first, ratios_final, valid)
    output = capsys.readouterr().out

    main, diagnostic = script.main_and_diagnostic_names(valid)
    assert diagnostic == [script.GLOBAL_STRATEGY]
    assert script.GLOBAL_STRATEGY not in main

    primary, rest = output.split("Diagnostic arm", maxsplit=1)
    assert "Global scaling" not in primary
    assert "Global scaling" in rest
    assert "0.00175" in rest
    assert "0.0713" in rest
    calibration = output.split("Relative calibration")[1]
    assert "Global scaling" not in calibration

    positions = [primary.index(name) for name in main + [script.BASELINE]]
    assert positions == sorted(positions)


# %%
# Centre sensitivity


def _centred_affine(shape=(16, 16, 16), index=(8, 8, 8)):
    """Affine that puts the frozen sphere centre at ``index`` in a small grid."""
    voxel = script.VOXEL_MM
    affine = np.eye(4)
    affine[0, 0] = voxel
    affine[1, 1] = voxel
    affine[2, 2] = -voxel
    affine[0, 3] = script.SPHERE_CENTRE_MM[0] - voxel * index[2]
    affine[1, 3] = script.SPHERE_CENTRE_MM[1] - voxel * index[1]
    affine[2, 3] = script.SPHERE_CENTRE_MM[2] + voxel * index[0]
    return affine


@pytest.mark.unit
def test_centre_sensitivity_uses_identical_offsets_and_keeps_the_nominal_centre():
    shape = (16, 16, 16)
    affine = _centred_affine(shape)
    sphere = script.sphere_fraction(
        script.SPHERE_CENTRE_MM, script.SPHERE_RADIUS_MM, affine, shape=shape
    )
    assert script.mask_volume_ml(sphere) == pytest.approx(8.08, abs=0.1)
    liver_fraction = np.ones(shape)
    image = np.ones(shape)
    image[sphere > 0] = 4.0

    result = script.centre_sensitivity(
        {"first": image, "second": image.copy()}, liver_fraction, affine
    )

    nominal = script.voi_ratio(
        image, sphere, script.liver_excluding_sphere(liver_fraction, sphere)
    )
    assert result["strategies"]["first"]["nominal"] == pytest.approx(nominal)
    assert result["strategies"]["first"] == result["strategies"]["second"]
    assert result["centres"] == 125
    assert result["combined"]["all_min"] <= nominal <= result["combined"]["all_max"]
    assert result["ordering_changes_over_125"] == 0
    assert set(result["offsets_voxels"]) == {-2, -1, 0, 1, 2}
    assert "best" not in result


@pytest.mark.unit
def test_centre_sensitivity_reports_ordering_changes():
    shape = (16, 16, 16)
    affine = _centred_affine(shape)
    liver_fraction = np.ones(shape)
    k, j, i = np.meshgrid(*[np.arange(axis) for axis in shape], indexing="ij")
    first = np.ones(shape) + 0.1 * i
    second = np.ones(shape) + 0.1 * j

    result = script.centre_sensitivity(
        {"first": first, "second": second}, liver_fraction, affine
    )

    assert 0 < result["ordering_changes_over_125"] < 125
    assert result["strategies"]["first"]["nominal"] == pytest.approx(
        result["strategies"]["second"]["nominal"]
    )


@pytest.mark.unit
def test_centre_sensitivity_pm1_cube_includes_the_nominal_centre():
    shape = (16, 16, 16)
    affine = _centred_affine(shape)
    liver_fraction = np.ones(shape)
    k, j, i = np.meshgrid(*[np.arange(axis) for axis in shape], indexing="ij")
    image = np.ones(shape) + 0.1 * i

    # With a single offset the ±1 cube is exactly {(0, 0, 0)}: a selection that
    # drops the nominal centre would leave the cube empty and raise instead of
    # reporting a range.
    result = script.centre_sensitivity(
        {"first": image}, liver_fraction, affine, offsets=(0,)
    )

    combined = result["combined"]
    assert result["centres"] == 1
    assert combined["pm1_min"] == pytest.approx(combined["nominal_min"])
    assert combined["pm1_max"] == pytest.approx(combined["nominal_max"])


@pytest.mark.unit
def test_centre_sensitivity_pm1_range_reflects_an_extremum_at_the_nominal_centre():
    shape = (16, 16, 16)
    affine = _centred_affine(shape)
    sphere = script.sphere_fraction(
        script.SPHERE_CENTRE_MM, script.SPHERE_RADIUS_MM, affine, shape=shape
    )
    liver_fraction = np.ones(shape)
    image = np.ones(shape)
    image[sphere > 0] = 4.0

    # The hot region sits exactly at the frozen centre, so shifting the VOI
    # away lowers the sphere-to-liver ratio: the nominal centre is the maximum.
    result = script.centre_sensitivity({"first": image}, liver_fraction, affine)

    nominal = script.voi_ratio(
        image, sphere, script.liver_excluding_sphere(liver_fraction, sphere)
    )
    combined = result["combined"]
    assert combined["all_max"] == pytest.approx(nominal)
    assert combined["pm1_max"] == pytest.approx(nominal)
    assert combined["pm1_min"] <= nominal <= combined["pm1_max"]


@pytest.mark.unit
def test_centre_sensitivity_pm2_cube_covers_every_offset():
    shape = (16, 16, 16)
    affine = _centred_affine(shape)
    liver_fraction = np.ones(shape)
    k, j, i = np.meshgrid(*[np.arange(axis) for axis in shape], indexing="ij")
    first = np.ones(shape) + 0.1 * i
    second = np.ones(shape) + 0.1 * j

    result = script.centre_sensitivity(
        {"first": first, "second": second}, liver_fraction, affine
    )

    assert result["centres"] == 125
    assert set(result["offsets_voxels"]) == {-2, -1, 0, 1, 2}
    combined = result["combined"]
    assert combined["pm2_min"] == pytest.approx(combined["all_min"])
    assert combined["pm2_max"] == pytest.approx(combined["all_max"])
    assert combined["pm2_min"] <= combined["nominal_min"]
    assert combined["nominal_max"] <= combined["pm2_max"]
    for values in result["strategies"].values():
        assert values["pm2_min"] == pytest.approx(values["all_min"])
        assert values["pm2_max"] == pytest.approx(values["all_max"])
        assert values["pm1_min"] >= values["pm2_min"]
        assert values["pm1_max"] <= values["pm2_max"]


# %%
# Geometry and reduction


@pytest.mark.unit
def test_view_selection_preserves_the_selected_counts():
    array = np.arange(1 * 8 * 12 * 8, dtype=float).reshape(1, 8, 12, 8)

    identity = script.select_views(array, 1)
    reduced = script.select_views(array, 3)

    assert np.array_equal(identity, array)
    assert reduced.shape == (1, 8, 4, 8)
    assert reduced.sum() == pytest.approx(array[:, :, ::3, :].sum())
    assert reduced.sum() < array.sum()


@pytest.mark.unit
def test_reduced_template_keeps_one_radius_per_selected_view(tmp_path):
    header = tmp_path / "peak.hs"
    radii = ",".join(str(300.0 + index * 0.001) for index in range(12))
    header.write_text(
        "!INTERFILE :=\n"
        "name of data file := peak.s\n"
        "!number of projections := 12\n"
        "!extent of rotation := 360\n"
        "!direction of rotation := CCW\n"
        "start angle := 360\n"
        f"orbit := non-circular\nradii := {{{radii}}}\n"
        "matrix size [1] := 128\n"
        "scaling factor (mm/pixel) [1] := 4.2578\n"
        "matrix size [2] := 128\n"
        "scaling factor (mm/pixel) [2] := 4.2578\n"
        "!image duration (sec)[1] := 480\n"
        "!END OF INTERFILE :=\n"
    )

    reduced = script.reduced_template(header, tmp_path / "reduced.hs", 3)
    geometry = read_projection_geometry(read_header(reduced))

    assert geometry.num_projections == 4
    assert geometry.radii_mm is not None
    assert len(geometry.radii_mm) == 4
    assert geometry.radii_mm[-1] == pytest.approx(300.009)


# %%
# Live checks


@pytest.fixture(scope="session")
def y90_data():
    from simind_python_connector.datasets import fetch_y90_phantom

    try:
        return fetch_y90_phantom()
    except Exception as exc:  # noqa: BLE001 - the local dataset is optional
        pytest.skip(f"Y-90 local data unavailable: {exc}")


@pytest.mark.requires_sirf
def test_sirf_measured_geometry_and_reconstruction_settings(y90_data):
    sirf = pytest.importorskip("sirf.STIR")

    measured = sirf.AcquisitionData(str(y90_data / "peak.hs"))
    geometry = read_projection_geometry(read_header(y90_data / "peak.hs"))

    assert measured.dimensions() == (1, 128, 120, 128)
    assert float(measured.sum()) == pytest.approx(3.78e6, rel=1e-3)
    assert geometry.num_projections == 120
    assert geometry.num_bins == 128
    assert geometry.num_axial == 128
    assert geometry.bin_size_mm == pytest.approx(4.2578)
    header = read_header(y90_data / "peak.hs")
    assert header.as_dict()["orbit"] == "non-circular"
    radii = header.as_dict()["radii"].strip("{} ").split(",")
    assert len(radii) == 120
    assert np.ptp([float(value) for value in radii]) < 1e-3
    assert geometry.radius_mm == pytest.approx(300.0, abs=1e-3)

    attenuation = script._image_from_array(
        np.zeros((script.MATRIX_SIZE,) * 3, dtype=np.float32)
    )
    model = script._make_model(measured, attenuation)
    assert type(model).__name__ == "AcquisitionModelUsingMatrix"


@pytest.mark.requires_simind
@pytest.mark.requires_sirf
def test_sparse_real_engine_density_and_template_integration(tmp_path):
    if shutil.which("simind") is None:
        pytest.skip("SIMIND executable not on PATH")
    pytest.importorskip("sirf.STIR")

    from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder

    grid = 32
    voxel_mm = 8.0
    builder = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": str(grid),
            "!matrix size [2]": str(grid),
            "!number of projections": "2",
            "scaling factor (mm/pixel) [1]": str(voxel_mm),
            "scaling factor (mm/pixel) [2]": str(voxel_mm),
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": "200",
        },
        backend="sirf",
    )
    template = builder.build(output_path=tmp_path / "template")
    assert template.dimensions() == (1, grid, 2, grid)

    density = np.full((grid, grid, grid), 0.5, dtype=np.float32)
    density_file = tmp_path / "density.npy"
    np.save(density_file, density)
    source = np.zeros((grid, grid, grid), dtype=np.float32)
    source[14:18, 14:18, 14:18] = 1.0

    raw = script.simulate_single(
        "sparse",
        source,
        str(tmp_path / "template.hs"),
        density_file,
        tmp_path / "run",
        quantization_scale=0.05 * script.DEFAULT_QUANTIZATION_SCALE,
    )

    assert raw["all_interactions"].shape == (1, grid, 2, grid)
    assert raw["geom_coll_primary"].shape == (1, grid, 2, grid)
    assert float(raw["all_interactions"].sum()) > 0.0
    assert float(raw["geom_coll_primary"].sum()) > 0.0
