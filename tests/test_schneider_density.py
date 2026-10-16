"""Schneider table units, interpolation, tissue boundaries and comparisons."""

import numpy as np
import pytest

from simind_python_connector.converters.attenuation import (
    compare_density_methods,
    get_schneider_tissue_info,
    hu_to_density,
    hu_to_density_schneider,
    hu_to_density_schneider_piecewise,
    load_schneider_data,
)


pytestmark = pytest.mark.unit


def test_schneider_table():
    data = load_schneider_data()
    assert len(data) == 44
    for tissue in data.values():
        assert tissue["HU_lo (HU)"] <= tissue["HU_hi (HU)"]
        assert tissue["density (mg/cm3)"] > 0
    assert data["Air_0"]["density (mg/cm3)"] == 1.21


def test_interpolation_between_air_and_lung_centres():
    # Table centres are -1000 and -901.442 HU; densities are given in mg/cm3.
    hu = np.array([-1000.0, -950.721, -901.442])
    np.testing.assert_allclose(
        hu_to_density_schneider(hu), [0.00121, 0.0519525, 0.102695], rtol=1e-6
    )


def test_piecewise_conversion_uses_tissue_boundaries():
    hu = np.array([-1050.0, -1000.0, -950.01, -950.0, -900.0])
    np.testing.assert_allclose(
        hu_to_density_schneider_piecewise(hu),
        [0.00121, 0.00121, 0.00121, 0.102695, 0.102695],
        rtol=1e-6,
    )


@pytest.mark.parametrize(
    "convert", [hu_to_density_schneider, hu_to_density_schneider_piecewise]
)
def test_outside_table_uses_endpoint_densities(convert):
    np.testing.assert_allclose(
        convert(np.array([-5000.0, 10000.0])), [0.00121, 2.9], rtol=1e-6
    )


@pytest.mark.parametrize(
    "convert", [hu_to_density_schneider, hu_to_density_schneider_piecewise]
)
@pytest.mark.parametrize("shape", [(24,), (4, 6), (2, 3, 4)])
def test_conversion_preserves_array_order_and_shape(convert, shape):
    hu = np.linspace(-1000, 1000, 24)
    converted = convert(hu.reshape(shape))
    assert converted.shape == shape
    np.testing.assert_allclose(converted.ravel(), convert(hu))


@pytest.mark.parametrize(
    "hu, name, density",
    [
        (-1050.0, "Air_0", 0.00121),
        (-950.01, "Air_0", 0.00121),
        (-950.0, "Lung_1", 0.102695),
        (4001.0, "MetallImplants_43", 2.9),
    ],
)
def test_tissue_lookup_includes_lower_and_final_upper_boundaries(hu, name, density):
    info = get_schneider_tissue_info(hu)
    assert info["name"] == name
    assert info["density_g_cm3"] == pytest.approx(density)


@pytest.mark.parametrize("hu", [-5000, 10000])
def test_tissue_lookup_outside_table_returns_none(hu):
    assert get_schneider_tissue_info(hu) is None


def test_comparison_reports_maps_and_absolute_difference_statistics():
    hu = np.array([[-1000.0, -500.0, 0.0], [50.0, 500.0, 1000.0]])
    comparison = compare_density_methods(hu)
    bilinear = hu_to_density(hu)
    np.testing.assert_allclose(comparison["bilinear"], bilinear)
    for method, suffix, convert in (
        ("schneider_interpolated", "interp", hu_to_density_schneider),
        ("schneider_piecewise", "piecewise", hu_to_density_schneider_piecewise),
    ):
        expected = convert(hu)
        np.testing.assert_allclose(comparison[method], expected)
        difference = expected - bilinear
        difference_key = (
            "difference_interpolated" if suffix == "interp" else "difference_piecewise"
        )
        np.testing.assert_allclose(comparison[difference_key], difference)
        assert comparison[f"max_diff_{suffix}"] == pytest.approx(abs(difference).max())
        assert comparison[f"mean_diff_{suffix}"] == pytest.approx(
            abs(difference).mean()
        )
