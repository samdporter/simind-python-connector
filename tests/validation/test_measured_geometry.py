import numpy as np
import pytest

from tests.test_measured_geometry_simind import (
    _ISOTROPIC,
    _N,
    _VIEWS,
    _VOXEL,
    PROFILE,
    _phantom,
    _simulate_and_compare,
    _template,
)


pytestmark = [
    pytest.mark.validation,
    pytest.mark.requires_simind,
    pytest.mark.requires_sirf,
]

assert PROFILE.name == "smoke", (
    "run at the A3 smoke profile; unset SIMIND_A3_FULL_VALIDATION"
)
NOTE = "A3 smoke profile; full profile ~20 h/case under amd64 emulation"


def _validate_geometry(validate, case, correlation, centre_gap):
    validate(
        f"{case}_correlation",
        float(correlation),
        lower=PROFILE.min_correlation,
        smoke=True,
        note=NOTE,
    )
    validate(
        f"{case}_centre_gap",
        float(centre_gap),
        upper=PROFILE.max_centre_gap,
        smoke=True,
        note=NOTE,
    )


@pytest.mark.parametrize("direction", ["cw", "ccw"])
def test_template_direction(tmp_path, validate, direction):
    activity, mu = _phantom()
    inputs = (activity, mu, _ISOTROPIC)
    correlation, centre_gap = _simulate_and_compare(
        direction,
        tmp_path,
        _template(tmp_path, direction.upper()),
        inputs,
        inputs,
    )
    _validate_geometry(validate, direction, correlation, centre_gap)


def test_non_circular_orbit(tmp_path, validate):
    angles = np.linspace(0, 2 * np.pi, _VIEWS, endpoint=False)
    radii = 250 + 30 * np.cos(2 * angles)
    activity, mu = _phantom()
    inputs = (activity, mu, _ISOTROPIC)
    correlation, centre_gap = _simulate_and_compare(
        "non-circular", tmp_path, _template(tmp_path, radii=radii), inputs, inputs
    )
    _validate_geometry(validate, "non_circular", correlation, centre_gap)


def test_anisotropic_voxels(tmp_path, validate):
    activity, mu = _phantom(dims=(_N // 2, _N, _N))
    simind_inputs = (activity, mu, (2 * _VOXEL, _VOXEL, _VOXEL))
    reference_inputs = (
        np.repeat(activity, 2, axis=0),
        np.repeat(mu, 2, axis=0),
        _ISOTROPIC,
    )
    correlation, centre_gap = _simulate_and_compare(
        "anisotropic", tmp_path, _template(tmp_path), simind_inputs, reference_inputs
    )
    _validate_geometry(validate, "anisotropic", correlation, centre_gap)
