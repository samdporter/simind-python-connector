import numpy as np
import pytest

from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder
from simind_python_connector.utils.interfile import (
    read_header,
    read_projection_geometry,
    write_in_template_geometry,
)


pytestmark = pytest.mark.requires_sirf


def _template(tmp_path):
    # The spec's template: 60 views, 64 x 64 bins of 4.42 mm, CW, start 180,
    # radius 250 mm.
    builder = STIRSPECTAcquisitionDataBuilder(
        header_overrides={
            "!matrix size [1]": "64",
            "!matrix size [2]": "64",
            "!number of projections": "60",
            "scaling factor (mm/pixel) [1]": "4.42",
            "scaling factor (mm/pixel) [2]": "4.42",
            "!direction of rotation": "CW",
            "start angle": "180",
            "Radius": "250",
        },
        backend="sirf",
    )
    return builder.build(output_path=tmp_path / "template")


def test_template_header_has_a_readable_geometry(tmp_path):
    geometry = read_projection_geometry(read_header(_template(tmp_path)))
    assert (geometry.num_projections, geometry.num_bins, geometry.num_axial) == (
        60,
        64,
        64,
    )
    assert geometry.direction == "CW"
    assert geometry.radius_mm == pytest.approx(250.0)


def test_template_geometry_round_trips_through_sirf(tmp_path):
    import sirf.STIR as sirf

    template = _template(tmp_path)
    header = read_header(template)
    # SIMIND order: (projection, axial, bin). The marked positions use different
    # indices on every axis, so any axis permutation moves them.
    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    _ = copy - template  # arithmetic only works for matching geometry

    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert sirf_array.shape == (64, 60, 64)
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array[10, 59, 50] == 1.0
    assert sirf_array[40, 3, 7] == 2.0
    assert sirf_array.sum() == 3.0


def test_write_in_template_geometry_loads_non_unit_scaling_unscaled(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    header = read_header(tmp_path / "template.hs")
    # STIR multiplies the payload by image scaling factor and quantification
    # units, so the template may carry non-unit values. They must not leak
    # into the written copy, whose payload is the raw counts already.
    header.set("image scaling factor [1]", "2.5")
    header.set("quantification units", "2.5")
    assert header.get("image scaling factor [1]") == "2.5"
    assert header.get("quantification units") == "2.5"
    header.write(tmp_path / "template.hs")

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    written = read_header(path).as_dict()
    assert written["image scaling factor [1]"] == "1"
    assert written["quantification units"] == "1"

    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert sirf_array.shape == (64, 60, 64)
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array[10, 59, 50] == 1.0
    assert sirf_array[40, 3, 7] == 2.0
    assert sirf_array.sum() == 3.0
