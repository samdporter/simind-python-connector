import numpy as np
import pytest

from simind_python_connector.builders import STIRSPECTAcquisitionDataBuilder
from simind_python_connector.utils.interfile import (
    InterfileHeader,
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


def test_underscored_scaling_keys_load_unscaled(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    # STIR normalises underscores as spaces, so SIMIND-style underscored
    # keys must be recognised and reset by write_in_template_geometry too.
    header = read_header(template_path)
    header.set("image_scaling_factor[1]", "2.5")
    header.set("quantification_units", "2.5")
    header.write(template_path)
    assert "image_scaling_factor[1] := 2.5" in template_path.read_text()
    assert "quantification_units := 2.5" in template_path.read_text()

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array[10, 59, 50] == 1.0
    assert sirf_array[40, 3, 7] == 2.0
    assert sirf_array.sum() == 3.0

    written = read_header(path).as_dict()
    assert written["image scaling factor [1]"] == "1"
    assert written["quantification units"] == "1"


def test_boundary_underscore_scaling_keys_load_unscaled(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    # STIR trims boundary underscores, so leading/trailing '_' spellings must
    # be reset by write_in_template_geometry too.
    header = read_header(template_path)
    header.set("image_scaling_factor_[1]", "2.5")
    header.set("quantification_units_", "2.5")
    header.write(template_path)

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array.sum() == 3.0

    written = read_header(path).as_dict()
    assert written["image scaling factor [1]"] == "1"
    assert written["quantification units"] == "1"


@pytest.mark.parametrize(
    "offset_key", ["data offset in bytes[1]", "data_offset_in_bytes"]
)
def test_indexed_or_underscored_offset_keys_are_replaced_by_the_unindexed_key(
    tmp_path, offset_key
):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    header = read_header(template_path)
    header.set(offset_key, "512")
    header.write(template_path)
    assert f"{offset_key} := 512" in template_path.read_text()

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    written = read_header(path).as_dict()
    assert written["data offset in bytes"] == "0"
    assert "data offset in bytes [1]" not in written

    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert sirf_array.shape == (64, 60, 64)
    assert sirf_array[10, 59, 50] == 1.0
    assert sirf_array.sum() == 1.0


def test_keys_after_the_terminator_do_not_override_geometry(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    text = template_path.read_text().replace(
        "!END OF INTERFILE :=",
        "!END OF INTERFILE :=\n!direction of rotation := CCW",
    )
    template_path.write_text(text)

    assert read_projection_geometry(read_header(template_path)).direction == "CW"
    assert sirf.AcquisitionData(str(template_path)) is not None


def test_mixed_marker_underscore_scaling_keys_load_unscaled(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    header = read_header(template_path)
    header.set("image_scaling_factor!_[1]", "2.5")
    header.set("_!quantification_units_", "2.5")
    header.write(template_path)

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array.sum() == 3.0

    written = read_header(path).as_dict()
    assert written["image scaling factor [1]"] == "1"
    assert written["quantification units"] == "1"


def test_hash_commented_scaling_key_is_ignored_and_active_key_resets(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    # STIR treats a #-prefixed line as a comment (it does not scale), so the
    # '#'-filed duplicate is ignored; only the live 'quantification units'
    # key is reset to 1 by write_in_template_geometry.
    text = template_path.read_text().replace(
        "!END OF INTERFILE :=",
        "quantification units := 2.5\n"
        "#_!quantification_units_ := 9\n"
        "!END OF INTERFILE :=",
    )
    header = InterfileHeader.from_text(text)
    values = header.as_dict()
    assert values["quantification units"] == "2.5"

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0
    array[3, 40, 7] = 2.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    sirf_array = np.squeeze(copy.as_array())  # (axial, view, bin)
    assert np.array_equal(sirf_array, array.transpose(1, 0, 2))
    assert sirf_array.sum() == 3.0

    written = read_header(path).as_dict()
    assert written["quantification units"] == "1"
    assert "#_!quantification_units_ := 9" in path.read_text()


def test_hash_shadowed_direction_is_ignored_and_geometry_stays_compatible(tmp_path):
    import sirf.STIR as sirf

    _template(tmp_path)
    template_path = tmp_path / "template.hs"
    # A '#'-prefixed duplicate direction line is a comment for STIR, so the
    # acquisition must stay CW; the written copy must remain loadable and
    # geometrically compatible with the clean CW template.
    text = template_path.read_text().replace(
        "!END OF INTERFILE :=",
        "#_!direction_of_rotation_ := CCW\n!END OF INTERFILE :=",
    )
    header = InterfileHeader.from_text(text)
    assert read_projection_geometry(header).direction == "CW"

    array = np.zeros((60, 64, 64), dtype=np.float32)
    array[59, 10, 50] = 1.0

    path = write_in_template_geometry(array, header, tmp_path / "copy.hs")
    copy = sirf.AcquisitionData(str(path))
    _ = copy - sirf.AcquisitionData(str(template_path))  # geometry must match
    sirf_array = np.squeeze(copy.as_array())
    assert sirf_array[10, 59, 50] == 1.0
    assert sirf_array.sum() == 1.0
