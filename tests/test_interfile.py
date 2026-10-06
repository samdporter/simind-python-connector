from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    check_geometry_match,
    load_interfile_array,
    normalise_key,
    parse_interfile_line,
    read_header,
    read_projection_geometry,
    read_simind_density_image,
    write_in_template_geometry,
)


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("!matrix size [1]", "matrix size [1]"),
        ("matrix size[1]", "matrix size [1]"),
        ("# scaling factor (mm/pixel) [3]", "scaling factor (mm/pixel) [3]"),
        ("Scaling Factor (mm/pixel)  [2]", "scaling factor (mm/pixel) [2]"),
        ("!image duration (sec)[1]", "image duration (sec) [1]"),
        ("  Radius ", "radius"),
        ("image_scaling_factor[1]", "image scaling factor [1]"),
        ("quantification_units", "quantification units"),
        ("data_offset_in_bytes", "data offset in bytes"),
        ("data offset in bytes[1]", "data offset in bytes [1]"),
        ("quantification_units_", "quantification units"),
        ("_data_offset_in_bytes[1]", "data offset in bytes [1]"),
        ("!_END_OF_INTERFILE:=", "end of interfile:="),
        ("_!quantification_units_", "quantification units"),
        ("_!data_offset_in_bytes[1]", "data offset in bytes [1]"),
        ("image_scaling_factor!_[1]", "image scaling factor [1]"),
        ("_!END_OF_INTERFILE:=", "end of interfile:="),
    ],
)
def test_normalise_key(raw, expected):
    assert normalise_key(raw) == expected


@pytest.mark.unit
def test_parse_interfile_line():
    assert parse_interfile_line("!matrix size [1] := 128") == (
        "!matrix size [1]",
        "128",
    )
    assert parse_interfile_line("scaling factor (mm/pixel) [1] := 4.419600") == (
        "scaling factor (mm/pixel) [1]",
        "4.419600",
    )
    assert parse_interfile_line(";# This is a comment") == (None, None)
    assert parse_interfile_line("!GENERAL DATA :=") == (None, None)
    assert parse_interfile_line("no separator here") == (None, None)


@pytest.mark.unit
def test_header_get_set_and_write(tmp_path):
    header_path = tmp_path / "test.hs"
    header_path.write_text(
        "!INTERFILE :=\npatient name := phantom\n!study ID := study123\n"
        "!END OF INTERFILE :=\n"
    )
    header = InterfileHeader.from_file(header_path)
    assert header.get("patient name") == "phantom"

    header.set("patient name", "updated")
    header.set("new field", 42)
    header.write(header_path)

    text = header_path.read_text()
    assert "patient name := updated" in text
    assert "new field := 42" in text


@pytest.mark.unit
def test_header_insert(tmp_path):
    header = InterfileHeader.from_text("a := 1\nc := 3\n")
    header.insert(1, "b", "{2, 2}")
    out = tmp_path / "insert.hs"
    header.write(out)
    assert out.read_text().splitlines()[1] == "b := {2, 2}"


@pytest.mark.unit
def test_header_lookup_ignores_prefix_case_and_spacing():
    header = InterfileHeader.from_text(
        "scaling factor (mm/pixel) [1] := 4.42\n# Image Position := 1 2 3\n"
    )
    assert header.get("!scaling factor (mm/pixel) [1]") == "4.42"
    assert header.get("Scaling Factor (mm/pixel)[1]") == "4.42"
    assert header.get("image position") == "1 2 3"


@pytest.mark.unit
def test_header_duplicate_keys_use_last_and_set_edits_it():
    header = InterfileHeader.from_text("!start angle := 0\nstart angle := 180\n")
    assert header.get("start angle") == "180"
    header.set("start angle", "90")
    assert header.items() == [("!start angle", "0"), ("start angle", "90")]


@pytest.mark.unit
@pytest.mark.parametrize(
    "key",
    ["imagedata byte order", "data offset in bytes[1]"],
)
def test_header_set_inserts_a_missing_key_before_the_terminator(tmp_path, key):
    header = InterfileHeader.from_text(
        "!INTERFILE :=\n"
        "!number of projections := 3\n"
        "!extent of rotation := 360\n"
        "!END OF INTERFILE :=\n"
    )
    assert key not in header.as_dict()

    header.set(key, 0)
    path = tmp_path / "header.hs"
    header.write(path)

    lines = path.read_text().splitlines()
    inserted = next(i for i, line in enumerate(lines) if line.startswith(key))
    terminator = next(i for i, line in enumerate(lines) if "!END OF INTERFILE" in line)
    assert inserted < terminator
    assert terminator == len(lines) - 1


@pytest.mark.unit
def test_header_set_appends_a_missing_key_when_no_terminator():
    header = InterfileHeader.from_text("!a := 1\n")
    header.set("b", 2)
    assert header.items() == [("!a", "1"), ("b", "2")]


@pytest.mark.unit
@pytest.mark.parametrize(
    "terminator",
    [
        "!END OF INTERFILE :=",
        "!END OF INTERFILE:=",
        "!_END_OF_INTERFILE:=",
        "_!END_OF_INTERFILE:=",
    ],
)
def test_header_set_inserts_before_both_terminator_spellings(tmp_path, terminator):
    header = InterfileHeader.from_text(f"a := 1\n{terminator}\n")

    header.set("b", 2)
    path = tmp_path / "header.hs"
    header.write(path)
    lines = path.read_text().splitlines()
    assert lines[1].startswith("b :=")
    assert lines[-1] == terminator


@pytest.mark.unit
def test_header_remove_drops_every_matching_entry():
    header = InterfileHeader.from_text(
        "DATA_OFFSET_IN_BYTES := 8\n"
        "name of data file := data.s\n"
        "!data offset in bytes := 512\n"
        "a := 1\n"
    )
    header.remove("data offset in bytes")
    assert header.get("data offset in bytes") is None
    assert header.items() == [("name of data file", "data.s"), ("a", "1")]


@pytest.mark.unit
def test_set_all_updates_every_normalised_equal_entry():
    header = InterfileHeader.from_text(
        "quantification units := 2.5\n#_!quantification_units_ := 9\na := 1\n"
    )

    header.set_all("quantification units", 1)

    values = [
        value
        for key, value in header.items()
        if normalise_key(key) == "quantification units"
    ]
    assert values == ["1", "1"]
    assert header.get("a") == "1"


@pytest.mark.unit
def test_write_in_template_geometry_resets_shadowed_scaling_and_data_file(tmp_path):
    template = InterfileHeader.from_text(
        "!matrix size [1] := 4\n"
        "!matrix size [2] := 3\n"
        "!number of projections := 2\n"
        "!number format := unsigned integer\n"
        "!number of bytes per pixel := 2\n"
        "imagedata byte order := BIGENDIAN\n"
        "!name of data file := template.s\n"
        "quantification units := 2.5\n"
        "#_!quantification_units_ := 9\n"
        "name of data file := template.s\n"
        "!END OF INTERFILE :=\n"
    )
    array = np.zeros((2, 3, 4), dtype=np.float32)

    path = write_in_template_geometry(array, template, tmp_path / "copy.hs")

    written = InterfileHeader.from_file(path)
    scaling = [
        value
        for key, value in written.items()
        if normalise_key(key) == "quantification units"
    ]
    names = [
        value
        for key, value in written.items()
        if normalise_key(key) == "name of data file"
    ]
    assert scaling == ["1", "1"]
    assert names == ["copy.s", "copy.s"]
    assert load_interfile_array(path).array.size == 24


@pytest.mark.unit
def test_header_handles_windows_line_endings():
    header = InterfileHeader.from_text("!matrix size [1] := 64\r\nRadius := 200\r\n")
    assert header.get("matrix size [1]") == "64"
    assert header.get("radius") == "200"


@pytest.mark.unit
def test_as_dict_uses_normalised_keys():
    header = InterfileHeader.from_text(
        "!INTERFILE :=\nname of data file := test.v\n!matrix size [1] := 64\n"
        "scaling factor (mm/pixel) [1] := 4.42\n!END OF INTERFILE :=\n"
    )
    values = header.as_dict()
    assert values["name of data file"] == "test.v"
    assert values["matrix size [1]"] == "64"
    assert values["scaling factor (mm/pixel) [1]"] == "4.42"


@pytest.mark.unit
def test_read_header_from_path(tmp_path):
    path = tmp_path / "template.hs"
    path.write_text("!INTERFILE :=\n!matrix size [1] := 64\nstart angle := 180\n")
    header = read_header(path)
    assert header.get("matrix size [1]") == "64"
    assert header.get("start angle") == "180"


@pytest.mark.unit
def test_read_header_from_object_with_write():
    class _InMemory:
        def write(self, path):
            Path(path).write_text("!number of projections := 60\n")

    assert read_header(_InMemory()).get("number of projections") == "60"


@pytest.mark.unit
def test_read_header_rejects_unsupported_objects():
    with pytest.raises(TypeError, match="path or an object"):
        read_header(object())


_CIRCULAR_HEADER = """!INTERFILE :=
!number of projections := 60
!extent of rotation := 360
!direction of rotation := CW
start angle := 180
Radius := 250
!matrix size [1] := 64
scaling factor (mm/pixel) [1] := 4.42
!matrix size [2] := 32
scaling factor (mm/pixel) [2] := 4.42
!image duration (sec)[1] := 15.5
!END OF INTERFILE :=
"""


@pytest.mark.unit
def test_read_projection_geometry_circular():
    geometry = read_projection_geometry(InterfileHeader.from_text(_CIRCULAR_HEADER))
    assert geometry == ProjectionGeometry(
        num_projections=60,
        extent_deg=360.0,
        direction="CW",
        start_angle_deg=180.0,
        radius_mm=250.0,
        radii_mm=None,
        num_bins=64,
        num_axial=32,
        bin_size_mm=4.42,
        axial_size_mm=4.42,
        image_duration_s=15.5,
    )


@pytest.mark.unit
def test_read_projection_geometry_non_circular():
    text = _CIRCULAR_HEADER.replace("Radius := 250", "Radii := {240, 250.5, 260}")
    geometry = read_projection_geometry(InterfileHeader.from_text(text))
    assert geometry.radius_mm is None
    assert geometry.radii_mm == (240.0, 250.5, 260.0)


@pytest.mark.unit
def test_read_projection_geometry_equal_radii_is_circular():
    text = _CIRCULAR_HEADER.replace("Radius := 250", "Radii := {250, 250, 250}")
    geometry = read_projection_geometry(InterfileHeader.from_text(text))
    assert geometry.radius_mm == 250.0
    assert geometry.radii_mm is None


@pytest.mark.unit
def test_read_projection_geometry_lists_missing_keys():
    with pytest.raises(ValueError) as excinfo:
        read_projection_geometry(InterfileHeader.from_text("!INTERFILE :=\n"))
    message = str(excinfo.value)
    assert "number of projections" in message
    assert "radius or radii" in message


@pytest.mark.unit
def test_read_projection_geometry_rejects_unknown_direction():
    text = _CIRCULAR_HEADER.replace(":= CW", ":= SIDEWAYS")
    with pytest.raises(ValueError, match="direction of rotation"):
        read_projection_geometry(InterfileHeader.from_text(text))


@pytest.mark.unit
def test_read_simind_density_image(tmp_path):
    header = tmp_path / "map.hct"
    header.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!matrix size [1] := 2",
                "!matrix size [2] := 3",
                "!matrix size [3] := 4",
                "scaling factor (mm/pixel) [1] := 1.5",
                "scaling factor (mm/pixel) [2] := 2.5",
                "# scaling factor (mm/pixel) [3] := 3.5",
            ]
        )
    )
    values = np.arange(24, dtype="<u2")
    (tmp_path / "map.ict").write_bytes(values.tobytes())

    density, voxel_sizes = read_simind_density_image(header)

    assert density.shape == (4, 3, 2)
    assert density.dtype == np.float32
    assert np.allclose(density.ravel(), values / 1000.0)
    assert voxel_sizes == (3.5, 2.5, 1.5)


@pytest.mark.unit
def test_read_simind_density_image_rejects_wrong_size(tmp_path):
    header = tmp_path / "map.hct"
    header.write_text(
        "!matrix size [1] := 2\n!matrix size [2] := 2\n!matrix size [3] := 2\n"
        "scaling factor (mm/pixel) [1] := 1\nscaling factor (mm/pixel) [2] := 1\n"
        "scaling factor (mm/pixel) [3] := 1\n"
    )
    (tmp_path / "map.ict").write_bytes(b"\x00\x00" * 3)
    with pytest.raises(ValueError, match="expected 8"):
        read_simind_density_image(header)


@pytest.mark.unit
@pytest.mark.parametrize(
    "offset_key",
    (
        "!data offset in bytes",
        "data offset in bytes",
        "data_offset_in_bytes",
    ),
)
def test_load_interfile_array_respects_data_offset(tmp_path: Path, offset_key: str):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    prefix = b"\xde\xad\xbe\xef"
    data_path.write_bytes(prefix + data.tobytes())

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                f"{offset_key} := 4",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)

    assert loaded.data_path == data_path.resolve()
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_strips_quoted_data_filename(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                '!name of data file := "projection.a00"',
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)

    assert loaded.data_path == data_path.resolve()
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_rejects_truncated_payload(tmp_path: Path):
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    np.arange(20, dtype=np.float32).tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    with pytest.raises(ValueError, match="Data size mismatch"):
        load_interfile_array(header_path)


@pytest.mark.unit
def test_load_interfile_array_rejects_trailing_payload(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    header_path = tmp_path / "projection.hs"
    (tmp_path / "projection.a00").write_bytes(data.tobytes() + b"\x00\x00\x00\x00")

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    with pytest.raises(ValueError, match="Data size mismatch"):
        load_interfile_array(header_path)


@pytest.mark.unit
def test_load_interfile_array_round_trip(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!matrix size [3] := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)

    assert loaded.header_path == header_path.resolve()
    assert loaded.data_path == data_path.resolve()
    assert loaded.array.shape == (2, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_requires_matrix_sizes(tmp_path: Path):
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    np.zeros(8, dtype=np.float32).tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    with pytest.raises(ValueError, match="matrix size"):
        load_interfile_array(header_path)


@pytest.mark.unit
def test_load_interfile_array_infers_projection_axis_from_header(tmp_path: Path):
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!number of projections := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (2, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_rejects_projection_count_payload_mismatch(
    tmp_path: Path,
):
    """A header projection count below the payload must be an error, not a
    silent truncation: the extra planes may be additional energy windows or
    corrupted data."""
    data = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!number of projections := 2",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    with pytest.raises(ValueError, match="Data size mismatch"):
        load_interfile_array(header_path)


@pytest.mark.unit
def test_load_interfile_array_infers_leading_axis_when_header_omits_count(
    tmp_path: Path,
):
    data = np.arange(3 * 3 * 4, dtype=np.float32).reshape(3, 3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (3, 3, 4)
    assert np.array_equal(loaded.array, data)


@pytest.mark.unit
def test_load_interfile_array_promotes_single_plane_to_3d(tmp_path: Path):
    data = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    data_path = tmp_path / "projection.a00"
    header_path = tmp_path / "projection.hs"
    data.tofile(data_path)

    header_path.write_text(
        "\n".join(
            [
                "!INTERFILE :=",
                "!number format := float",
                "!number of bytes per pixel := 4",
                "imagedata byte order := LITTLEENDIAN",
                "!matrix size [1] := 4",
                "!matrix size [2] := 3",
                "!name of data file := projection.a00",
                "!END OF INTERFILE :=",
            ]
        )
    )

    loaded = load_interfile_array(header_path)
    assert loaded.array.shape == (1, 3, 4)
    assert np.array_equal(loaded.array[0], data)


def _geometry(**changes):
    values = dict(
        num_projections=60,
        extent_deg=360.0,
        direction="CW",
        start_angle_deg=180.0,
        radius_mm=250.0,
        radii_mm=None,
        num_bins=64,
        num_axial=32,
        bin_size_mm=4.42,
        axial_size_mm=4.42,
        image_duration_s=1200.0,
    )
    values.update(changes)
    return ProjectionGeometry(**values)


@pytest.mark.unit
def test_time_per_projection():
    assert _geometry().time_per_projection_s == pytest.approx(20.0)
    assert _geometry(image_duration_s=None).time_per_projection_s is None


@pytest.mark.unit
def test_geometry_match_within_tolerances():
    template = _geometry(start_angle_deg=359.8)
    produced = _geometry(
        start_angle_deg=0.1, extent_deg=360.4, radius_mm=250.4, bin_size_mm=4.4205
    )
    assert check_geometry_match(produced, template) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes, word",
    [
        ({"num_projections": 59}, "num_projections"),
        ({"num_bins": 63}, "num_bins"),
        ({"num_axial": 31}, "num_axial"),
        ({"direction": "CCW"}, "direction"),
        ({"extent_deg": 359.0}, "extent"),
        ({"start_angle_deg": 181.0}, "start angle"),
        ({"bin_size_mm": 4.43}, "bin_size_mm"),
        ({"axial_size_mm": 4.43}, "axial_size_mm"),
        ({"radius_mm": 251.0}, "radii"),
    ],
)
def test_geometry_mismatch_is_reported(changes, word):
    differences = check_geometry_match(_geometry(**changes), _geometry())
    assert any(word in difference for difference in differences)


@pytest.mark.unit
def test_geometry_match_compares_non_circular_radii():
    radii = tuple(240.0 + i for i in range(60))
    template = _geometry(radius_mm=None, radii_mm=radii)
    assert (
        check_geometry_match(_geometry(radius_mm=None, radii_mm=radii), template) == []
    )
    shifted = tuple(r + 1.0 for r in radii)
    produced = _geometry(radius_mm=None, radii_mm=shifted)
    assert "radii" in check_geometry_match(produced, template)[0]


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes",
    [
        {"extent_deg": 360.5},
        {"start_angle_deg": 180.5},
        {"start_angle_deg": -179.5},
        {"bin_size_mm": 4.4205},
        {"axial_size_mm": 4.4205},
        {"radius_mm": 250.5},
    ],
)
def test_geometry_match_accepts_differences_inside_tolerance(changes):
    assert check_geometry_match(_geometry(**changes), _geometry()) == []


@pytest.mark.unit
@pytest.mark.parametrize(
    "changes, word",
    [
        ({"extent_deg": 360.6}, "extent"),
        ({"start_angle_deg": 180.6}, "start angle"),
        ({"start_angle_deg": -179.4}, "start angle"),
        ({"bin_size_mm": 4.4211}, "bin_size_mm"),
        ({"axial_size_mm": 4.4211}, "axial_size_mm"),
        ({"radius_mm": 250.6}, "radii"),
    ],
)
def test_geometry_match_rejects_differences_outside_tolerance(changes, word):
    differences = check_geometry_match(_geometry(**changes), _geometry())
    assert any(word in difference for difference in differences)


@pytest.mark.unit
def test_geometry_match_compares_each_non_circular_radius():
    radii = tuple(240.0 + i for i in range(60))
    template = _geometry(radius_mm=None, radii_mm=radii)
    inside = tuple(r + 0.5 * i / len(radii) for i, r in enumerate(radii))
    assert (
        check_geometry_match(_geometry(radius_mm=None, radii_mm=inside), template) == []
    )
    outside = tuple(r + (0.6 if i == 7 else 0.0) for i, r in enumerate(radii))
    assert check_geometry_match(
        _geometry(radius_mm=None, radii_mm=outside), template
    ) == ["radii differ by more than 0.5 mm"]


@pytest.mark.unit
def test_geometry_match_reports_every_difference():
    differences = check_geometry_match(
        _geometry(
            num_projections=59,
            num_bins=63,
            direction="CCW",
            extent_deg=359.0,
            start_angle_deg=181.0,
            bin_size_mm=4.43,
            axial_size_mm=4.43,
            radius_mm=251.0,
        ),
        _geometry(),
    )
    assert len(differences) == 8
    for word in (
        "num_projections",
        "num_bins",
        "direction",
        "extent",
        "start angle",
        "bin_size_mm",
        "axial_size_mm",
        "radii",
    ):
        assert any(word in difference for difference in differences)


_TEMPLATE_TEXT = """!INTERFILE :=
  !name of data file := template.s
  !number format := short float
  !number of bytes per pixel := 2
  imagedata byte order := BIGENDIAN
  data offset in bytes[1] := 512
  !number of projections := 3
  image scaling factor [1] := 2.5
  image scaling factor [2] := 2.5
  quantification units := 0.5
  !matrix size [1] := 4
  !matrix size [2] := 2
  !END OF INTERFILE :=
  """


@pytest.mark.unit
def test_write_in_template_geometry_rewrites_data_keys(tmp_path):
    template = InterfileHeader.from_text(_TEMPLATE_TEXT)
    array = np.arange(24, dtype=np.float64).reshape(3, 2, 4)

    path = write_in_template_geometry(array, template, tmp_path / "copy.hs")

    header = InterfileHeader.from_file(path)
    assert header.get("name of data file") == "copy.s"
    assert header.get("number format") == "float"
    assert header.get("number of bytes per pixel") == "4"
    assert header.get("imagedata byte order") == "LITTLEENDIAN"
    assert header.get("data_offset_in_bytes") == "0"
    assert header.get("data offset in bytes[1]") is None
    assert header.get("number of projections") == "3"
    loaded = load_interfile_array(path)
    expected_storage = array.astype("<f4")
    assert np.array_equal(loaded.array.ravel(), expected_storage.ravel())


@pytest.mark.unit
def test_write_in_template_geometry_checks_size(tmp_path):
    template = InterfileHeader.from_text(_TEMPLATE_TEXT)
    with pytest.raises(ValueError, match="expected 24"):
        write_in_template_geometry(np.zeros(23), template, tmp_path / "copy.hs")


def _template(offset_key=None):
    """_TEMPLATE_TEXT with its data-offset line dropped or re-spelled."""
    lines = [
        line
        for line in _TEMPLATE_TEXT.splitlines()
        if "data offset in bytes" not in line and "data_offset_in_bytes" not in line
    ]
    if offset_key is not None:
        lines.insert(4, f"{offset_key} := 512")
    return InterfileHeader.from_text("\n".join(lines) + "\n")


@pytest.mark.unit
@pytest.mark.parametrize(
    "template_key",
    [
        "data offset in bytes[1]",
        "data_offset_in_bytes[1]",
        "_data_offset_in_bytes[1]",
        "data_offset_in_bytes_",
        "_!data_offset_in_bytes[1]",
        None,
    ],
)
def test_write_in_template_geometry_zeroes_the_data_offset(tmp_path, template_key):
    template = _template(template_key)
    array = np.arange(24, dtype=np.float64).reshape(3, 2, 4)

    path = write_in_template_geometry(array, template, tmp_path / "copy.hs")

    written = InterfileHeader.from_file(path)
    assert written.get("data_offset_in_bytes") == "0"
    assert written.get("data offset in bytes[1]") is None
    assert load_interfile_array(path).array.size == 24


@pytest.mark.unit
def test_write_in_template_geometry_writes_little_endian_float32(tmp_path):
    template = InterfileHeader.from_text(_TEMPLATE_TEXT)
    array = np.arange(24, dtype=np.float64).reshape(3, 2, 4)

    path = write_in_template_geometry(array, template, tmp_path / "copy.hs")

    payload = path.with_suffix(".s").read_bytes()
    assert payload == array.astype("<f4").tobytes()
    assert payload != array.astype(">f4").tobytes()


@pytest.mark.unit
def test_write_in_template_geometry_keeps_geometry_keys_and_the_template(tmp_path):
    template = InterfileHeader.from_text(_TEMPLATE_TEXT)
    before = template.as_dict()
    array = np.arange(24, dtype=np.float64).reshape(3, 2, 4)

    path = write_in_template_geometry(array, template, tmp_path / "copy.hs")

    rewritten = {
        "name of data file",
        "number format",
        "number of bytes per pixel",
        "imagedata byte order",
        "data offset in bytes [1]",  # removed; replaced by the unindexed key
        "image scaling factor [1]",
        "image scaling factor [2]",
        "quantification units",
    }
    written = InterfileHeader.from_file(path).as_dict()
    for key, value in before.items():
        if key not in rewritten:
            assert written[key] == value
    assert written["data offset in bytes"] == "0"
    assert "data offset in bytes [1]" not in written
    assert written["image scaling factor [1]"] == "1"
    assert written["image scaling factor [2]"] == "1"
    assert written["quantification units"] == "1"
    assert template.as_dict() == before
