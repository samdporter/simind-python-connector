from pathlib import Path

import numpy as np
import pytest

from simind_python_connector.utils.interfile import (
    InterfileHeader,
    ProjectionGeometry,
    load_interfile_array,
    normalise_key,
    parse_interfile_line,
    read_header,
    read_projection_geometry,
    read_simind_density_image,
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
