import math

import pytest

from simind_python_connector.utils.io_utils import temporary_directory
from simind_python_connector.utils.simind_utils import create_window_file
from simind_python_connector.utils.stir_utils import parse_sinogram


@pytest.mark.unit
def test_create_window_file_writes_expected_lines(tmp_path):
    win_stem = tmp_path / "energy"
    create_window_file(140.0, 160.0, 0, output_filename=str(win_stem))

    win_file = win_stem.with_suffix(".win")
    assert win_file.exists()

    lines = win_file.read_text().splitlines()
    assert lines == ["140.0,160.0,0"]


@pytest.mark.unit
def test_create_window_file_single_positive_order(tmp_path):
    win_stem = tmp_path / "energy"
    create_window_file(140.0, 160.0, 1, output_filename=str(win_stem))

    lines = win_stem.with_suffix(".win").read_text().splitlines()
    assert lines == ["140.0,160.0,1"]


@pytest.mark.unit
def test_create_window_file_multiple_all_zero_windows(tmp_path):
    win_stem = tmp_path / "energy"
    create_window_file(
        [120.0, 126.0], [140.0, 154.0], [0, 0], output_filename=str(win_stem)
    )

    lines = win_stem.with_suffix(".win").read_text().splitlines()
    assert lines == ["120.0,140.0,0", "126.0,154.0,0"]


@pytest.mark.unit
def test_create_window_file_multiple_windows_with_positive_order(tmp_path):
    win_stem = tmp_path / "energy"
    create_window_file(
        [100.0, 126.0, 160.0],
        [120.0, 154.0, 180.0],
        [1, 0, 1],
        output_filename=str(win_stem),
    )

    lines = win_stem.with_suffix(".win").read_text().splitlines()
    assert lines == ["100.0,120.0,1", "126.0,154.0,0", "160.0,180.0,1"]


@pytest.mark.unit
def test_create_window_file_rejects_empty_windows(tmp_path):
    with pytest.raises(ValueError, match="provided"):
        create_window_file([], [], [], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_rejects_mismatched_lengths(tmp_path):
    with pytest.raises(ValueError, match="same length"):
        create_window_file([120, 130], [140], [0], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_rejects_inverted_bounds(tmp_path):
    with pytest.raises(ValueError, match="lower"):
        create_window_file([150.0], [140.0], [0], output_filename=str(tmp_path / "w"))
    with pytest.raises(ValueError, match="lower"):
        create_window_file([140.0], [140.0], [0], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_rejects_non_finite_bounds(tmp_path):
    for bad in (math.nan, math.inf, -math.inf):
        with pytest.raises(ValueError, match="finite"):
            create_window_file([bad], [160.0], [0], output_filename=str(tmp_path / "w"))
        with pytest.raises(ValueError, match="finite"):
            create_window_file([120.0], [bad], [0], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_rejects_negative_scatter_orders(tmp_path):
    with pytest.raises(ValueError, match="[Ss]catter"):
        create_window_file([120.0], [140.0], [-1], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_rejects_non_integral_scatter_orders(tmp_path):
    with pytest.raises(ValueError, match="[Ss]catter"):
        create_window_file([120.0], [140.0], [1.5], output_filename=str(tmp_path / "w"))


@pytest.mark.unit
def test_create_window_file_overwrites_existing(tmp_path):
    win_file = tmp_path / "overwrite.win"
    win_file.write_text("old contents")

    create_window_file([120], [140], [1], output_filename=str(win_file))

    contents = win_file.read_text()
    assert contents != "old contents"
    assert "120.0,140.0,1" in contents


@pytest.mark.unit
def test_temporary_directory_context_manager():
    with temporary_directory() as tmpdir:
        assert tmpdir.exists()
        marker = tmpdir / "marker.txt"
        marker.write_text("ok")
        assert marker.exists()

    # Context manager should clean up the directory tree
    assert not tmpdir.exists()


@pytest.mark.unit
def test_parse_sinogram_from_path(tmp_path):
    header = tmp_path / "template.hs"
    header.write_text("!INTERFILE :=\n!matrix size [1] := 64\nstart angle := 180\n")

    values = parse_sinogram(header)
    assert values["!matrix size [1]"] == "64"
    assert values["start angle"] == "180"
