import pytest

from simind_python_connector import datasets


pytestmark = pytest.mark.unit


def _layout(root):
    for name in ("peak.dcm", "peak.hs", "peak.s", "umap_zoomed.hv", "umap_zoomed.v"):
        (root / name).write_bytes(b"x")
    (root / "ctac").mkdir()
    return root


def test_environment_directory_is_used_as_is(tmp_path, monkeypatch):
    monkeypatch.setenv("SIMIND_CONNECTOR_Y90_DATA", str(_layout(tmp_path)))
    assert datasets.fetch_y90_phantom() == tmp_path


def test_missing_files_are_listed(tmp_path, monkeypatch):
    _layout(tmp_path)
    (tmp_path / "peak.s").unlink()
    monkeypatch.setenv("SIMIND_CONNECTOR_Y90_DATA", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="peak.s"):
        datasets.fetch_y90_phantom()


def test_missing_environment_directory_lists_the_files(tmp_path, monkeypatch):
    missing = tmp_path / "no_such_directory"
    monkeypatch.setenv("SIMIND_CONNECTOR_Y90_DATA", str(missing))
    with pytest.raises(FileNotFoundError) as excinfo:
        datasets.fetch_y90_phantom()
    message = str(excinfo.value)
    assert str(missing) in message
    assert "peak.dcm" in message and "ctac" in message


def test_unpublished_record_points_at_the_environment_variable(monkeypatch):
    monkeypatch.delenv("SIMIND_CONNECTOR_Y90_DATA", raising=False)
    monkeypatch.setattr(datasets, "Y90_PHANTOM_RECORD", {"url": "", "sha256": ""})
    with pytest.raises(RuntimeError, match="SIMIND_CONNECTOR_Y90_DATA"):
        datasets.fetch_y90_phantom()


def test_published_record_without_a_checksum_is_rejected(monkeypatch):
    monkeypatch.delenv("SIMIND_CONNECTOR_Y90_DATA", raising=False)
    monkeypatch.setattr(
        datasets, "Y90_PHANTOM_RECORD", {"url": "https://x/y.zip", "sha256": ""}
    )
    with pytest.raises(RuntimeError, match="checksum"):
        datasets.fetch_y90_phantom()


def test_published_record_is_fetched_and_unzipped_with_its_hash(tmp_path, monkeypatch):
    pooch = pytest.importorskip("pooch")
    monkeypatch.delenv("SIMIND_CONNECTOR_Y90_DATA", raising=False)
    monkeypatch.setattr(
        datasets, "Y90_PHANTOM_RECORD", {"url": "https://x/y.zip", "sha256": "ab"}
    )
    seen = {}

    def fake_retrieve(url, known_hash, path, processor):
        seen.update(url=url, known_hash=known_hash, path=path, processor=processor)
        return [str(f) for f in _layout(tmp_path / "y90_phantom").rglob("*")]

    (tmp_path / "y90_phantom").mkdir()
    monkeypatch.setattr(pooch, "retrieve", fake_retrieve)
    assert datasets.fetch_y90_phantom(data_home=tmp_path) == tmp_path / "y90_phantom"
    assert seen["url"] == "https://x/y.zip" and seen["known_hash"] == "sha256:ab"
    assert isinstance(seen["processor"], pooch.Unzip)
