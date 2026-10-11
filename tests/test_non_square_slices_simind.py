from pathlib import Path

import numpy as np
import pytest

from simind_python_connector import SimindPythonConnector
from simind_python_connector.configs import get


pytestmark = [pytest.mark.integration, pytest.mark.ci_skip, pytest.mark.requires_simind]

_VOXEL_MM = 4.0
_BIN_MM = 4.0


def _phantom() -> tuple[np.ndarray, np.ndarray]:
    """8 slices of 24 x 40 voxels: a hot cube 40 mm along x and -16 mm along y
    from the centre, inside an attenuating elliptical body."""
    source = np.zeros((8, 24, 40), np.float32)
    source[2:6, 6:10, 28:32] = 1.0
    yy, xx = np.mgrid[0:24, 0:40]
    body = ((xx + 0.5 - 20) / 19) ** 2 + ((yy + 0.5 - 12) / 11) ** 2 <= 1
    mu = np.zeros_like(source)
    mu[:, body] = 0.15
    return source, mu


def _simulate(output_dir: Path, source: np.ndarray, mu: np.ndarray) -> np.ndarray:
    connector = SimindPythonConnector(get("Example.yaml"), output_dir, "case01")
    connector.get_config().set_flag(11, True)
    connector.add_config_value(29, 4)
    connector.add_config_value(53, 0)
    connector.add_config_value(28, _BIN_MM / 10)
    connector.add_config_value(76, 48)
    connector.add_config_value(77, 8)
    connector.configure_voxel_phantom(source, mu, _VOXEL_MM)
    connector.set_energy_windows([126], [154], [0])
    for switch, value in (("FI", "tc99m"), ("CC", "ma-lehr"), ("NN", 10), ("RR", 7)):
        connector.add_runtime_switch(switch, value)
    return connector.run()["tot_w1"].projection


def test_non_square_slices_match_the_same_slices_padded_to_square(tmp_path: Path):
    source, mu = _phantom()
    padding = ((0, 0), (8, 8), (0, 0))
    projection = _simulate(tmp_path / "non_square", source, mu)
    padded = _simulate(
        tmp_path / "padded", np.pad(source, padding), np.pad(mu, padding)
    )

    # (views, axial rows, bins), sized by Index 29, 77 and 76, not by the image.
    assert projection.shape == padded.shape == (4, 8, 48)
    bins_mm = (np.arange(48) + 0.5 - 24) * _BIN_MM

    def centroids(p):
        per_view = p.sum(axis=1)
        return (per_view * bins_mm).sum(axis=1) / per_view.sum(axis=1)

    assert np.abs(centroids(projection)) == pytest.approx([40, 16, 40, 16], abs=1.0)
    assert centroids(projection) == pytest.approx(centroids(padded), abs=0.5)
    counts = projection.sum(axis=(1, 2))
    assert counts == pytest.approx(padded.sum(axis=(1, 2)), rel=0.03)
