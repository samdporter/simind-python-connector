"""Check physical source geometry with the real SIMIND executable."""

import numpy as np
import pytest

from simind_python_connector.configs import get
from simind_python_connector.connectors import SimindPythonConnector


pytestmark = [pytest.mark.integration, pytest.mark.requires_simind]


def test_thick_slices_match_the_same_phantom_split_into_cubic_voxels(tmp_path):
    source = np.zeros((8, 16, 24), dtype=np.float32)
    source[5:7, 4:6, 16:18] = 1
    y, x = np.indices(source.shape[1:])
    body = ((x + 0.5 - 12) / 11) ** 2 + ((y + 0.5 - 8) / 7) ** 2 < 1
    mu = np.zeros_like(source)
    mu[:, body] = 0.15

    def simulate(name, activity, attenuation, spacing):
        connector = SimindPythonConnector(get("Example.yaml"), tmp_path / name, name)
        connector.get_config().set_flag(11, True)
        for index, value in ((29, 4), (53, 0), (28, 0.4), (76, 32), (77, 16)):
            connector.add_config_value(index, value)
        connector.configure_voxel_phantom(activity, attenuation, spacing)
        connector.set_energy_windows([126], [154], [0])
        for switch, value in (
            ("FI", "tc99m"),
            ("CC", "ma-lehr"),
            ("NN", 10),
            ("RR", 7),
        ):
            connector.add_runtime_switch(switch, value)
        return connector.run()["tot_w1"].projection

    thick = simulate("thick", source, mu, (8.0, 4.0, 4.0))
    cubic = simulate("cubic", source.repeat(2, axis=0), mu.repeat(2, axis=0), 4.0)
    assert thick.shape == cubic.shape == (4, 16, 32)
    axial_mm = (np.arange(16) + 0.5 - 8) * 4
    bins_mm = (np.arange(32) + 0.5 - 16) * 4

    def centroids(projection):
        counts = projection.sum(axis=(1, 2))
        axial = (projection.sum(axis=2) * axial_mm).sum(axis=1) / counts
        transverse = (projection.sum(axis=1) * bins_mm).sum(axis=1) / counts
        return axial, transverse

    axial, transverse = centroids(thick)
    reference_axial, reference_transverse = centroids(cubic)
    np.testing.assert_allclose(axial, reference_axial, atol=1.0)
    np.testing.assert_allclose(transverse, reference_transverse, atol=1.0)
    np.testing.assert_allclose(np.abs(axial), 16, atol=1.5)
    np.testing.assert_allclose(np.abs(transverse), [20, 12, 20, 12], atol=1.5)
