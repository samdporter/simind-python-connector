import numpy as np
import pytest

from simind_python_connector.recon import projector as projector_mod
from simind_python_connector.recon.projector import (
    SimindProjector,
    reference_normaliser,
)
from tests.recon.fakes import FakeAdaptor, FakeData


pytestmark = pytest.mark.unit


def _adaptor():
    return FakeAdaptor(
        {
            "geom_coll_primary": FakeData(np.full((2, 3, 4), 2.0)),
            "all_interactions": FakeData(np.full((2, 3, 4), 5.0)),
        }
    )


def _unit(outputs, image):
    return 1.0


@pytest.mark.parametrize(
    "kwargs", [{}, {"normalise": _unit, "activity_mbq": lambda x: 1.0}]
)
def test_projector_needs_exactly_one_scaling_route(kwargs):
    with pytest.raises(ValueError, match="exactly one"):
        SimindProjector(_adaptor(), "template", "mu", **kwargs)


def test_route_two_scales_every_output_and_clips_the_image():
    adaptor = _adaptor()
    seen = {}

    def normalise(outputs, image):
        seen["outputs"], seen["image"] = outputs, image
        return 3.0

    projector = SimindProjector(adaptor, "template", "mu", normalise=normalise)
    outputs = projector.project(FakeData([[[-1.0, 2.0]]]))

    assert projector.last_scale == 3.0
    assert projector.last_outputs is outputs
    np.testing.assert_allclose(outputs["geom_coll_primary"].as_array(), 6.0)
    np.testing.assert_allclose(outputs["all_interactions"].as_array(), 15.0)
    np.testing.assert_allclose(seen["outputs"]["geom_coll_primary"].as_array(), 2.0)
    np.testing.assert_array_equal(seen["image"].as_array(), [[[0.0, 2.0]]])
    source = next(call[1] for call in adaptor.calls if call[0] == "set_source")
    np.testing.assert_array_equal(source.as_array(), [[[0.0, 2.0]]])
    assert "set_activity" not in adaptor.names()


def test_route_one_sets_the_activity_before_running_and_does_not_scale():
    adaptor = _adaptor()
    projector = SimindProjector(
        adaptor,
        "template",
        "mu",
        activity_mbq=lambda x: 10.0 * x.sum(),
        time_per_projection_s=20.0,
    )
    outputs = projector.project(FakeData([[[1.0, 2.0]]]))

    assert adaptor.calls.index(("set_activity", 30.0, 20.0)) < adaptor.calls.index(
        ("run",)
    )
    assert projector.last_scale == 1.0
    np.testing.assert_allclose(outputs["geom_coll_primary"].as_array(), 2.0)


def test_template_and_mu_map_are_set_once_and_the_source_is_masked():
    adaptor = _adaptor()
    projector = SimindProjector(
        adaptor, "template", "mu", normalise=_unit, mask=FakeData([[[0.0, 1.0]]])
    )
    projector.project(FakeData([[[4.0, 5.0]]]))
    projector.project(FakeData([[[6.0, 7.0]]]))

    names = adaptor.names()
    assert names.count("set_template") == 1 and names.count("set_mu_map") == 1
    assert names.index("set_template") < names.index("set_source")
    sources = [call[1] for call in adaptor.calls if call[0] == "set_source"]
    np.testing.assert_array_equal(sources[1].as_array(), [[[0.0, 7.0]]])


def test_seed_changes_on_every_call():
    adaptor = _adaptor()
    projector = SimindProjector(adaptor, "template", "mu", normalise=_unit, seed=100)
    for _ in range(3):
        projector.project(FakeData(np.ones((1, 1, 2))))
    seeds = [
        call[2] for call in adaptor.calls if call[:2] == ("add_runtime_switch", "RR")
    ]
    assert seeds == [100, 101, 102]


def test_no_seed_means_no_rr_switch():
    adaptor = _adaptor()
    SimindProjector(adaptor, "template", "mu", normalise=_unit).project(
        FakeData(np.ones((1, 1, 2)))
    )
    assert "add_runtime_switch" not in adaptor.names()


def test_reference_normaliser_passes_the_component_and_the_fast_projection(
    monkeypatch,
):
    captured = {}

    def fake_scale(component, reference, **kwargs):
        captured.update(component=component, reference=reference, kwargs=kwargs)
        return 2.5

    monkeypatch.setattr(projector_mod, "scale_to_reference", fake_scale)

    class _FastModel:
        def direct(self, image):
            captured["image"] = image
            return FakeData(image.as_array() * 4.0)

    normalise = reference_normaliser(
        _FastModel(), method="trimmed", mask="mask", trim_fraction=0.2
    )
    image = FakeData(np.ones((1, 2, 2)))
    outputs = {
        "geom_coll_primary": FakeData(np.full((1, 2, 2), 7.0)),
        "all_interactions": FakeData(np.zeros((1, 2, 2))),
    }

    assert normalise(outputs, image) == 2.5
    assert captured["image"] is image
    np.testing.assert_array_equal(captured["component"], np.full((1, 2, 2), 7.0))
    np.testing.assert_array_equal(captured["reference"], np.full((1, 2, 2), 4.0))
    assert captured["kwargs"] == {
        "method": "trimmed",
        "mask": "mask",
        "trim_fraction": 0.2,
    }


def test_reference_normaliser_defaults_to_the_ratio_of_sums():
    class _FastModel:
        def direct(self, image):
            return FakeData(np.full((1, 2, 2), 6.0))

    normalise = reference_normaliser(_FastModel())
    outputs = {"geom_coll_primary": FakeData(np.full((1, 2, 2), 2.0))}
    assert normalise(outputs, FakeData(np.ones((1, 2, 2)))) == pytest.approx(3.0)
