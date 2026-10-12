from importlib import metadata

import pytest
from packaging.requirements import Requirement

import simind_python_connector


pytestmark = pytest.mark.unit


def test_public_import_package_is_simind_python_connector():
    assert simind_python_connector.__name__ == "simind_python_connector"


def test_distribution_version_matches_import_package_version():
    assert simind_python_connector.__version__ == metadata.version(
        "simind-python-connector"
    )


def test_installed_metadata_caps_numpy_below_2():
    requirements = [
        Requirement(value)
        for value in metadata.requires("simind-python-connector") or []
    ]
    numpy_requirements = [req for req in requirements if req.name == "numpy"]
    assert len(numpy_requirements) == 1, numpy_requirements
    requirement = numpy_requirements[0]
    assert requirement.marker is None, requirement.marker
    assert requirement.specifier.contains("1.26.4")
    assert not requirement.specifier.contains("2.0.0")
