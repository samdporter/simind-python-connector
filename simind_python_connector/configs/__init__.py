"""
Packaged scanner presets (YAML or SMC).

Example
-------
>>> from simind_python_connector.configs import get
>>> cfg = get("AnyScan.yaml")          # Traversable

>>> from simind_python_connector.core import SimulationConfig
>>> sim_cfg = SimulationConfig(cfg)    # load file
"""

from importlib import resources as _res
from typing import Union


try:  # Python 3.11 moved Traversable into importlib.resources.abc
    from importlib.resources.abc import Traversable
except ImportError:  # pragma: no cover - Python 3.10
    from importlib.abc import Traversable


__all__ = ["get", "list"]


def get(name: str) -> Traversable:
    """
    Return the bundled config file as an importlib.resources Traversable (a Path for
    normal installs).

    Parameters
    ----------
    name : str
        Filename, e.g. ``"AnyScan.yaml"`` or ``"input.smc"``.
    """
    return _res.files(__package__).joinpath(name)


def list(ext: Union[str, tuple[str, ...]] = (".yaml", ".smc")) -> list[str]:
    """
    List files in this package (default: *.yaml / *.smc).
    """
    return [
        p.name
        for p in _res.files(__package__).iterdir()
        if p.suffix in (ext if isinstance(ext, tuple) else (ext,))
    ]
