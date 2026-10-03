"""
Grab-bag of general utilities shared across modules.
They are re-exported here for a single, easy import path.
"""

import contextlib
import importlib


# Lazy imports keep SIRF/STIR out of the import path
def __getattr__(name):
    if name in ("interfile", "simind_utils"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def get_array(obj):
    """Return a NumPy array from a SIRF or STIR object.

    - SIRF objects: ``.asarray()`` or ``.as_array()``
    - STIR objects: ``stirextra.to_numpy()``

    Raises:
        AttributeError: If the object cannot be converted.
    """
    if hasattr(obj, "as_array") and callable(obj.as_array):
        return obj.as_array()

    if hasattr(obj, "asarray"):
        try:
            return obj.asarray()
        except Exception:
            return obj.as_array()

    with contextlib.suppress(ImportError, TypeError, AttributeError):
        import stirextra

        return stirextra.to_numpy(obj)
    raise AttributeError(
        f"Cannot convert {type(obj)} to numpy array. "
        f"Object must have asarray(), as_array() method, or be a STIR object."
    )


__all__ = ["get_array", "interfile", "simind_utils"]
