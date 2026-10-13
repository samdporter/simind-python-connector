"""
Grab-bag of general utilities shared across modules.
They are re-exported here for a single, easy import path.
"""

import contextlib
import importlib
import warnings


# Lazy imports keep SIRF/STIR out of the import path
def __getattr__(name):
    if name in ("interfile", "simind_utils"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def _import_stir_extra():
    """Return the STIR array-conversion module.

    Modern STIR Python exposes the conversion helpers as ``stir.extra``. Older
    installs ship them as the standalone ``stirextra`` module, which is used
    only when ``stir.extra`` is genuinely absent. A missing dependency inside
    ``stir.extra`` is allowed to propagate rather than being masked by the
    legacy fallback.
    """
    try:
        import stir.extra
    except ModuleNotFoundError as exc:
        if exc.name not in (None, "stir", "stir.extra"):
            raise
    else:
        return stir.extra

    with warnings.catch_warnings():
        # The legacy replacement shim warns on import; suppress only that.
        warnings.simplefilter("ignore", DeprecationWarning)
        import stirextra

    return stirextra


def get_array(obj):
    """Return a NumPy array from a SIRF or STIR object.

    - SIRF objects: ``.asarray()`` or ``.as_array()``
    - STIR objects: ``stir.extra.to_numpy()`` (legacy ``stirextra`` fallback)

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
        return _import_stir_extra().to_numpy(obj)
    raise AttributeError(
        f"Cannot convert {type(obj)} to numpy array. "
        f"Object must have asarray(), as_array() method, or be a STIR object."
    )


__all__ = ["get_array", "interfile", "simind_utils"]
