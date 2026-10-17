"""Y-90 phantom data for the tutorials.

Dataset facts, quoted by the tutorials:

- Mediso AnyScan, MLEGP collimator, 75-225 keV energy window.
- 120 views x 40 s, counter-clockwise, non-circular orbit.
- 128 x 128 matrix at 4.2578 mm.
- Truth table, with activities taken at the scan:
  - liver: 156.4 MBq, 1149.49 g, 136.1 kBq/g;
  - hot sphere: 4.40 MBq, 8.082 g, 544.4 kBq/g;
  - total: 160.8 MBq.
"""

import os
import pathlib


Y90_PHANTOM_FILES: tuple[str, ...] = (
    "peak.dcm",
    "peak.hs",
    "peak.s",
    "umap_zoomed.hv",
    "umap_zoomed.v",
    "ctac",
)

Y90_PHANTOM_RECORD: dict[str, str] = {"url": "", "sha256": ""}

_ENVIRONMENT_VARIABLE = "SIMIND_CONNECTOR_Y90_DATA"


def fetch_y90_phantom(data_home: str | os.PathLike | None = None) -> pathlib.Path:
    """Return the directory holding the Y-90 phantom files.

    The directory named by ``SIMIND_CONNECTOR_Y90_DATA`` is used as is. Otherwise
    the published archive is downloaded with pooch into ``data_home`` (default:
    the pooch cache) and unpacked there.
    """
    override = os.environ.get(_ENVIRONMENT_VARIABLE)
    if override:
        directory = pathlib.Path(override)
    elif not Y90_PHANTOM_RECORD["url"]:
        raise RuntimeError(
            "The Y-90 phantom dataset is not published yet. Set "
            f"{_ENVIRONMENT_VARIABLE} to a directory holding the phantom files."
        )
    else:
        import pooch

        cache = data_home or pooch.os_cache("simind-python-connector")
        pooch.retrieve(
            Y90_PHANTOM_RECORD["url"],
            known_hash=f"sha256:{Y90_PHANTOM_RECORD['sha256']}",
            path=cache,
            processor=pooch.Unzip(extract_dir="y90_phantom"),
        )
        directory = pathlib.Path(cache) / "y90_phantom"

    missing = [name for name in Y90_PHANTOM_FILES if not (directory / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Y-90 phantom files missing from {directory}: {', '.join(missing)}"
        )
    return directory
