"""Parsing and weighting helpers for the Y-90 spectrum tutorial."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np


WINDOW_LOW_KEV = 75.0
WINDOW_HIGH_KEV = 225.0
CHANNEL_KEV = 4.0


@dataclass
class IsotopeSpectrum:
    energies_kev: np.ndarray
    yields: np.ndarray
    continuous: bool
    continuous_scale: float | None
    bremsstrahlung: str | None
    half_life: float | None

    @property
    def kind(self):
        return "continuum" if self.continuous else "discrete"


def read_isotope_spectrum(path):
    """Read a SIMIND ``.isd`` spectrum."""
    energies, yields, header = [], [], {}
    for number, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            header[key.strip().lower()] = value.strip()
        elif ":" in line and not line[0].isdigit() and line[0] not in "+-.":
            key, value = line.split(":", 1)
            header[key.strip().lower()] = value.strip()
        else:
            fields = line.split()
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected two columns")
            energies.append(float(fields[0]))
            yields.append(float(fields[1]))
    energies = np.asarray(energies, float)
    yields = np.asarray(yields, float)
    if not energies.size:
        raise ValueError(f"{path}: no spectrum points found")
    if not (np.isfinite(energies).all() and np.isfinite(yields).all()):
        raise ValueError(f"{path}: non-finite spectrum values")
    if (energies < 0).any() or (yields < 0).any() or np.any(np.diff(energies) <= 0):
        raise ValueError(f"{path}: invalid spectrum ordering or range")
    scale = header.get("continuous")
    if scale is not None:
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{path}: CONTINUOUS must be positive")
    return IsotopeSpectrum(
        energies_kev=energies,
        yields=yields,
        continuous="continuous" in header,
        continuous_scale=scale,
        bremsstrahlung=header.get("bremsstrahlung"),
        half_life=float(header["halflife"]) if "halflife" in header else None,
    )


def find_isotope_file(name):
    """Locate an isotope file in SIMIND's data directories."""
    roots = [
        Path(value).expanduser()
        for key in ("SIMIND_SMC_DIR", "SMC_DIR")
        if (value := os.environ.get(key))
    ]
    if root := os.environ.get("SIMIND_ROOT"):
        roots.append(Path(root) / "smc_dir")
    if binary := shutil.which(os.environ.get("SIMIND_BIN") or "simind"):
        roots.append(Path(binary).resolve().parent / "smc_dir")
    for directory in roots:
        if (directory / name).is_file():
            return directory / name
    raise FileNotFoundError(f"Could not find {name} in SIMIND's data directories.")


LIST_MODE_DTYPE = np.dtype(
    [("x", "<i2"), ("y", "<i2"), ("z", "<i2"), ("energy", "<i2"), ("weight", "<f8")]
)


@dataclass
class ListModeEvents:
    energies_kev: np.ndarray
    weights: np.ndarray
    projections: np.ndarray
    crystal_cm: np.ndarray


def read_list_mode(path):
    """Read a SIMIND ``/OU:11`` list-mode file."""
    path = Path(path)
    if path.stat().st_size == 0 or path.stat().st_size % LIST_MODE_DTYPE.itemsize:
        raise ValueError(f"{path}: not a whole number of list-mode records")
    records = np.fromfile(path, dtype=LIST_MODE_DTYPE)
    weight = records["weight"].astype(float)
    projections = np.cumsum(weight < 0)
    return ListModeEvents(
        energies_kev=records["energy"].astype(float) / 10.0,
        weights=np.abs(weight),
        projections=projections - projections[0],
        crystal_cm=np.column_stack(
            [records[name].astype(float) for name in ("x", "y", "z")]
        )
        / 100.0,
    )


def histogram_spectrum(energies_kev, weights, bin_width_kev=CHANNEL_KEV):
    edges = np.arange(0.0, 2048.0 + bin_width_kev, bin_width_kev)
    counts, _ = np.histogram(energies_kev, bins=edges, weights=weights)
    return counts, edges


def window_weight(energies_kev, weights, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    inside = (energies_kev >= lower) & (energies_kev <= upper)
    return float(np.sum(weights[inside]))


def window_channel_weights(weights, edges, lower=WINDOW_LOW_KEV, upper=WINDOW_HIGH_KEV):
    """Clip channel weights, splitting channels that cross a window edge."""
    weights = np.asarray(weights, float)
    edges = np.asarray(edges, float)
    if weights.shape != (edges.size - 1,) or np.any(weights < 0):
        raise ValueError("invalid spectrum weights")
    width = np.diff(edges)
    if np.any(width <= 0):
        raise ValueError("bin edges must be strictly increasing")
    low = np.maximum(edges[:-1], lower)
    high = np.minimum(edges[1:], upper)
    fraction = np.clip(high - low, 0.0, None) / width
    keep = fraction > 0
    return 0.5 * (low[keep] + high[keep]), weights[keep] * fraction[keep]


def weighted_mean_energy(energies_kev, weights):
    if np.sum(weights) <= 0:
        raise ValueError("cannot average energies with zero total weight")
    return float(np.average(energies_kev, weights=weights))
