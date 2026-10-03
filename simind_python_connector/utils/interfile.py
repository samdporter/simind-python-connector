"""Interfile header parsing and editing, and loading Interfile data into NumPy."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np


PathLike = Union[str, os.PathLike]

_KEY_PREFIX = re.compile(r"^[!#\s]+")
_WHITESPACE = re.compile(r"\s+")
_BRACKET = re.compile(r"\s*\[")
_MATRIX_SIZE_KEY = re.compile(r"matrix size \[(\d+)\]")


def normalise_key(key: str) -> str:
    """Canonical form of an Interfile key, used for every lookup.

    Leading '!' and '#' markers, letter case and spacing differ between
    SIMIND, STIR and our own builders, so they are ignored.
    """
    key = _KEY_PREFIX.sub("", key.strip())
    key = _WHITESPACE.sub(" ", key)
    key = _BRACKET.sub(" [", key)
    return key.lower()


def parse_interfile_line(line: str) -> tuple[Optional[str], Optional[str]]:
    """Return (key, value) for a 'key := value' line, else (None, None).

    Comments (';'), blank lines and section headers ('key :=') are skipped.
    Keys are returned as written; use normalise_key() to compare them.
    """
    line = line.strip()
    if not line or line.startswith(";") or line.endswith(":="):
        return None, None
    if ":=" not in line:
        return None, None
    key, _, value = line.partition(":=")
    return key.strip(), value.strip()


def _format_value(value) -> str:
    if isinstance(value, (tuple, list)):
        return "{" + ", ".join(str(v) for v in value) + "}"
    return str(value)


@dataclass
class InterfileEntry:
    """One line of a header; key and value are None for non-key lines."""

    text: str
    key: Optional[str]
    value: Optional[str]

    @classmethod
    def from_line(cls, line: str) -> "InterfileEntry":
        key, value = parse_interfile_line(line)
        if not line.endswith("\n"):
            line = line + "\n"
        return cls(text=line, key=key, value=value)

    @classmethod
    def from_key_value(cls, key: str, value) -> "InterfileEntry":
        value_str = _format_value(value)
        return cls(text=f"{key} := {value_str}\n", key=key, value=value_str)

    def set_value(self, value) -> None:
        self.value = _format_value(value)
        self.text = f"{self.key} := {self.value}\n"


class InterfileHeader:
    """Editable Interfile header that keeps the original lines when written."""

    def __init__(self, entries: list[InterfileEntry]):
        self._entries = entries

    @classmethod
    def from_file(cls, path: PathLike) -> "InterfileHeader":
        with open(path, "r") as file:
            return cls([InterfileEntry.from_line(line) for line in file])

    @classmethod
    def from_text(cls, text: str) -> "InterfileHeader":
        lines = text.splitlines(keepends=True)
        return cls([InterfileEntry.from_line(line) for line in lines])

    def copy(self) -> "InterfileHeader":
        return InterfileHeader(
            [InterfileEntry(e.text, e.key, e.value) for e in self._entries]
        )

    def _find_last(self, key: str) -> Optional[InterfileEntry]:
        wanted = normalise_key(key)
        for entry in reversed(self._entries):
            if entry.key is not None and normalise_key(entry.key) == wanted:
                return entry
        return None

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        entry = self._find_last(key)
        return entry.value if entry is not None else default

    def set(self, key: str, value) -> None:
        entry = self._find_last(key)
        if entry is None:
            self._entries.append(InterfileEntry.from_key_value(key, value))
        else:
            entry.set_value(value)

    def insert(self, index: int, key: str, value) -> None:
        index = max(0, min(index, len(self._entries)))
        self._entries.insert(index, InterfileEntry.from_key_value(key, value))

    def items(self) -> list[tuple[str, str]]:
        return [(e.key, e.value) for e in self._entries if e.key is not None]

    def as_dict(self) -> dict[str, str]:
        return {normalise_key(key): value for key, value in self.items()}

    def write(self, path: PathLike) -> None:
        with open(path, "w") as file:
            for entry in self._entries:
                file.write(entry.text)


def read_header(source) -> InterfileHeader:
    """Read a header from a path, or from an object that can write itself."""
    if isinstance(source, (str, os.PathLike)):
        return InterfileHeader.from_file(source)
    writer = getattr(source, "write", None) or getattr(source, "write_to_file", None)
    if writer is None:
        raise TypeError("source must be a path or an object with a write(path) method")
    with tempfile.TemporaryDirectory() as tmp_dir:
        header_path = Path(tmp_dir) / "header.hs"
        writer(str(header_path))
        return InterfileHeader.from_file(header_path)


@dataclass(frozen=True)
class InterfileArray:
    """NumPy payload and file references parsed from an Interfile header."""

    array: np.ndarray
    header_path: Path
    data_path: Path
    metadata: dict[str, str]


def _matrix_shape(values: dict[str, str]) -> tuple[int, ...]:
    sizes: dict[int, int] = {}
    for key, raw in values.items():
        match = _MATRIX_SIZE_KEY.fullmatch(key)
        if match:
            sizes[int(match.group(1))] = int(raw)
    if not sizes:
        raise ValueError("No 'matrix size [i]' entries found in Interfile header")
    ordered = [sizes[index] for index in sorted(sizes)]
    if any(size <= 0 for size in ordered):
        raise ValueError(f"Invalid matrix sizes in header: {ordered}")
    # Interfile axis [1] changes fastest, so it is the last NumPy axis.
    return tuple(reversed(ordered))


def _number_dtype(values: dict[str, str]) -> np.dtype:
    number_format = values.get("number format", "float").strip()
    bytes_raw = values.get("number of bytes per pixel")
    bytes_per_pixel = int(float(bytes_raw)) if bytes_raw else 4
    lowered = number_format.lower()
    if "float" in lowered:
        kind = "f"
    elif "unsigned" in lowered:
        kind = "u"
    elif "signed" in lowered or "integer" in lowered:
        kind = "i"
    else:
        raise ValueError(f"Unsupported Interfile number format: {number_format!r}")
    dtype = np.dtype(f"{kind}{bytes_per_pixel}")
    byte_order = values.get("imagedata byte order", "").lower()
    if "big" in byte_order:
        return dtype.newbyteorder(">")
    if "little" in byte_order:
        return dtype.newbyteorder("<")
    return dtype.newbyteorder("=")


def _projection_count(values: dict[str, str]) -> Optional[int]:
    for key in (
        "number of projections",
        "total number of images",
        "number of images/energy window",
    ):
        raw = values.get(key)
        if raw is None:
            continue
        try:
            count = int(float(raw))
        except ValueError:
            return None
        return count if count > 0 else None
    return None


def _leading_axis_count(
    values: dict[str, str], plane_elements: int, payload_elements: int
) -> Optional[int]:
    if plane_elements <= 0:
        return None
    count = _projection_count(values)
    if count is not None:
        return count
    if payload_elements % plane_elements == 0:
        inferred = payload_elements // plane_elements
        if inferred > 0:
            return inferred
    return 1


def load_interfile_array(header_path: PathLike) -> InterfileArray:
    """Load the data referenced by an Interfile header into NumPy."""
    header_path = Path(header_path).expanduser().resolve()
    header = InterfileHeader.from_file(header_path)
    values = header.as_dict()

    data_filename = values.get("name of data file")
    if not data_filename:
        raise ValueError(
            f"Interfile header {header_path} does not define 'name of data file'"
        )
    data_path = (header_path.parent / data_filename.strip().strip("'\"")).resolve()
    if not data_path.exists():
        raise FileNotFoundError(
            f"Interfile data file referenced by header does not exist: {data_path}"
        )

    dtype = _number_dtype(values)
    shape = _matrix_shape(values)
    expected_elements = int(np.prod(shape))

    offset_raw = values.get("data offset in bytes", values.get("data_offset_in_bytes"))
    offset = int(float(offset_raw)) if offset_raw else 0
    if offset < 0:
        raise ValueError(f"Negative data offset in Interfile header: {offset}")
    if offset % dtype.itemsize != 0:
        raise ValueError(
            f"Data offset {offset} is not aligned to a {dtype.itemsize}-byte "
            "pixel boundary"
        )

    flat = np.fromfile(data_path, dtype=dtype, offset=offset)

    # Projection headers often declare only [1] and [2]; recover the
    # projection axis from the count keys or from the payload length.
    if len(shape) == 2:
        leading = _leading_axis_count(values, expected_elements, flat.size)
        if leading is not None:
            shape = (leading, *shape)
            expected_elements = int(np.prod(shape))

    if flat.size != expected_elements:
        raise ValueError(
            f"Data size mismatch for {data_path}: expected {expected_elements} "
            f"elements for shape {shape}, found {flat.size}"
        )

    return InterfileArray(
        array=flat.reshape(shape),
        header_path=header_path,
        data_path=data_path,
        metadata=dict(header.items()),
    )


@dataclass(frozen=True)
class ProjectionGeometry:
    """SPECT acquisition geometry read from a projection header."""

    num_projections: int
    extent_deg: float
    direction: str
    start_angle_deg: float
    radius_mm: Optional[float]
    radii_mm: Optional[tuple[float, ...]]
    num_bins: int
    num_axial: int
    bin_size_mm: float
    axial_size_mm: float
    image_duration_s: Optional[float]


_REQUIRED_GEOMETRY_KEYS = (
    "number of projections",
    "extent of rotation",
    "direction of rotation",
    "start angle",
    "matrix size [1]",
    "matrix size [2]",
    "scaling factor (mm/pixel) [1]",
    "scaling factor (mm/pixel) [2]",
)


def read_projection_geometry(header: InterfileHeader) -> ProjectionGeometry:
    """Extract the acquisition geometry from a STIR/SIRF projection header."""
    values = header.as_dict()
    missing = [key for key in _REQUIRED_GEOMETRY_KEYS if key not in values]
    if "radii" not in values and "radius" not in values:
        missing.append("radius or radii")
    if missing:
        raise ValueError(
            "Interfile header is missing projection geometry keys: "
            + ", ".join(missing)
        )

    direction = values["direction of rotation"].strip().upper()
    if direction not in ("CW", "CCW"):
        raise ValueError(
            f"Unsupported direction of rotation {values['direction of rotation']!r}"
        )

    radius_mm: Optional[float] = None
    radii_mm: Optional[tuple[float, ...]] = None
    if "radii" in values:
        radii = tuple(float(v) for v in values["radii"].strip("{} ").split(","))
        mean = float(np.mean(radii))
        if float(np.std(radii)) / mean > 1e-6:
            radii_mm = radii
        else:
            radius_mm = mean
    else:
        radius_mm = float(values["radius"])

    duration = values.get(
        "image duration (sec) [1]", values.get("image duration (sec)")
    )
    return ProjectionGeometry(
        num_projections=int(values["number of projections"]),
        extent_deg=float(values["extent of rotation"]),
        direction=direction,
        start_angle_deg=float(values["start angle"]),
        radius_mm=radius_mm,
        radii_mm=radii_mm,
        num_bins=int(values["matrix size [1]"]),
        num_axial=int(values["matrix size [2]"]),
        bin_size_mm=float(values["scaling factor (mm/pixel) [1]"]),
        axial_size_mm=float(values["scaling factor (mm/pixel) [2]"]),
        image_duration_s=float(duration) if duration is not None else None,
    )


def read_simind_density_image(
    header_path: PathLike,
) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Read SIMIND's aligned density image (Flag 15: .hct header, .ict data).

    The data file holds density * 1000 as 16-bit unsigned integers.
    Returns (density in g/cm^3 with shape (z, y, x), voxel sizes (z, y, x) mm).
    """
    header_path = Path(header_path)
    values = InterfileHeader.from_file(header_path).as_dict()
    try:
        shape = tuple(int(values[f"matrix size [{axis}]"]) for axis in (3, 2, 1))
        voxel_sizes = tuple(
            float(values[f"scaling factor (mm/pixel) [{axis}]"]) for axis in (3, 2, 1)
        )
    except KeyError as exc:
        raise ValueError(f"{header_path} is missing {exc.args[0]!r}") from exc

    data_path = header_path.with_suffix(".ict")
    raw = np.fromfile(data_path, dtype="<u2")
    expected = int(np.prod(shape))
    if raw.size != expected:
        raise ValueError(
            f"{data_path} holds {raw.size} values, expected {expected} "
            f"for shape {shape}"
        )
    density = raw.reshape(shape).astype(np.float32) / 1000.0
    return density, voxel_sizes


__all__ = [
    "InterfileArray",
    "InterfileEntry",
    "InterfileHeader",
    "ProjectionGeometry",
    "load_interfile_array",
    "normalise_key",
    "parse_interfile_line",
    "read_header",
    "read_projection_geometry",
    "read_simind_density_image",
]
