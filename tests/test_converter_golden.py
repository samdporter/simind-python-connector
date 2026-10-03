"""Golden-file tests pinning the SIMIND -> STIR header conversion output."""

import tempfile
from pathlib import Path

import pytest

from simind_python_connector.converters.simind_to_stir import SimindToStirConverter
from simind_python_connector.core.types import PenetrateOutputType


pytestmark = pytest.mark.unit

GOLDEN_DIR = Path(__file__).parent / "fixtures" / "golden"
CASES = ("circular", "orbit", "ignored_substrings", "penetrate_component_12")

_SAMPLE_HEAD = """!INTERFILE :=
!imaging modality := nucmed
!originating system := simind
!version of keys := 3.3
program author := M Ljungberg, Lund University

!GENERAL DATA :=
!data offset in bytes := 0
!name of data file := output.a00
patient name := test_patient

!GENERAL IMAGE DATA :=
!type of data := tomographic
imagedata byte order := LITTLEENDIAN
;energy window lower level := 126.16
;energy window upper level := 36.84
!number of energy windows := 1
!matrix size [1] := 128
!matrix size [2] := 128
!number format := short float
!number of bytes per pixel := 4
scaling factor (mm/pixel) [1] := 4.419600
scaling factor (mm/pixel) [2] := 4.419600

!SPECT STUDY (General) :=
!extent of rotation := 360
!process status := acquired
!number of projections := 60
image duration (sec) := 60.000000

!SPECT STUDY (acquired data) :=
"""

_SAMPLE_TAIL = """!direction of rotation := CW
start angle := 0.000000

!END OF INTERFILE :=
"""

_IGNORED_SUBSTRINGS_H00 = "\n".join(
    [
        "!INTERFILE :=",
        "!name of data file := patient.a00",
        "patient name := John Doe",
        "!study ID := study123",
        "!END OF INTERFILE :=",
    ]
)

_PENETRATE_H00 = "\n".join(
    [
        "!name of data file := output.a00",
        "patient name := phantom",
        "!study ID := study123",
        "data description := placeholder",
        "!END OF INTERFILE :=",
    ]
)


def _sample_h00(orbit: bool) -> str:
    if orbit:
        orbit_lines = "orbit := noncircular\n;# Non-Uniform Orbit File := output.cor\n"
    else:
        orbit_lines = "orbit := circular\n"
    return _SAMPLE_HEAD + orbit_lines + _SAMPLE_TAIL


def render_case(case: str, work_dir: Path) -> str:
    """Write the inputs for case into work_dir, convert them, return the text."""
    converter = SimindToStirConverter()
    h00 = work_dir / "output.h00"
    if case in ("circular", "orbit"):
        h00.write_text(_sample_h00(orbit=case == "orbit"))
        if case == "orbit":
            (work_dir / "output.cor").write_text(
                "".join(
                    f"      {15.0 + (i % 10) * 0.5:6.3f}    64\n" for i in range(60)
                )
            )
        out = work_dir / "output.hs"
        converter.convert_file(str(h00), str(out))
        return out.read_text()
    if case == "ignored_substrings":
        h00.write_text(_IGNORED_SUBSTRINGS_H00)
        out = work_dir / "output.hs"
        converter.convert_file(str(h00), str(out))
        return out.read_text()
    if case == "penetrate_component_12":
        h00.write_text(_PENETRATE_H00)
        component = PenetrateOutputType.COLL_SCATTER_PRIMARY_ATT_BACK
        (work_dir / f"output.b{component.value:02d}").write_bytes(b"\x00" * 4)
        converter.create_penetrate_headers_from_template(
            str(h00), "output", str(work_dir)
        )
        header = work_dir / f"output_component_{component.value:02d}.hs"
        return header.read_text()
    raise ValueError(f"unknown case {case!r}")


def write_goldens() -> None:
    """Regenerate the golden files. Only use this for intended output changes."""
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for case in CASES:
        with tempfile.TemporaryDirectory() as tmp:
            (GOLDEN_DIR / f"{case}.hs").write_text(render_case(case, Path(tmp)))


@pytest.mark.parametrize("case", CASES)
def test_converter_output_matches_golden(case: str, tmp_path: Path) -> None:
    expected = (GOLDEN_DIR / f"{case}.hs").read_text()
    assert render_case(case, tmp_path) == expected
