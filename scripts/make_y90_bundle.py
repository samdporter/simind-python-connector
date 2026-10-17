#!/usr/bin/env python
"""Build the Zenodo archive of the Y-90 phantom data.

Usage: python scripts/make_y90_bundle.py SOURCE_DIR OUT_DIR

Writes OUT_DIR/y90_phantom.zip (de-identified DICOM copies, the other phantom
files and a README) and OUT_DIR/zenodo_metadata.json (a draft to complete
before upload). The source files are never modified. Prints the archive sha256.
"""

import argparse
import hashlib
import io
import json
import pathlib
import zipfile

import pydicom

from simind_python_connector import datasets


IDENTIFYING_TAGS = (
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "StationName",
    "OperatorsName",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "PhysiciansOfRecord",
    "DeviceSerialNumber",
)

README_INTRO = """# Y-90 phantom data

Real SPECT/CT acquisition of a liver phantom with a hot sphere, used by the
tutorials of simind-python-connector. The DICOM files are de-identified: the
institution, station, operator, physician and device serial number tags are
blank. The patient name is the phantom name.

Files: `peak.dcm` (projections), `peak.hs` and `peak.s` (Interfile projections),
`umap_zoomed.hv` and `umap_zoomed.v` (attenuation map), `ctac/` (CT slices).

"""

KEYWORDS = ["SPECT", "Y-90", "phantom", "SIMIND", "radioembolization"]


def _deidentified(path, counts):
    ds = pydicom.dcmread(path)
    for keyword in IDENTIFYING_TAGS:
        if keyword in ds:
            ds[keyword].value = ""
            counts[keyword] += 1
    buffer = io.BytesIO()
    ds.save_as(buffer)
    return buffer.getvalue()


def _find_peak_dicom(source):
    for candidate in (source / "peak.dcm", source / "spect_dicom" / "peak.dcm"):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"peak.dcm not found in {source} or {source}/spect_dicom")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source_dir", type=pathlib.Path)
    parser.add_argument("out_dir", type=pathlib.Path)
    args = parser.parse_args(argv)

    counts = dict.fromkeys(IDENTIFYING_TAGS, 0)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    archive = args.out_dir / "y90_phantom.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("README.md", README_INTRO + datasets.__doc__)
        bundle.writestr(
            "peak.dcm", _deidentified(_find_peak_dicom(args.source_dir), counts)
        )
        for name in datasets.Y90_PHANTOM_FILES:
            if name not in ("peak.dcm", "ctac"):
                bundle.write(args.source_dir / name, name)
        for path in sorted((args.source_dir / "ctac").glob("*.dcm")):
            bundle.writestr(f"ctac/{path.name}", _deidentified(path, counts))

    metadata = {
        "title": "Y-90 SPECT/CT liver phantom with a hot sphere",
        "upload_type": "dataset",
        "description": datasets.__doc__.splitlines()[0]
        + " Real acquisition data for the simind-python-connector tutorials.",
        "keywords": KEYWORDS,
        "license": "cc-by-4.0",
        "creators": [{"name": "Porter, Sam"}],
    }
    (args.out_dir / "zenodo_metadata.json").write_text(json.dumps(metadata, indent=2))

    print("De-identified tags (files changed per tag):")
    for keyword, count in counts.items():
        print(f"  {keyword}: {count}")
    print(f"sha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
