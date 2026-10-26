"""Helpers for the sphinx-gallery configuration and its release gate.

The gallery gate is deliberately static: it reads the committed gallery output,
the per-example figure manifest and the reported run summaries.  It never
executes an example, so the offline documentation build can run the same checks
without SIMIND, STIR, SIRF, PyTomography or the restricted Y-90 data.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import re
import zipfile
from pathlib import Path
from typing import NamedTuple

from sphinx_gallery.utils import get_md5sum


BACKENDS = ("python", "stir", "sirf", "pytomography")
MANIFEST_NAME = "gallery_figures.json"
PROVENANCE_NAME = "auto_examples/provenance.json"

_ADAPTOR_ROOTS = {
    "sirf": "sirf",
    "cil": "sirf",
    "stir": "stir",
    "pytomography": "pytomography",
}
_IMAGE_DIRECTIVE = re.compile(r"^\s*\.\. image-sg::\s+(?P<path>\S+)\s*$")
_ALT_DIRECTIVE = re.compile(r"^\s+:alt:\s*(?P<alt>.*)$")
_IMAGE_SRC = re.compile(r'<img[^>]+src="([^"]+)"')
_MISSING = object()


class GalleryProblem(NamedTuple):
    """A single reason the committed gallery is not release-ready."""

    example: str
    code: str
    detail: str

    def __str__(self) -> str:
        return f"{self.example}: {self.code}: {self.detail}"


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def example_backend(path: Path) -> str:
    roots = _import_roots(path)
    backends = {_ADAPTOR_ROOTS[root] for root in roots & _ADAPTOR_ROOTS.keys()}
    if len(backends) > 1:
        raise ValueError(f"{path.name} imports more than one backend: {sorted(roots)}")
    return backends.pop() if backends else "python"


def gallery_examples(examples_dir: Path) -> list[Path]:
    return sorted(examples_dir.glob("*/plot_*.py"))


def example_key(path: Path) -> str:
    """Return the manifest key for a gallery source file."""

    return f"{path.parent.name}/{path.stem}"


def execution_pattern(examples_dir: Path, backend: str) -> str:
    paths = [
        path
        for path in gallery_examples(examples_dir)
        if example_backend(path) == backend
    ]
    if not paths:
        return r"^$"
    names = "|".join(
        f"{path.parent.name}/{path.name}".replace(".", r"\.") for path in paths
    )
    return f"/examples/({names})$"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_path(docs_dir: Path) -> Path:
    return docs_dir / MANIFEST_NAME


def provenance_path(root: Path) -> Path:
    return root / "docs" / PROVENANCE_NAME


def dependency_files(root: Path) -> list[Path]:
    """Files whose contents change the figures independently of the example.

    The example source itself is tracked by its ``.md5`` sidecar.  These are the
    shared helpers, the documentation settings and the packaged presets/data
    that examples rely on implicitly.
    """

    package = root / "simind_python_connector"
    files = [
        root / "docs" / "conf.py",
        root / "examples" / "_python_connector_helpers.py",
    ]
    if package.exists():
        files.extend(
            path for path in package.rglob("*.py") if "__pycache__" not in path.parts
        )
        for name in ("configs", "data"):
            directory = package / name
            if directory.exists():
                files.extend(
                    path
                    for path in directory.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                )
    return sorted({path for path in files if path.is_file()})


def dependency_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in dependency_files(root):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_manifest(docs_dir: Path) -> dict:
    path = _manifest_path(docs_dir)
    if not path.exists():
        return {"examples": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def load_provenance(root: Path) -> dict:
    path = provenance_path(root)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("examples", {})


def _gallery_base(gallery_dir: Path, key: str) -> tuple[Path, str, str]:
    section, stem = key.split("/", 1)
    return gallery_dir / section, section, stem


def _saved_figures(base: Path, stem: str) -> list[Path]:
    return sorted((base / "images").glob(f"sphx_glr_{stem}_*.png"))


def _thumbnail(base: Path, stem: str) -> Path:
    return base / "images" / "thumb" / f"sphx_glr_{stem}_thumb.png"


def record_provenance(
    root: Path,
    docs_dir: Path | None = None,
    backends: tuple[str, ...] = BACKENDS,
    *,
    build_failed: bool = False,
    error: str | None = None,
) -> Path:
    """Merge the current gallery state into the committed provenance sidecar.

    Called by the Sphinx ``build-finished`` hook only when a backend build was
    requested.  A failed build marks its examples as failed instead of silently
    leaving the old, complete-looking provenance in place.
    """

    docs_dir = docs_dir or root / "docs"
    examples_dir = root / "examples"
    gallery_dir = docs_dir / "auto_examples"
    path = provenance_path(root)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    else:
        data = {"version": 1}
    entries = data.setdefault("examples", {})
    digest = dependency_digest(root)

    for example in gallery_examples(examples_dir):
        if example_backend(example) not in backends:
            continue
        key = example_key(example)
        base, _, stem = _gallery_base(gallery_dir, key)
        entry = entries.get(key, {})
        entry.update(
            {
                "source_md5": get_md5sum(example, mode="t"),
                "dependencies_sha256": digest,
                "build_failed": bool(build_failed),
            }
        )
        if error:
            entry["error"] = str(error)
        else:
            entry.pop("error", None)
        if not build_failed:
            rst = base / f"{stem}.rst"
            if rst.exists():
                entry["rst_sha256"] = _sha256(rst)
            entry["figures"] = {
                figure.name: _sha256(figure) for figure in _saved_figures(base, stem)
            }
            thumbnail = _thumbnail(base, stem)
            if thumbnail.exists():
                entry["thumbnail_sha256"] = _sha256(thumbnail)
            entry["downloads"] = {
                download.name: _sha256(download)
                for suffix in (".py", ".ipynb", ".zip")
                if (download := base / f"{stem}{suffix}").exists()
            }
        entries[key] = entry

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


def record_gallery_build(
    root: Path,
    docs_dir: Path,
    backend: str,
    exception: BaseException | None,
) -> None:
    """Record a successful backend gallery build, or mark it failed."""

    record_provenance(
        root,
        docs_dir,
        (backend,),
        build_failed=exception is not None,
        error=None if exception is None else f"{type(exception).__name__}: {exception}",
    )


def embedded_figures(rst_path: Path) -> list[dict[str, str]]:
    """Return the ``.. image-sg::`` entries embedded in a gallery page."""

    entries = []
    lines = rst_path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        match = _IMAGE_DIRECTIVE.match(line)
        if not match:
            continue
        alt = ""
        for following in lines[index + 1 : index + 10]:
            alt_match = _ALT_DIRECTIVE.match(following)
            if alt_match:
                alt = alt_match.group("alt").strip()
                break
            if following and not following.startswith(" "):
                break
        entries.append({"path": match.group("path"), "alt": alt})
    return entries


def _resolve_image(path_text: str, docs_dir: Path, gallery_dir: Path) -> Path:
    if path_text.startswith("/"):
        return docs_dir / path_text.lstrip("/")
    return gallery_dir / path_text


def _image_problem(path: Path) -> str | None:
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
    except Exception as error:  # pragma: no cover - message is asserted in tests
        return f"{type(error).__name__}: {error}"
    return None


def _matches_default_thumbnail(path: Path, candidate: Path) -> bool:
    try:
        from PIL import Image, ImageChops

        if not candidate.exists():
            return False
        with Image.open(path) as image_source, Image.open(candidate) as default_source:
            image = image_source.convert("RGB")
            reference = default_source.convert("RGB")
            if image.size != reference.size:
                reference = reference.resize(image.size)
            difference = ImageChops.difference(image, reference)
            total = sum(sum(pixel) for pixel in difference.getdata())
            mean = total / (image.size[0] * image.size[1] * 3)
    except Exception:
        return False
    return mean < 2.0


def default_thumbnail_kind(path: Path) -> str | None:
    """Name the sphinx-gallery fallback image, if this is one."""

    try:
        from sphinx_gallery.gen_gallery import glr_path_static

        static = Path(glr_path_static())
    except Exception:
        return None
    for name, label in (
        ("no_image.png", "default gallery flower"),
        ("broken_example.png", "broken-example placeholder"),
    ):
        if _matches_default_thumbnail(path, static / name):
            return label
    return None


def is_placeholder_thumbnail(path: Path) -> bool:
    """Return true when the thumbnail is a sphinx-gallery fallback."""

    return default_thumbnail_kind(path) is not None


def _matches_spec(spec: dict, name: str, alt: str | None) -> bool:
    accepts = spec.get("accepts", [])
    if accepts and not any(fnmatch.fnmatch(name, pattern) for pattern in accepts):
        return False
    tokens = spec.get("alt_contains")
    if tokens and alt is not None:
        if not any(token.lower() in alt.lower() for token in tokens):
            return False
    return True


def _config_roles(entry: dict) -> list[dict]:
    return list(entry.get("figures", []))


def _figure_count_problem(entry: dict, count: int) -> str | None:
    expected = entry.get("figure_count")
    if expected is None:
        roles = _config_roles(entry)
        if not roles:
            return None
        minimum = sum(0 if role.get("optional") else 1 for role in roles)
        maximum = len(roles)
    elif isinstance(expected, int):
        minimum = maximum = expected
    else:
        minimum, maximum = expected
    if not minimum <= count <= maximum:
        return f"expected {minimum}-{maximum} embedded figures, found {count}"
    return None


def _flatten_problem(example: str, code: str, detail: str) -> GalleryProblem:
    return GalleryProblem(example, code, detail)


def _get_field(payload: dict, dotted: str):
    value = payload
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return _MISSING if value is None else value


def _check_downloads(
    example: Path, base: Path, stem: str, key: str
) -> list[GalleryProblem]:
    problems = []
    source = example.read_bytes()
    python = base / f"{stem}.py"
    if not python.exists():
        problems.append(_flatten_problem(key, "download_missing", python.name))
    elif python.read_bytes() != source:
        problems.append(_flatten_problem(key, "download_source_mismatch", python.name))

    notebook = base / f"{stem}.ipynb"
    if not notebook.exists():
        problems.append(_flatten_problem(key, "download_missing", notebook.name))
    else:
        try:
            json.loads(notebook.read_text(encoding="utf-8"))
        except Exception as error:
            problems.append(
                _flatten_problem(key, "download_corrupt", f"{notebook.name}: {error}")
            )

    archive_path = base / f"{stem}.zip"
    if not archive_path.exists():
        problems.append(_flatten_problem(key, "download_missing", archive_path.name))
    else:
        try:
            with zipfile.ZipFile(archive_path) as archive:
                names = archive.namelist()
                if f"{stem}.py" not in names:
                    problems.append(
                        _flatten_problem(
                            key,
                            "download_corrupt",
                            f"{archive_path.name} has no source",
                        )
                    )
                elif archive.read(f"{stem}.py") != source:
                    problems.append(
                        _flatten_problem(
                            key,
                            "download_source_mismatch",
                            f"{archive_path.name}:{stem}.py",
                        )
                    )
        except zipfile.BadZipFile as error:
            problems.append(
                _flatten_problem(
                    key, "download_corrupt", f"{archive_path.name}: {error}"
                )
            )
    return problems


def _check_run_summary(
    entry: dict, docs_dir: Path, key: str, page_text: str
) -> list[GalleryProblem]:
    block = entry.get("run_summary")
    if not block:
        return []
    problems = []
    summary_path = docs_dir / block["path"]
    if not summary_path.exists():
        return [
            _flatten_problem(
                key,
                "missing_run_summary",
                summary_path.relative_to(docs_dir).as_posix(),
            )
        ]
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception as error:
        return [_flatten_problem(key, "corrupt_run_summary", str(error))]

    for field in block.get("required_fields", []):
        if _get_field(summary, field) is _MISSING:
            problems.append(_flatten_problem(key, "missing_summary_field", field))

    claims = block.get("statistical_claims")
    if claims:
        gates = summary.get("gates", {})
        required = claims.get("required_gates", [])
        failed = [gate for gate in required if gates.get(gate) is not True]
        if claims.get("quantitative"):
            if summary.get("status") != "accepted":
                problems.append(
                    _flatten_problem(
                        key,
                        "quantitative_claim_not_accepted",
                        f"status is {summary.get('status')!r}",
                    )
                )
            if failed:
                problems.append(
                    _flatten_problem(key, "statistical_gate_failed", ", ".join(failed))
                )
        if claims.get("negative_result"):
            for marker in claims.get("page_markers", []):
                if marker.lower() not in page_text.lower():
                    problems.append(
                        _flatten_problem(key, "negative_result_unmarked", marker)
                    )
            for marker in claims.get("forbidden_page_markers", []):
                if marker.lower() in page_text.lower():
                    problems.append(
                        _flatten_problem(key, "negative_result_mislabelled", marker)
                    )
            if summary.get("status") == "accepted":
                problems.append(
                    _flatten_problem(
                        key,
                        "negative_result_mislabelled",
                        "summary claims acceptance on a negative-result page",
                    )
                )
            if not failed:
                problems.append(
                    _flatten_problem(
                        key,
                        "negative_result_without_failure",
                        "no required gate failed",
                    )
                )
    return problems


def _broken_html_images(page: Path, label: str) -> list[GalleryProblem]:
    problems = []
    text = page.read_text(encoding="utf-8", errors="replace")
    for source in _IMAGE_SRC.findall(text):
        if source.startswith(("http:", "https:", "data:")):
            continue
        target = (page.parent / source.split("?")[0].split("#")[0]).resolve()
        if not target.exists():
            problems.append(_flatten_problem(label, "broken_html_image", source))
    return problems


def _check_html(
    html_dir: Path, examples: list[Path], entries: dict
) -> list[GalleryProblem]:
    problems = []
    gallery_root = html_dir / "auto_examples"
    for example in examples:
        key = example_key(example)
        _, section, stem = _gallery_base(gallery_root, key)
        page = gallery_root / section / f"{stem}.html"
        if not page.exists():
            problems.append(_flatten_problem(key, "html_page_missing", page.name))
            continue
        problems.extend(_broken_html_images(page, key))
        sources = _IMAGE_SRC.findall(page.read_text(encoding="utf-8", errors="replace"))
        entry = entries.get(key, {})
        if entry.get("intentional_zero_figures"):
            continue
        for spec in _config_roles(entry):
            if spec.get("optional"):
                continue
            if not any(
                _matches_spec(spec, Path(source).name, None) for source in sources
            ):
                problems.append(
                    _flatten_problem(
                        key,
                        "html_missing_figure",
                        spec.get("role", "unnamed"),
                    )
                )
    for page in sorted(gallery_root.rglob("index.html")):
        problems.extend(
            _broken_html_images(page, page.relative_to(html_dir).as_posix())
        )
    return problems


def check_gallery(
    examples_dir: Path,
    gallery_dir: Path,
    docs_dir: Path,
    backends: tuple[str, ...] = BACKENDS,
    *,
    root: Path | None = None,
    html_dir: Path | None = None,
) -> list[GalleryProblem]:
    """Check figure completeness and provenance without executing an example."""

    root = root or examples_dir.parent
    manifest = load_manifest(docs_dir)
    entries = manifest.get("examples", {})
    provenance = load_provenance(root)
    provenance_exists = provenance_path(root).exists()
    digest = dependency_digest(root)
    examples = [
        path
        for path in gallery_examples(examples_dir)
        if example_backend(path) in backends
    ]
    problems: list[GalleryProblem] = []

    for example in examples:
        key = example_key(example)
        entry = entries.get(key)
        if entry is None:
            problems.append(
                _flatten_problem(key, "manifest_missing", "no figure expectations")
            )
            continue

        current_md5 = get_md5sum(example, mode="t")
        md5_file = gallery_dir / example.parent.name / f"{example.name}.md5"
        if not md5_file.exists() or md5_file.read_text() != current_md5:
            problems.append(
                _flatten_problem(
                    key, "source_stale", "gallery source does not match the built page"
                )
            )

        if not provenance_exists:
            problems.append(
                _flatten_problem(
                    key, "provenance_missing", "no committed provenance sidecar"
                )
            )
        else:
            record = provenance.get(key)
            if record is None:
                problems.append(
                    _flatten_problem(key, "provenance_missing", "no entry for example")
                )
            else:
                if record.get("build_failed"):
                    problems.append(
                        _flatten_problem(
                            key,
                            "build_failed",
                            record.get("error", "the gallery build failed"),
                        )
                    )
                if record.get("source_md5") != current_md5:
                    problems.append(
                        _flatten_problem(
                            key, "provenance_stale", "recorded source md5 differs"
                        )
                    )
                if record.get("dependencies_sha256") != digest:
                    problems.append(
                        _flatten_problem(
                            key,
                            "dependency_changed",
                            "helpers, configs or settings changed after the build",
                        )
                    )

        base, _, stem = _gallery_base(gallery_dir, key)
        rst = base / f"{stem}.rst"
        if not rst.exists():
            problems.append(_flatten_problem(key, "page_missing", rst.name))
            continue
        page_text = rst.read_text(encoding="utf-8")
        embedded = embedded_figures(rst)
        embedded_names = []
        for item in embedded:
            image_path = _resolve_image(item["path"], docs_dir, gallery_dir)
            embedded_names.append(image_path.name)
            if not image_path.exists():
                problems.append(
                    _flatten_problem(
                        key,
                        "missing_image",
                        item["path"],
                    )
                )
                continue
            image_problem = _image_problem(image_path)
            if image_problem:
                problems.append(
                    _flatten_problem(
                        key, "corrupt_image", f"{image_path.name}: {image_problem}"
                    )
                )

        intentional = bool(entry.get("intentional_zero_figures"))
        if intentional and embedded:
            problems.append(
                _flatten_problem(
                    key, "unexpected_figure", "page is declared as zero-figure"
                )
            )
        if not intentional and not embedded:
            problems.append(
                _flatten_problem(
                    key, "no_embedded_figures", "no image-sg directives in the page"
                )
            )

        roles = _config_roles(entry)
        resolved = [
            (
                _resolve_image(item["path"], docs_dir, gallery_dir).name,
                item["alt"],
            )
            for item in embedded
        ]
        for spec in roles:
            if spec.get("optional"):
                continue
            if not any(_matches_spec(spec, name, alt) for name, alt in resolved):
                problems.append(
                    _flatten_problem(
                        key,
                        "missing_required_figure",
                        spec.get("role", "unnamed"),
                    )
                )

        count_problem = _figure_count_problem(entry, len(embedded))
        if count_problem:
            problems.append(_flatten_problem(key, "figure_count", count_problem))

        saved = _saved_figures(base, stem)
        unembedded = [
            figure.name for figure in saved if figure.name not in embedded_names
        ]
        if unembedded:
            problems.append(
                _flatten_problem(
                    key,
                    "unembedded_figure",
                    ", ".join(unembedded) + " saved but not embedded",
                )
            )
        if roles:
            undeclared = [
                name
                for name, alt in resolved
                if not any(_matches_spec(spec, name, alt) for spec in roles)
            ]
            if undeclared:
                problems.append(
                    _flatten_problem(key, "unexpected_figure", ", ".join(undeclared))
                )

        thumbnail = _thumbnail(base, stem)
        if not thumbnail.exists():
            problems.append(_flatten_problem(key, "thumbnail_missing", thumbnail.name))
        else:
            thumbnail_problem = _image_problem(thumbnail)
            if thumbnail_problem:
                problems.append(
                    _flatten_problem(
                        key,
                        "corrupt_thumbnail",
                        f"{thumbnail.name}: {thumbnail_problem}",
                    )
                )
            elif entry.get("thumbnail", "real") == "real" and (
                thumbnail_kind := default_thumbnail_kind(thumbnail)
            ):
                problems.append(
                    _flatten_problem(key, "placeholder_thumbnail", thumbnail_kind)
                )

        problems.extend(_check_downloads(example, base, stem, key))
        problems.extend(_check_run_summary(entry, docs_dir, key, page_text))

        record = provenance.get(key, {})
        if record.get("rst_sha256") and _sha256(rst) != record["rst_sha256"]:
            problems.append(
                _flatten_problem(
                    key, "page_changed", "page differs from the recorded build"
                )
            )
        for name, expected_hash in record.get("figures", {}).items():
            figure = base / "images" / name
            if figure.exists() and _sha256(figure) != expected_hash:
                problems.append(_flatten_problem(key, "figure_changed", name))
        for name, expected_hash in record.get("downloads", {}).items():
            download = base / name
            if download.exists() and _sha256(download) != expected_hash:
                problems.append(_flatten_problem(key, "download_changed", name))
        expected_thumbnail = record.get("thumbnail_sha256")
        if (
            expected_thumbnail
            and thumbnail.exists()
            and _sha256(thumbnail) != expected_thumbnail
        ):
            problems.append(_flatten_problem(key, "thumbnail_changed", thumbnail.name))

    all_keys = {example_key(path) for path in gallery_examples(examples_dir)}
    for key in sorted(entries):
        if key not in all_keys:
            problems.append(
                _flatten_problem(key, "manifest_orphan", "no matching example source")
            )

    if html_dir is not None:
        problems.extend(_check_html(html_dir, examples, entries))
    return problems


def stale_examples(
    examples_dir: Path,
    gallery_dir: Path,
    backends: tuple[str, ...] = BACKENDS,
    *,
    root: Path | None = None,
) -> list[Path]:
    """Return examples whose source or shared dependencies changed."""

    root = root or examples_dir.parent
    provenance = load_provenance(root)
    has_provenance = provenance_path(root).exists()
    digest = dependency_digest(root)
    stale = []
    for path in gallery_examples(examples_dir):
        if example_backend(path) not in backends:
            continue
        md5_file = gallery_dir / path.parent.name / f"{path.name}.md5"
        source_md5 = get_md5sum(path, mode="t")
        if not md5_file.exists() or md5_file.read_text() != source_md5:
            stale.append(path)
            continue
        if not has_provenance:
            continue
        record = provenance.get(example_key(path))
        if (
            record is None
            or record.get("source_md5") != source_md5
            or record.get("dependencies_sha256") != digest
            or record.get("build_failed")
        ):
            stale.append(path)
    return stale
