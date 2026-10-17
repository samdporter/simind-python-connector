"""Sphinx configuration file for simind-python-connector documentation."""

import os
import sys
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path

from sphinx_gallery.sorting import ExplicitOrder


# Add the package to the path
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("."))

from _gallery import BACKENDS, execution_pattern  # noqa: E402


# Project information
project = "simind-python-connector"
copyright = f"{datetime.now().year}, Sam Porter, Efstathios Varzakis"
author = "Sam Porter, Efstathios Varzakis"
release = "2.0.1"
for dist_name in ("simind-python-connector",):
    try:
        release = importlib_metadata.version(dist_name)
        break
    except importlib_metadata.PackageNotFoundError:
        continue
version = ".".join(release.split(".")[:2])

# General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "myst_parser",
    "sphinx_gallery.gen_gallery",
]
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "superpowers/**",
    "examples",
]

# Options for HTML output
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}

# Autodoc configuration
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}
autodoc_mock_imports = [
    "sirf",
    "stir",
    "stirextra",
    "cil",
    "setr",
    "torch",
    "pytomography",
]

# Napoleon settings for NumPy and Google style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
    "tasklist",
]

# Sphinx-gallery: each build executes only the examples of one backend, the
# rest render from their committed outputs.
backend = os.environ.get("SIMIND_GALLERY_BACKEND")
if backend and backend not in BACKENDS:
    raise ValueError(
        f"SIMIND_GALLERY_BACKEND must be one of {BACKENDS}, got {backend!r}"
    )

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": (
        execution_pattern(Path("../examples").resolve(), backend) if backend else r"^$"
    ),
    "plot_gallery": "True" if backend else "False",
    "ignore_pattern": r"^(?!plot_)[^/]*\.py$",
    "within_subsection_order": "FileNameSortKey",
    "subsection_order": ExplicitOrder(
        ["../examples/getting_started", "../examples/backends"]
    ),
    "abort_on_example_error": True,
    "remove_config_comments": True,
    "download_all_examples": False,
    "image_scrapers": ("matplotlib",),
    "matplotlib_animations": False,
}
