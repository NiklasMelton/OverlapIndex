"""Sphinx configuration for the OverlapIndex documentation."""

from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path
import re
import sys


REPOSITORY_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))


project = "OverlapIndex"
author = "Niklas M. Melton"
copyright = "2026, Niklas M. Melton"
try:
    release = package_version("overlapindex")
except PackageNotFoundError:
    pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text()
    release = re.search(r'^version = "([^"]+)"', pyproject, re.MULTILINE).group(1)
version = release

extensions = [
    "myst_parser",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

# MyST leaves dollar-delimited math disabled unless this extension is enabled.
# It applies to both inline ``$...$`` and display ``$$...$$`` expressions in
# the Markdown documentation; Sphinx renders the resulting math with MathJax.
myst_enable_extensions = ["dollarmath"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_logo = "../img/overlap_index_logo.png"
html_theme_options = {
    "logo_only": True,
    "navigation_depth": 3,
}
html_static_path = ["_static"]
html_css_files = ["css/custom.css"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_preserve_defaults = True
autoclass_content = "both"
napoleon_numpy_docstring = True
napoleon_google_docstring = False
