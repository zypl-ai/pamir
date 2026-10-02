"""Sphinx configuration for PaMIR documentation."""

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "PaMIR"
copyright = "2026, zypl.ai"
author = "zypl.ai"
release = "0.4.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "sphinx_llms_txt",
    "myst_parser",
    "sphinxcontrib.mermaid",
    "sphinx.ext.mathjax",
]

# MyST extensions: dollarmath renders $$…$$ / $…$ (LaTeX) via MathJax; amsmath
# handles aligned/cases environments used in the design record.
myst_enable_extensions = ["dollarmath", "amsmath"]

# ── sphinx-llms-txt ────────────────────────────────────────────────
llms_txt_description = (
    "PaMIR: an open benchmark for credit-default prediction when labels are "
    "scarce and arrive late: 19 public credit-default datasets rebuilt from "
    "pinned sources, a label-delayed streaming protocol that scores each "
    "application on arrival, an i.i.d. split, and a leakage-controlled "
    "synthetic-data harness."
)

# MyST (markdown) support
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
# Render ```mermaid fenced blocks (GitHub renders them natively too) via the
# sphinxcontrib.mermaid directive instead of trying to syntax-highlight them.
myst_fence_as_directive = ["mermaid"]

# sphinxcontrib.mermaid already picks the mermaid theme from the page's
# light/dark state, so diagrams re-theme with Furo.  The diagrams therefore
# carry no hard-coded fills (only coloured strokes for semantics), which keeps
# node text legible in both themes; see custom.css for the wide-diagram scroll.

templates_path = ["_templates"]
exclude_patterns = ["_build"]

# ── Furo theme ──────────────────────────────────────────────────────
html_theme = "furo"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#1a6fb4",
        "color-brand-content": "#1a6fb4",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eaadf",
        "color-brand-content": "#5eaadf",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_buttons": [],
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]

# ── sphinx-copybutton ──────────────────────────────────────────────
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# ── autodoc ────────────────────────────────────────────────────────
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}
