import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
source_suffix = ".rst"
master_doc = "index"

project = "torch_ransac3d"
copyright = "2023, Harry Dobbs"
author = "Harry Dobbs"

version = ""
release = ""

language = "en"  # Changed from None to 'en'

exclude_patterns = []
pygments_style = "sphinx"

html_theme = "alabaster"  # Using a simple, built-in theme

html_static_path = ["_static"]
