# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Ex-Fuzzy'
copyright = '2023, Javier Fumanal Idocin'
author = 'Javier Fumanal Idocin'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = []

autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}
autosummary_generate = True

import sys
import os
print(os.getcwd())
sys.path.insert(0, os.path.abspath('../../ex_fuzzy/'))


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ['_static']

html_show_sourcelink = True


