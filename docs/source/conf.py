# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
import os
from datetime import datetime

# -- Path setup --------------------------------------------------------------
sys.path.insert(0, os.path.abspath('../../ex_fuzzy/'))

# -- Project information -----------------------------------------------------
project = 'Ex-Fuzzy'
copyright = f'2023-{datetime.now().year}, Javier Fumanal Idocin'
author = 'Javier Fumanal Idocin'
release = '1.0.0'
version = '1.0.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.doctest',
    'sphinx.ext.duration',
    'sphinx.ext.extlinks',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx_copybutton',
    'sphinx_design',
    'myst_parser',
    'nbsphinx',
]

# Templates and patterns
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Extension configuration -------------------------------------------------

# Autodoc
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Autosummary
autosummary_generate = True
autosummary_generate_overwrite = True

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
}

# External links
extlinks = {
    'issue': ('https://github.com/fuminides/ex-fuzzy/issues/%s', 'issue %s'),
    'pr': ('https://github.com/fuminides/ex-fuzzy/pull/%s', 'PR %s'),
}

# Todo configuration
todo_include_todos = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'
html_title = f"Ex-Fuzzy {version}"

html_theme_options = {
    "logo": {
        "text": "Ex-Fuzzy",
        "image_light": "_static/logo-light.png",
        "image_dark": "_static/logo-dark.png",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/fuminides/ex-fuzzy",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/ex-fuzzy/",
            "icon": "fas fa-box",
            "type": "fontawesome",
        },
    ],
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["navbar-icon-links", "theme-switcher"],
    "navbar_persistent": ["search-button"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "show_prev_next": False,
    "search_bar_text": "Search the docs...",
    "navigation_with_keys": False,
    "show_toc_level": 2,
    "announcement": None,
}

html_context = {
    "github_user": "fuminides",
    "github_repo": "ex-fuzzy",
    "github_version": "main",
    "doc_path": "docs/source",
}

html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Favicon
html_favicon = '_static/favicon.ico'

# Show source links
html_show_sourcelink = True
html_copy_source = True

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    ('index', 'ex-fuzzy.tex', 'Ex-Fuzzy Documentation',
     'Javier Fumanal Idocin', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'ex-fuzzy', 'Ex-Fuzzy Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'ex-fuzzy', 'Ex-Fuzzy Documentation',
     author, 'ex-fuzzy', 'A library for explainable fuzzy logic inference.',
     'Miscellaneous'),
]

# -- Options for epub output -------------------------------------------------
epub_title = project
epub_exclude_files = ['search.html']


