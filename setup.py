#! /usr/bin/env python
"""Toolbox for explainable AI using fuzzy logic."""
from __future__ import absolute_import

import os
# read the contents of your README file
from os import path

from setuptools import setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

DISTNAME = "ex_fuzzy"
DESCRIPTION = "Library to perform explainable AI using fuzzy logic."
MAINTAINER = "Javier Fumanal Idocin"
MAINTAINER_EMAIL = "javierfumanalidocin@gmail.com"
URL = "https://github.com/Fuminides/ex-fuzzy"
LICENSE = "GPL-3.0"
DOWNLOAD_URL = "https://pypi.org/project/ex-fuzzy/"
version_ns = {}
with open(path.join(this_directory, "ex_fuzzy", "ex_fuzzy", "_version.py"), encoding="utf-8") as f:
    exec(f.read(), version_ns)

VERSION = version_ns["__version__"]
INSTALL_REQUIRES = ["numpy", "matplotlib", "pymoo", "pandas", "scikit-learn"]
DOCS_REQUIRES = [
    "sphinx>=5.0.0",
    "pydata-sphinx-theme>=0.13.0",
    "sphinx-design>=0.4.0",
    "myst-parser>=0.18.0",
    "sphinx-copybutton>=0.5.0",
    "sphinx-autobuild>=2021.3.14",
    "nbsphinx>=0.8.0",
    "sphinxcontrib-bibtex>=2.5.0",
    "sphinx-gallery>=0.11.0",
    "sphinx-autodoc-typehints>=1.19.0",
    "sphinx-autoapi>=2.1.0",
    "pygments>=2.12.0",
    "jupyter>=1.0.0",
    "ipykernel>=6.0.0",
    "sphinx-math-dollar>=1.2.0",
    "seaborn>=0.11.0",
    "sphinxext-opengraph>=0.7.0",
    "sphinx-external-toc>=0.3.0",
    "doc8>=0.11.0",
    "rstcheck>=6.0.0",
]
OPTIONAL_REQUIRES = {
    "viz": ["networkx"],
    "gpu": ["torch"],
    "evox": ["evox[jax]"],
    "docs": DOCS_REQUIRES,
    "test": ["pytest", "pytest-cov"],
}
OPTIONAL_REQUIRES["dev"] = OPTIONAL_REQUIRES["test"] + OPTIONAL_REQUIRES["docs"]
OPTIONAL_REQUIRES["all"] = sorted(
    {dependency for dependencies in OPTIONAL_REQUIRES.values() for dependency in dependencies}
)
CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    "Programming Language :: Python",
    "Programming Language :: Python :: 3.7",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Software Development :: Libraries",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    packages=['ex_fuzzy'],
    package_dir={'ex_fuzzy': 'ex_fuzzy/ex_fuzzy'},
    install_requires=INSTALL_REQUIRES,
    extras_require=OPTIONAL_REQUIRES,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,

)
