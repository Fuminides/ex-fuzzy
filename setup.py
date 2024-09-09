#! /usr/bin/env python
"""Toolbox for explainable AI using fuzzy logic."""
from __future__ import absolute_import

import os
# read the contents of your README file
from os import path

from setuptools import find_packages, setup

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
VERSION = "1.4.5"
INSTALL_REQUIRES = ["numpy", "networkx", "matplotlib", "pymoo", "pandas", "scikit-learn"]
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
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=CLASSIFIERS,
)
