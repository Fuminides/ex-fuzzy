#! /usr/bin/env python
"""Build script for Ex-Fuzzy.

The package metadata lives in ``pyproject.toml``. This script only adds the
opt-in native FERL extension, which needs a C compiler and Cython.
"""
import os

from setuptools import setup, Extension

# Native FERL is opt-in so ordinary installations need no compiler or Cython.
ext_modules = []
if os.environ.get("EX_FUZZY_BUILD_FERL") == "1":
    from Cython.Build import cythonize
    import numpy as np

    ext_modules = cythonize(
        [Extension(
            "ex_fuzzy._ferl_kernels",
            ["ex_fuzzy/_ferl_kernels.pyx"],
            include_dirs=[np.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            extra_compile_args=(["/O2", "/fp:strict"] if os.name == "nt"
                                else ["-O3", "-ffp-contract=off"]),
        )],
        compiler_directives={"language_level": "3"},
        build_dir="build/cython",
    )

setup(ext_modules=ext_modules)
