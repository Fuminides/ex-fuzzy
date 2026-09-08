"""Lazy loading for the optional compiled FERL backend."""
from importlib import import_module


def load_kernels():
    """Load the extension only when explicitly requested by a FERL user."""
    try:
        if __package__:
            return import_module("._ferl_kernels", __package__)
        return import_module("_ferl_kernels")
    except ImportError as exc:
        raise ImportError(
            "FERL backend='cython' requires the compiled extension. From the "
            "ex-fuzzy source directory run `python -m pip install Cython numpy` "
            "then `EX_FUZZY_BUILD_FERL=1 python -m pip install "
            "--no-build-isolation -e .`. A C compiler is required. "
            "Use backend='python' for the default implementation."
        ) from exc
