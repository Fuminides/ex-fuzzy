"""Smoke tests for the benchmark-only firing backend factories."""

import importlib.util
from pathlib import Path

import pytest


_MODULE_PATH = Path(__file__).parents[1] / "benchmarks" / "prototype_firing_backends.py"
_SPEC = importlib.util.spec_from_file_location("prototype_firing_backends", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.skipif(
    importlib.util.find_spec("numba") is None,
    reason="Numba is optional for the benchmark prototype",
)
def test_numba_factories_are_callable_when_numba_is_installed():
    """Both rank-specific factories must not silently skip when available."""
    assert callable(_MODULE._make_numba_product())
    assert callable(_MODULE._make_numba_product_t2())
