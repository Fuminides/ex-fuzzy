"""Tests for small helpers of the fitness and FERL backends."""

import importlib

import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef

import _ferl_backend
import _fitness


def test_mcc_falls_back_to_sklearn_for_non_integer_labels():
    y = np.array([0.0, 1.0, 1.0, 0.0])
    prediction = np.array([0.0, 1.0, 0.0, 0.0])

    assert _fitness._mcc(y, prediction) == matthews_corrcoef(y, prediction)
    assert _fitness._mcc(y.astype(int), prediction.astype(int)) == pytest.approx(matthews_corrcoef(y, prediction))


def test_kernels_load_from_the_package_or_explain_how_to_build_them(monkeypatch):
    package_backend = importlib.import_module('ex_fuzzy.ex_fuzzy._ferl_backend')
    try:
        kernels = package_backend.load_kernels()
    except ImportError as error:
        assert 'EX_FUZZY_BUILD_FERL=1' in str(error)
    else:
        # Editable builds may expose the extension under its historical
        # top-level name even when it was requested through the package.
        assert kernels.__name__.rsplit('.', 1)[-1] == '_ferl_kernels'

    def missing(*args, **kwargs):
        raise ImportError('no extension')

    monkeypatch.setattr(_ferl_backend, 'import_module', missing)
    with pytest.raises(ImportError, match="backend='python'"):
        _ferl_backend.load_kernels()
