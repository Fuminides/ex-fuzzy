"""Importing the package loads nothing heavy until a submodule or class is used."""
import os
from pathlib import Path
import subprocess
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[1]


def _run(source, tmp_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT)
    return subprocess.run([sys.executable, '-c', textwrap.dedent(source)], cwd=tmp_path, env=env,
                          check=True, capture_output=True, text=True)


def test_package_import_is_lazy(tmp_path):
    _run("""
        import sys
        import ex_fuzzy
        assert ex_fuzzy.__version__
        loaded = {name for name in sys.modules if name.startswith('ex_fuzzy.')}
        assert loaded == {'ex_fuzzy._version'}, loaded
        assert 'sklearn' not in sys.modules and 'pandas' not in sys.modules

        # Attribute access, from-imports and submodule imports all resolve.
        from ex_fuzzy import BaseFuzzyRulesClassifier
        assert 'ex_fuzzy.evolutionary_fit' in sys.modules
        assert ex_fuzzy.BaseFuzzyRulesClassifier is BaseFuzzyRulesClassifier
        assert ex_fuzzy.FUZZY_SETS is ex_fuzzy.fuzzy_sets.FUZZY_SETS
        import ex_fuzzy.rules
        assert ex_fuzzy.rules.RuleSimple([0]).antecedents == [0]
        assert 'rules' in dir(ex_fuzzy) and 'FERL' in dir(ex_fuzzy)
        try:
            ex_fuzzy.no_such_name
        except AttributeError as error:
            assert 'no_such_name' in str(error)
        else:
            raise AssertionError('unknown attributes must raise AttributeError')
    """, tmp_path)


def test_star_import_exposes_every_export(tmp_path):
    _run("""
        namespace = {}
        exec('from ex_fuzzy import *', namespace)
        for name in ('FERL', 'DeepFERL', 'FuzzyRulesClassifier', 'BaseFuzzyRulesRegressor',
                     'fuzzyVariable', 'rules', 'utils', 'persistence'):
            assert name in namespace, name
    """, tmp_path)
