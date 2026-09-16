"""
Ex-Fuzzy: explainable fuzzy rule-based learning.

Submodules and the top-level classes are imported on first access, so
``import ex_fuzzy`` is cheap and each module's dependencies load only when
that module is used. ``from ex_fuzzy import BaseFuzzyRulesClassifier`` and
``ex_fuzzy.rules`` work as before.
"""
import importlib

from ._version import __version__

_SUBMODULES = (
    'bootstrapping_test', 'centroid', 'classifiers', 'cognitive_maps', 'conformal', 'eval_rules',
    'eval_tools', 'evolutionary_backends', 'evolutionary_fit', 'evolutionary_fit_regression',
    'evolutionary_search', 'ferl', 'ferl_deep', 'ferl_partitions', 'fuzzy_sets', 'pattern_stability',
    'permutation_test', 'persistence', 'rule_mining', 'rules', 'temporal', 'utils', 'vis_rules',
)

#: Top-level names and the submodule that defines each of them.
_EXPORTS = {
    'BaseFuzzyRulesClassifier': 'evolutionary_fit',
    'FitRuleBase': 'evolutionary_fit',
    'BaseFuzzyRulesRegressor': 'evolutionary_fit_regression',
    'FitRuleBaseRegression': 'evolutionary_fit_regression',
    'RuleBaseT1MamdaniRegression': 'evolutionary_fit_regression',
    'RuleBaseT1Regression': 'evolutionary_fit_regression',
    'FuzzyRulesClassifier': 'classifiers',
    'RuleFineTuneClassifier': 'classifiers',
    'RuleMineClassifier': 'classifiers',
    'ConformalFuzzyClassifier': 'conformal',
    'evaluate_conformal_coverage': 'conformal',
    'FS': 'fuzzy_sets',
    'FUZZY_SETS': 'fuzzy_sets',
    'fuzzyVariable': 'fuzzy_sets',
    'FERL': 'ferl',
    'DeepFERL': 'ferl_deep',
    'learn_partitions_mdlp': 'ferl_partitions',
    'mdlp_cuts': 'ferl_partitions',
}

__all__ = ['__version__', *_SUBMODULES, *_EXPORTS]


def __getattr__(name):
    if name in _SUBMODULES:
        return importlib.import_module(f'.{name}', __name__)
    if name in _EXPORTS:
        value = getattr(importlib.import_module(f'.{_EXPORTS[name]}', __name__), name)
        globals()[name] = value
        return value
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | set(__all__))
