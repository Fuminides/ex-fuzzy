=============
API Reference
=============

This section provides detailed documentation for all Ex-Fuzzy classes, functions, and modules.

.. currentmodule:: ex_fuzzy

Module Overview
===============

.. toctree::
   :maxdepth: 1

   fuzzy_sets
   rules
   classifiers
   evolutionary_fit
   rule_mining
   eval_tools
   regression

Quick Reference
===============

Most Common Classes
-------------------

- :class:`ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier`
- :class:`ex_fuzzy.evolutionary_fit_regression.BaseFuzzyRulesRegressor`
- :class:`ex_fuzzy.fuzzy_sets.fuzzyVariable`
- :class:`ex_fuzzy.fuzzy_sets.FS`
- :class:`ex_fuzzy.rules.RuleSimple`
- :class:`ex_fuzzy.rules.RuleBase`
- :class:`ex_fuzzy.eval_tools.FuzzyEvaluator`

Most Common Functions
---------------------

- :func:`ex_fuzzy.eval_tools.eval_fuzzy_model`
- :func:`ex_fuzzy.utils.construct_partitions`
- :func:`ex_fuzzy.utils.t1_fuzzy_partitions_dataset`
- :func:`ex_fuzzy.utils.t2_fuzzy_partitions_dataset`
- :func:`ex_fuzzy.vis_rules.plot_fuzzy_variable`
- :func:`ex_fuzzy.vis_rules.visualize_rulebase`
- :func:`ex_fuzzy.persistence.save_fuzzy_variables`
- :func:`ex_fuzzy.persistence.load_fuzzy_variables`

Constants and Enums
===================

- :class:`ex_fuzzy.fuzzy_sets.FUZZY_SETS`

Type Hints
==========

Ex-Fuzzy uses type hints throughout the codebase. Here are the most common types:

.. code-block:: python

   from typing import List, Dict, Tuple, Optional, Union
   import numpy as np
   import pandas as pd
   
   # Common type aliases used in Ex-Fuzzy
   ArrayLike = Union[np.ndarray, List, Tuple]
   DataFrame = pd.DataFrame
   FuzzySet = 'ex_fuzzy.fuzzy_sets.FS'
   FuzzyVariable = 'ex_fuzzy.fuzzy_sets.fuzzyVariable'
   Rule = 'ex_fuzzy.rules.RuleSimple'
