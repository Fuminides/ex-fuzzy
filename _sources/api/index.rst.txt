=============
API Reference
=============

This section provides detailed documentation for all Ex-Fuzzy classes, functions, and modules.

.. currentmodule:: ex_fuzzy

Core Modules
============

.. autosummary::
   :toctree: generated/
   :template: module_template.rst

   fuzzy_sets
   evolutionary_fit
   rules
   classifiers
   eval_tools

Analysis and Visualization
==========================

.. autosummary::
   :toctree: generated/
   :template: module_template.rst

   pattern_stability
   vis_rules
   cognitive_maps
   temporal

Utilities and Support
=====================

.. autosummary::
   :toctree: generated/
   :template: module_template.rst

   utils
   persistence
   bootstrapping_test
   eval_rules

Quick Reference
===============

Most Common Classes
-------------------

.. autosummary::
   :toctree: generated/

   evolutionary_fit.BaseFuzzyRulesClassifier
   fuzzy_sets.fuzzyVariable
   fuzzy_sets.FS
   rules.RuleSimple
   rules.RuleBase
   eval_tools.FuzzyEvaluator

Most Common Functions
---------------------

.. autosummary::
   :toctree: generated/

   utils.load_data
   utils.preprocess_data
   persistence.save_model
   persistence.load_model
   vis_rules.plot_fuzzy_variable
   pattern_stability.pattern_stabilizer

By Category
===========

Fuzzy Set Operations
--------------------

.. autosummary::
   :toctree: generated/

   fuzzy_sets.FS
   fuzzy_sets.gaussianFS
   fuzzy_sets.gaussianIVFS
   fuzzy_sets.categoricalFS
   fuzzy_sets.fuzzyVariable

Classification
--------------

.. autosummary::
   :toctree: generated/

   evolutionary_fit.BaseFuzzyRulesClassifier
   classifiers.RuleMineClassifier
   classifiers.DoubleGo

Rule Management
---------------

.. autosummary::
   :toctree: generated/

   rules.RuleSimple
   rules.RuleBase
   rules.MasterRuleBase
   rules.generate_rule_string

Evaluation and Metrics
-----------------------

.. autosummary::
   :toctree: generated/

   eval_tools.FuzzyEvaluator
   eval_tools.accuracy_score
   eval_tools.classification_report
   bootstrapping_test.bootstrap_test

Visualization
-------------

.. autosummary::
   :toctree: generated/

   vis_rules.plot_fuzzy_variable
   vis_rules.plot_rules
   vis_rules.plot_membership_functions
   pattern_stability.pattern_stabilizer.pie_chart_basic

Model Persistence
-----------------

.. autosummary::
   :toctree: generated/

   persistence.save_model
   persistence.load_model
   persistence.save_fuzzy_variables
   persistence.load_fuzzy_variables

Constants and Enums
===================

.. autosummary::
   :toctree: generated/

   fuzzy_sets.FUZZY_SETS
   fuzzy_sets.LINGUISTIC_TYPES

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

Inheritance Diagrams
====================

.. inheritance-diagram:: ex_fuzzy.fuzzy_sets.FS ex_fuzzy.fuzzy_sets.gaussianFS ex_fuzzy.fuzzy_sets.gaussianIVFS
   :parts: 1

.. inheritance-diagram:: ex_fuzzy.rules.RuleSimple ex_fuzzy.rules.RuleBase ex_fuzzy.rules.MasterRuleBase
   :parts: 1

Module Documentation
====================

.. toctree::
   :maxdepth: 1

   fuzzy_sets
   classifiers
   rules
   evolutionary_fit
   rule_mining
   eval_tools
   pattern_stability
   vis_rules
   cognitive_maps
   temporal
   utils
   persistence
   bootstrapping_test
   eval_rules
