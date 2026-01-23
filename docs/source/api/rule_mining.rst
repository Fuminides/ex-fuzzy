Rule Mining Module
==================

The :mod:`ex_fuzzy.rule_mining` module provides fuzzy rule mining capabilities for extracting meaningful rules from datasets.

.. currentmodule:: ex_fuzzy.rule_mining

Overview
--------

This module implements algorithms for discovering frequent fuzzy patterns and generating fuzzy rules using support-based itemset mining.

Core Functions
--------------

Rule Search
~~~~~~~~~~~

.. autofunction:: rule_search

Rule Generation
~~~~~~~~~~~~~~~

.. autofunction:: generate_rules_from_itemsets

Mining Functions
~~~~~~~~~~~~~~~~

.. autofunction:: mine_rulebase_support
.. autofunction:: multiclass_mine_rulebase
.. autofunction:: simple_mine_rulebase
.. autofunction:: simple_multiclass_mine_rulebase

Rule Pruning
~~~~~~~~~~~~

.. autofunction:: prune_rules_confidence_lift

Examples
--------

Basic Rule Mining
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.rule_mining as rm
   import ex_fuzzy.fuzzy_sets as fs
   import pandas as pd
   import numpy as np

   # Prepare data
   X = np.random.rand(100, 4)
   y = np.random.randint(0, 3, 100)
   data = pd.DataFrame(X)

   # Create fuzzy variables
   import ex_fuzzy.utils as utils
   fuzzy_vars = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)

   # Mine rules
   itemsets = rm.rule_search(data, fuzzy_vars, support_threshold=0.1)
   rules = rm.generate_rules_from_itemsets(itemsets, nAnts=3)

Multiclass Rule Mining
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mine rules for multiclass problem
   rules = rm.multiclass_mine_rulebase(
       x=data,
       y=y,
       fuzzy_variables=fuzzy_vars,
       support_threshold=0.05,
       max_depth=3
   )

See Also
--------

* :mod:`ex_fuzzy.rules` : Rule representation classes
* :mod:`ex_fuzzy.fuzzy_sets` : Fuzzy variable definitions
* :mod:`ex_fuzzy.evolutionary_fit` : Rule optimization
