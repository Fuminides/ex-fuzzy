Evolutionary Fit Module
=======================

The :mod:`ex_fuzzy.evolutionary_fit` module implements genetic algorithm-based optimization for learning fuzzy rule bases.

.. currentmodule:: ex_fuzzy.evolutionary_fit

Overview
--------

This module provides automatic rule discovery, parameter tuning, and structure optimization for fuzzy inference systems using evolutionary computation techniques.

**Core Capabilities:**

* **Automatic Rule Learning**: Discover optimal rule antecedents and consequents
* **Multi-objective Optimization**: Balance accuracy vs. interpretability
* **Parallel Evaluation**: Efficient fitness computation using threading
* **Cross-validation**: Robust fitness evaluation with stratified CV
* **Pymoo Integration**: Leverages the powerful Pymoo optimization framework

**Optimization Targets:**

* Rule antecedents (variable and linguistic term selection)
* Rule consequents (output class assignments)  
* Rule structure (number of rules, complexity constraints)
* Membership function parameters (integration with other modules)

Classes
-------

Optimization Problem
--------------------

FitRuleBase
~~~~~~~~~~~

.. autoclass:: FitRuleBase
   :members:
   :inherited-members:
   :show-inheritance:

Examples
--------

Basic Rule Base Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.evolutionary_fit as evf
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np
   from sklearn.datasets import load_iris

   # Load data
   X, y = load_iris(return_X_y=True)

   import ex_fuzzy.utils as utils

   # Create linguistic variables
   antecedents = utils.construct_partitions(X, fs.FUZZY_SETS.t1, n_partitions=3)

   # Train a fuzzy classifier
   classifier = evf.BaseFuzzyRulesClassifier(
       nRules=10,
       nAnts=3,
       linguistic_variables=antecedents,
   )
   classifier.fit(X, y, n_gen=50, pop_size=100)

See Also
--------

* :mod:`ex_fuzzy.classifiers` : High-level classification interfaces
* :mod:`ex_fuzzy.rules` : Rule base classes and inference
* :mod:`ex_fuzzy.rule_mining` : Rule discovery algorithms
* :mod:`ex_fuzzy.eval_tools` : Performance evaluation utilities
