================
Fuzzy Regression
================

This guide covers regression with Ex-Fuzzy: predicting a continuous target with
rules you can read. It assumes you have met :doc:`core-concepts`.

Introduction
============

:class:`ex_fuzzy.BaseFuzzyRulesRegressor` learns Type-1 fuzzy rules for a
numeric target using a genetic algorithm. The input partitions are fixed before
the search starts, so every membership can be precomputed once and each
candidate rule base can be scored through a vectorized NumPy or PyTorch path.

The default ``backend="pymoo"`` runs that search on the CPU. With the optional
``backend="evox"``, population evolution and batched regression fitness run in
PyTorch on CUDA when a compatible GPU is available. EvoX automatically uses
the same PyTorch implementation on the CPU when CUDA is unavailable.

The estimator follows the scikit-learn API, so it works with ``cross_val_score``,
``Pipeline`` and ``GridSearchCV``.

Basic Workflow
==============

1. Prepare ``X`` (samples x features) and a one-dimensional numeric ``y``
2. Create a ``BaseFuzzyRulesRegressor``
3. Call ``fit`` with a generation and population budget
4. Call ``predict``, ``score`` and ``print_rules``

Quick Start Example
===================

.. code-block:: python

   import numpy as np
   from sklearn.datasets import make_friedman1
   from sklearn.model_selection import train_test_split
   from ex_fuzzy import BaseFuzzyRulesRegressor

   X, y = make_friedman1(n_samples=500, n_features=5, noise=0.5, random_state=0)
   X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

   regressor = BaseFuzzyRulesRegressor(nRules=20, nAnts=3, n_linguistic_variables=3)
   regressor.fit(X_train, y_train, n_gen=50, pop_size=50)

   print(regressor.score(X_test, y_test))
   regressor.print_rules()

GPU-Accelerated Search
======================

Install the optional backend and select it on the estimator:

.. code-block:: bash

   python -m pip install "ex-fuzzy[evox]"

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=30,
       nAnts=4,
       backend="evox",
       verbose=True,
   )
   regressor.fit(X_train, y_train, n_gen=50, pop_size=100)

After fitting, ``optimization_device_`` is ``"cuda"`` or ``"cpu"`` and
``gpu_accelerated_`` records whether CUDA handled the optimization. Both crisp
and fuzzy consequents and both rule modes use the batched PyTorch fitness path.
Population and sample chunks are sized from available memory to reduce the
risk of out-of-memory errors.

Consequent Types
================

Crisp Consequents (default)
---------------------------

Zero-order Takagi-Sugeno. Each rule carries a number, and a prediction is the
firing-strength-weighted average of the numbers of the rules that fired::

   Rule 1: IF x0 IS Low AND x2 IS High THEN output = 41.8203

**When to use:** the default. It gives the best numeric resolution, because a
consequent is a value rather than a label.

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=20, nAnts=3, consequent_type="crisp"
   )

Fuzzy Consequents (Mamdani Inference)
-------------------------------------

Each rule names an output fuzzy set. Inference clips every consequent set by
its rule's firing strength, aggregates the clipped sets with ``max``, and
defuzzifies by taking the centroid over a discretized universe::

   Rule 1: IF x0 IS Low AND x2 IS High THEN output IS Output_3

The output sets are not fixed in advance -- their trapezoids are evolved
alongside the rules. Each set's four breakpoints are sorted on decoding, so
every chromosome yields a well-formed trapezoid inside the target range.

**When to use:** when a rule should read as a complete linguistic statement,
and you can trade some numeric resolution for it.

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       consequent_type="fuzzy",
       n_output_lvs=4,          # number of output sets to evolve
       n_universe_points=101,   # centroid integration grid
   )

Rule Modes
==========

Additive Mode (default)
-----------------------

All rules contribute to every prediction, weighted by how strongly they fire.

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(nRules=10, nAnts=2, rule_mode="additive")

Sufficient Mode
---------------

Only each sample's strongest rule fires -- winner-takes-all. If even that rule
fires at or below ``tolerance``, the sample falls back to the training-target
mean.

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=10, nAnts=2, rule_mode="sufficient", tolerance=0.05
   )

**When to use:** when you want each prediction attributable to exactly one
rule. With crisp consequents the model becomes piecewise constant, since every
prediction is then one rule's consequent.

Reading the Fitted Model
========================

.. code-block:: python

   regressor.print_rules()                     # IF-THEN text
   text = regressor.print_rules(return_rules=True)
   rulebase = regressor.get_rulebase()         # the rule base object
   firing = rulebase.compute_rule_antecedent_memberships(X_test)

``firing`` has shape ``(n_samples, n_rules)`` and tells you which rule drove
each prediction.

Precomputed Linguistic Variables
================================

Pass your own partitions to keep the input labels fixed and comparable across
models:

.. code-block:: python

   from ex_fuzzy import utils, FUZZY_SETS

   partitions = utils.construct_partitions(X_train, FUZZY_SETS.t1, n_partitions=3)
   regressor = BaseFuzzyRulesRegressor(nRules=20, linguistic_variables=partitions)

Practical Notes
===============

- The search maximizes **training** :math:`R^2`. Estimate generalization with
  ``cross_val_score``, refitting inside each fold.
- ``n_gen`` matters more than ``pop_size`` for final quality; budget it first.
- ``nAnts`` is capped at the number of features. Rules may end up with fewer
  effective antecedents when the search selects the same feature twice.
- Only Type-1 fuzzy sets are supported so far.
