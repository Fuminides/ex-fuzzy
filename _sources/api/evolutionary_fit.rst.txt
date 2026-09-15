Evolutionary Fit Module
=======================

The :mod:`ex_fuzzy.evolutionary_fit` module implements evolutionary optimization
for fuzzy rule-based classifiers.

.. currentmodule:: ex_fuzzy.evolutionary_fit

Overview
--------

This module provides:

- :class:`BaseFuzzyRulesClassifier` for end-to-end fuzzy classifier training.
- :class:`FitRuleBase` as the optimization problem used internally.

Training performance
--------------------

The built-in T1/T2 classification objective automatically uses an optimized CPU
evaluator. It decodes each chromosome with the reference rule constructor,
computes firing strengths once, and reuses them for dominance scoring, pruning
and final winner-rule prediction. Rules that never win a correctly classified
sample are still removed. Reporting metrics are computed for the selected model
rather than repeatedly for every candidate. Integer-label MCC uses a direct
confusion-matrix calculation.

No public parameters change, and no compiler or additional dependency is needed.
Fixed partitions retain their precomputed memberships; optimized partitions are
recomputed for each chromosome. Custom losses, nonnumeric internal labels and
other fuzzy types retain the full evaluator. EvoX classification also uses this
CPU path. Regression and candidate-rule mining retain their existing evaluators.

To compare exact fitness values, selected chromosomes and predictions while
measuring candidate evaluation and complete seeded fits, run from the repository
root:

.. code-block:: bash

   python benchmarks/benchmark_classifier_fitness.py --samples 1000

The script reports median times over three runs for both fixed and optimized
partitions, using identical population sizes and generation budgets with early
stopping disabled. Run it without competing workloads for useful timings.
Speedups depend on data size, rule count and hardware. Population batching and
additional parallelism are separate future optimization steps.

BaseFuzzyRulesClassifier
------------------------

.. autoclass:: BaseFuzzyRulesClassifier
   :members:
   :show-inheritance:

FitRuleBase
-----------

.. autoclass:: FitRuleBase
   :members:
   :show-inheritance:

See Also
--------

* :mod:`ex_fuzzy.classifiers`
* :mod:`ex_fuzzy.rules`
* :mod:`ex_fuzzy.rule_mining`
* :mod:`ex_fuzzy.eval_tools`
