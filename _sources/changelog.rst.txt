=========
Changelog
=========

This document tracks all notable changes to Ex-Fuzzy.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

[3.1.0] - 2026-09-15
====================

Added
-----
- **EvoX GPU classification fitness**: On a CUDA device the EvoX backend can
  score whole generations of the built-in Type-1 classification objective with
  an exact PyTorch implementation, for fixed or optimized partitions. Each fit
  checks it against the CPU on a few candidates of its first generation and
  then uses it only where it measures faster, so results are unchanged

Changed
-------
- **EvoX classification speed**: EvoX fits use the fit-local fitness and firing
  caches and population batching, and score chromosomes repeated within a
  generation once. Results are unchanged
- **Final model evaluation**: Classification fits reuse the selected fuzzy
  memberships and firing strengths while calculating rule weights, pruning,
  and final metrics. The temporary arrays are released before resampling or
  fit return, and the final model is unchanged
- The fitness cache holds four populations of chromosomes (at least 256)
- **pymoo is imported only when used**: importing Ex-Fuzzy and EvoX fits no
  longer import pymoo. ``FitRuleBase``, ``FitRuleBaseRegression`` and
  ``ExploreRuleBases`` no longer subclass pymoo's ``Problem``; the PyMoo backend
  and the temporal classifier wrap them when they run. To pass one to pymoo
  directly, wrap it with ``evolutionary_backends.as_pymoo_problem``

Fixed
-----
- The batched population evaluator added the ``alpha`` size penalty when every
  surviving rule scored exactly the tolerance; the reference adds none
- Pattern stability usage charts show "No variable usage" instead of failing
  when no variable is used

Removed
-------
- The unused NetworkX rule-graph visualization and ``viz`` installation extra.
  Text/LaTeX rule output and fuzzy-partition plotting remain available.

[3.0.0] - 2026-09-14
====================

Added
-----
- **FERL**: Native Fast Evidential Rule Learning with compact or learned fuzzy
  splits, soft rule aggregation, Dempster--Shafer belief/plausibility,
  ignorance, prediction sets, missing-feature masks, and MDLP partitioning
- **FERL Demo and Documentation**: Runnable Iris example, user guide, and API
  reference
- **DeepFERL**: Deep evidential rule trees grown recursively with weighted-Gini
  learned soft splits and a leaves-only soft vote, with bounded-support
  out-of-distribution handling, missing-feature masks, and the same evidential
  outputs as FERL. Ports ``LearnedFuzzyTree`` (FERL-deep) from
  ``fuzzy_greedy_tree`` and reproduces its trees and predictions
- **FERL evidence**: The ``"mixture"`` combination rule, now shared with
  DeepFERL through a common evidence module
- **FuzzyRulesClassifier**: Rebuilt as a FARC-HD-style fuzzy association rule
  classifier. It caps features by relevance, mines candidate rules on fixed
  partitions, prescreens them by covering subgroup discovery, and selects a
  compact certainty-factor-weighted rule base with a vectorized genetic
  algorithm. It is a proper scikit-learn estimator; the earlier constructor
  and ``fit`` arguments still work. Rules combine additively or
  sufficiently (``rule_mode``), as in Ex-Fuzzy regression. Rules with four
  or five conditions are mined with Apriori-style support pruning, which keeps
  the same candidates as exhaustive enumeration
- **Fuzzy Regression**: Scikit-learn-compatible Type-1 rule learning for
  continuous targets with crisp Takagi-Sugeno and fuzzy Mamdani consequents
- **GPU-Accelerated Regression**: EvoX/PyTorch population evaluation for both
  regression consequent types and additive or sufficient rule modes
- **Regression Runtime Metadata**: Fitted estimators expose ``backend_``,
  ``optimization_device_``, ``gpu_accelerated_``, and generation status
- **EvoX Backend Support**: GPU-accelerated evolutionary optimization using EvoX and PyTorch
- **Automatic Memory Management**: Batch processing for large datasets to prevent out-of-memory errors
- **Performance Improvements**: 2-10x speedup for large datasets with GPU acceleration
- **Backend Selection**: Easy switching between PyMoo (CPU) and EvoX (GPU) backends
- Comprehensive test suite with 100% statement and branch coverage
- Modern documentation website with PyData theme
- Interactive examples with Jupyter notebooks
- GitHub Actions CI/CD pipeline
- Type hints throughout the codebase
- Performance benchmarking suite

Changed
-------
- **Fuzzy Tree Learning**: The experimental prototype implementations were
  replaced by the public :class:`ex_fuzzy.FERL` estimator
- **FERL validation**: ``split_mode="learned"`` with ``target_metric="purity"``
  now raises ``ValueError`` instead of silently ignoring the learned splits, and
  an unknown ``target_metric`` is rejected
- **FERL evidence**: An unknown ``rule`` in ``predict_ds`` and related methods
  now raises ``ValueError`` instead of silently using Dempster's rule
- **Rule mining**: ``rule_mining`` accepts NumPy arrays as well as DataFrames,
  reading columns by position, so ``RuleMineClassifier`` no longer requires a
  DataFrame
- **RuleFineTuneClassifier**: Builds partitions before mining when none are
  given, and predicts with its second-stage model; both steps previously
  failed. ``fit`` returns the estimator for all rule-mining classifiers
- **Evolutionary Optimization**: Vectorized fitness evaluation for significant speedups
- **Memory Efficiency**: Automatic batching prevents memory overflow on large datasets
- **GPU Utilization**: Seamless GPU/CPU switching based on hardware availability
- Improved API consistency across all modules
- Better error messages and exception handling
- Enhanced visualization capabilities
- Optimized memory usage for large datasets

Fixed
-----
- Unknown classification consequents no longer cause invalid one-hot indices
  during CUDA population evaluation
- Bug in fuzzy set membership calculation
- Memory leak in evolutionary optimization
- Incorrect rule dominance score calculation
- Threading issues in parallel processing

Deprecated
----------
- Old maintenance module (mnt.*) - will be removed in v2.0
- Legacy configuration format - use new YAML format

[1.0.0] - 2023-12-15
====================

Added
-----
- Complete fuzzy logic inference system
- Evolutionary optimization for rule discovery
- Pattern stability analysis tools
- Comprehensive visualization suite
- Type-1 and Type-2 fuzzy set support
- Multi-objective optimization capabilities
- Rule mining and analysis tools
- Model persistence and serialization

Changed
-------
- Complete API redesign for better usability
- Improved performance with vectorized operations
- Enhanced documentation and examples
- Better integration with scikit-learn

Fixed
-----
- Various numerical stability issues
- Compatibility with newer Python versions
- Edge cases in fuzzy set operations

[0.9.0] - 2023-06-20
====================

Added
-----
- Initial pattern stability analysis
- Basic visualization tools
- Evolutionary algorithm optimization
- Type-1 fuzzy sets implementation

Changed
-------
- Refactored core fuzzy logic engine
- Improved rule representation
- Better handling of categorical variables

Fixed
-----
- Issues with rule evaluation
- Memory usage optimization
- Threading synchronization

[0.8.0] - 2023-03-15
====================

Added
-----
- Basic fuzzy classification system
- Rule-based inference engine
- Simple optimization algorithms
- Core fuzzy set operations

Changed
-------
- Initial stable API design
- Basic documentation structure

[0.7.0] - 2023-01-10
====================

Added
-----
- Initial release
- Basic fuzzy logic capabilities
- Simple rule representation
- Experimental optimization

Migration Guides
================

Migrating from 0.9.x to 1.0.0
-----------------------------

**API Changes:**

.. code-block:: python

   # Old way (0.9.x)
   from ex_fuzzy import FuzzyClassifier
   classifier = FuzzyClassifier(rules=10, antecedents=4)
   
   # New way (1.0.x)
   from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
   classifier = BaseFuzzyRulesClassifier(nRules=10, nAnts=4)

**Configuration Changes:**

.. code-block:: python

   # Old way (0.9.x)
   classifier.set_config('tolerance', 0.1)
   
   # New way (1.0.x)
   classifier = BaseFuzzyRulesClassifier(tolerance=0.1)

**Visualization Changes:**

.. code-block:: python

   # Old way (0.9.x)
   classifier.plot_rules()
   
   # New way (1.0.x)
   from ex_fuzzy.eval_tools import FuzzyEvaluator
   evaluator = FuzzyEvaluator(classifier)
   evaluator.eval_fuzzy_model(X_train, y_train, X_test, y_test, plot_rules=True)

Breaking Changes
================

Version 1.0.0
-------------

- **Removed** deprecated `mnt` module
- **Changed** main classifier import path
- **Renamed** several configuration parameters
- **Modified** visualization API for consistency

Version 0.9.0
-------------

- **Changed** rule representation format
- **Removed** experimental features
- **Updated** optimization algorithm interface

Notable Improvements
====================

Performance Improvements
------------------------

**Version 1.0.0:**
- 40% faster rule evaluation
- 60% reduction in memory usage
- 3x improvement in optimization speed
- Better scaling for large datasets

**Version 0.9.0:**
- 25% faster fuzzy set operations
- Improved numerical stability
- Better caching mechanisms

Documentation Improvements
--------------------------

**Version 1.0.0:**
- Complete documentation overhaul
- Interactive examples and tutorials
- Comprehensive API reference
- Best practices guide

**Version 0.9.0:**
- Added user guide
- Basic examples and tutorials
- API documentation improvements

Acknowledgments
===============

We thank all contributors who made these releases possible:

**Version 1.0.0 Contributors:**
- Javier Fumanal Idocin - Lead developer
- Community contributors - Bug reports and feature requests
- Beta testers - Early feedback and testing

**Version 0.9.0 Contributors:**
- Initial development team
- Academic collaborators
- Open source community
