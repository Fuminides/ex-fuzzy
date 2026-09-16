=========
Changelog
=========

This document tracks all notable changes to Ex-Fuzzy.

The format is based on `Keep a Changelog <https://keepachangelog.com/en/1.0.0/>`_,
and this project adheres to `Semantic Versioning <https://semver.org/spec/v2.0.0.html>`_.

[Unreleased]
============

[3.2.0] - 2026-09-16
====================

Changed
-------
- **Rule text without statistics loads**: the ``WITH`` clause of a printed
  rule lists whichever of ``DS``, ``ACC`` and ``WGHT`` the rule has, and
  ``load_fuzzy_rules`` accepts any subset of them, a missing ``WITH`` clause,
  and ignores a ``THEN`` clause, so rules printed before evaluation round-trip
- ``RuleMineClassifier`` and ``RuleFineTuneClassifier`` take ``n_gen``,
  ``pop_size``, ``patience`` and ``random_state`` in their constructors,
  with the same fit-time overrides as ``BaseFuzzyRulesClassifier``
- **Lazy package import**: ``import ex_fuzzy`` no longer imports every
  submodule; submodules and the top-level classes load on first access, which
  takes the import from about two seconds to a few milliseconds
- **Docstrings** follow the Google style throughout, with triple double quotes
- **Demos** import through the package without path tricks; every notebook and script was executed against this release
- **Search settings are constructor parameters**: ``n_gen``, ``pop_size``,
  ``patience``, ``min_delta``, ``random_state``, ``var_prob``, ``sbx_eta``,
  ``mutation_eta`` and ``tournament_size`` can be given to
  ``BaseFuzzyRulesClassifier``, so ``clone`` and ``GridSearchCV`` see them.
  Passing them to ``fit`` overrides them for that fit only, as before
- ``ds_mode`` accepts the names ``'dominance'``, ``'unweighted'`` and
  ``'optimized'`` besides ``0``, ``1`` and ``2``, and rejects anything else in
  the classifier, ``FitRuleBase`` and ``MasterRuleBase``
- ``explainable_predict`` returns an ``ExplainedPrediction`` named tuple with
  the fields ``prediction``, ``winning_rule``, ``association_degree`` and
  ``confidence_interval``; it still unpacks as a four-tuple
- **Conformal prediction uses the fitted labels**: calibration encodes them
  through ``classes_``, and prediction sets, p-values, rule contributions,
  calibration info and coverage reports are keyed by label
- ``RuleMineClassifier`` passes its ``nAnts`` to the inner classifier
- The ``runner`` parameter documents that threads disable the fit-local
  caches, so serial fits are usually faster
- **Rule mining**: the itemset search prunes with the Apriori property
  (min t-norm support never grows when an item is added) and the confidence
  and lift pruning computes the memberships once per variable. The candidate
  rules, their order and the pruned rule bases are identical
- **Packaging**: metadata moved to ``pyproject.toml`` with
  ``requires-python >= 3.10``, the AGPL license, and classifiers for Python
  3.10 to 3.13; ``setup.py`` only builds the optional native FERL extension.
  CI tests Python 3.10 to 3.13 and runs the EvoX backend tests on CPU
- **Import time**: matplotlib is imported when a plot is drawn, not when the
  package is imported
- **Warnings and errors**: the classifier reports the backend fallback,
  antecedent clamping and unsupported checkpoints with ``warnings.warn``
  instead of printing; incoherent interval fuzzy sets, missing dominance
  scores, empty rule lists and temporal sets without a fixed time raise
  ``ValueError`` instead of ``AssertionError``; ``FuzzyEvaluator.get_metric``
  raises instead of returning an error string
- **Names**: ``MasterRuleBase.compute_firing_strengths`` and
  ``BaseFuzzyRulesClassifier.reparametrize_loss`` are the spelled names;
  the former misspellings remain as deprecated aliases
- **Docs**: the duplicate Getting Started page was removed, the step-by-step
  pages form a Tutorial section, the multiprocessing advice was replaced by
  what speeds up a fit, and the README points at the documentation site
- **Package layout**: the package now lives in ``ex_fuzzy/`` at the
  repository root instead of ``ex_fuzzy/ex_fuzzy/``, and the outer re-export
  shim is gone. Modules import each other with relative imports only, without
  ``try/except ImportError`` fallbacks, and the tests, benchmarks and demos
  import through the package. ``FUZZY_SETS`` uses the default enum equality
  and hashing again, since every module now exists once per process.
  Reinstall with ``pip install -e .`` after updating a checkout
- **Type-1 rule-base inference** computes the centroid of every sample with
  one matrix-vector product instead of a per-sample loop. Outputs agree with
  the former computation to about 1e-12 relative
- **BaseFuzzyRulesClassifier is a scikit-learn estimator**: the constructor
  keeps every argument under its own name, so ``get_params``, ``clone``,
  ``repr``, ``cross_val_score`` and ``Pipeline`` work. ``fit`` returns the
  classifier and sets ``classes_`` and ``n_features_in_``. Labels are encoded
  as consequent indexes for the search and ``predict`` returns the labels the
  classifier was fitted with, so ``score`` works with string labels and
  integer labels need not be ``0..n_classes-1``. Samples with no firing rule
  predict ``-1`` for numeric labels and ``'Unknown'`` otherwise. Classifiers
  built from precomputed rules still predict consequent indexes. The
  ``backend`` parameter also accepts a backend instance
- **Prediction computes memberships and firing once**: a ``MasterRuleBase``
  evaluates the antecedent memberships once for all its rule bases instead of
  once per rule base, and the winning-rule prediction no longer computes the
  firing strengths twice. Predictions are unchanged; the winning-rule
  predictions of a rule base are now integer arrays
- **Rule evaluation computes firing once**: the ``evalRuleBase`` weight,
  accuracy and metric methods share one firing computation per call instead
  of recomputing it for support, confidence and dominance. Results are
  unchanged
- **Rule identity**: ``RuleSimple`` equality and hashing now depend on the
  antecedents, consequent and modifiers only, not on the score, weight or
  accuracy attached later. Rule-base construction therefore removes rules
  with equal antecedents even when their weights differ, keeping the first
  one. This changes ``ds_mode=2`` searches whose candidates repeat an
  antecedent pattern in one class; the array evaluator applies the same rule
- **RuleMineClassifier and RuleFineTuneClassifier are scikit-learn
  estimators**: constructor arguments are stored under their own names and the
  inner classifiers are built when ``fit`` is called, so ``clone`` and
  cross-validation work. ``fit`` sets ``classes_`` and ``n_features_in_``, and
  ``predict`` returns the fitted labels. The ``fl_classifier``,
  ``fl_classifier1`` and ``fl_classifier2`` attributes remain available after
  ``fit``
- **ConformalFuzzyClassifier** takes the wrapped classifier as ``estimator``
  (``clf_or_nRules`` is still accepted), which ``get_params`` and ``clone``
  see; ``clf`` remains as an alias. ``fit`` now also fits a wrapped classifier
  that has not been fitted yet
- **TemporalFuzzyRulesClassifier** gained ``predict(X, time_moments)``,
  and its ``fit`` returns the classifier

Fixed
-----
- Missing or infinite feature values were silently assigned class 0, because
  NaN firing strengths made ``argmax`` pick the first rule, and gave NaN
  probabilities. ``fit`` and the prediction methods now reject them with a
  ``ValueError`` naming the columns
- Conformal calibration crashed with string labels and mis-indexed integer
  labels that do not start at 0
- Categorical fuzzy sets now define ``domain`` and ``membership_parameters``
  as ``None``, so ``fuzzyVariable.domain()`` no longer raises for a
  categorical variable, and membership computation no longer relies on a
  swallowed exception to skip the domain clipping
- ``load_fuzzy_rules`` parses the ``IF``, ``IS``, ``AND`` and ``WITH``
  keywords as whole words, so variable and label names containing them (for
  example DISTANCE or WITHIN) load correctly, and rule text without
  ``Rules for`` headers loads as a single rule base instead of failing
- ``multiclass_mine_rulebase`` pruned rules by comparing the labels with the
  rule base indexes, which failed for string labels and silently mismatched
  integer labels that do not start at 0; the labels are now encoded in the
  order the rule bases are built
- ``RuleMineClassifier`` mined its candidate rules on freshly built
  partitions and ignored the ``linguistic_variables`` it was given; it now
  mines on the partitions it predicts with
- ``BaseFuzzyRulesClassifier.score`` returned 0 for string labels because
  ``predict`` returned class indexes
- General Type-2 rule bases with ``ds_mode=2`` failed to predict because the
  rule weights were not broadcast over the interval axis

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
