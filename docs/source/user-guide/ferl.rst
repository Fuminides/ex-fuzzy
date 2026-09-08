=====================================
Fast Evidential Rule Learning (FERL)
=====================================

FERL is Ex-Fuzzy's greedy fuzzy rule-tree classifier. It learns interpretable
rules and derives evidential predictions directly from their fuzzy firing
strengths. A fitted model can return:

- ordinary class labels and probabilities;
- belief and plausibility for each class;
- a scalar ignorance mass for each sample; and
- native set-valued predictions for abstention or cautious decisions.

FERL is implemented inside Ex-Fuzzy and has no dependency on a separate FERL
or fuzzy-tree repository.

Quick start
===========

The default compact configuration creates three quantile-based linguistic
terms per feature and uses additive soft rule voting for point predictions.

.. code-block:: python

   from ex_fuzzy import FERL
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0, stratify=y
   )

   model = FERL(max_rules=15, random_state=0)
   model.fit(X_train, y_train)

   labels = model.predict(X_test)
   probabilities = model.predict_proba(X_test)

Optional compiled backend
=========================

``FERL(backend="cython")`` uses a native additive-vote scoring kernel adapted
from ``fgrt_fast`` in the ``fuzzy_greedy_tree`` repository. That implementation
uses Cython compiled to **C**, rather than C++. Ex-Fuzzy includes its own source;
the sibling repository is not a runtime dependency.

Build from the Ex-Fuzzy source directory with a C compiler and the development
headers for your Python interpreter installed:

.. code-block:: bash

   python -m pip install Cython numpy
   EX_FUZZY_BUILD_FERL=1 python -m pip install --no-build-isolation -e .

On PowerShell, set ``$env:EX_FUZZY_BUILD_FERL = "1"`` before running the second
Python command. Ordinary installations do not build the extension and do not
require Cython or a compiler.

.. code-block:: python

   model = FERL(backend="cython", max_rules=15, random_state=0)
   model.fit(X_train, y_train)
   probabilities = model.predict_proba(X_test)

The initial compiled backend accelerates candidate vote simulation for
consistent CCI with soft voting, for both fixed and learned splits. It avoids
allocating a sample-by-class vote matrix for every candidate. Partitioning,
normalization, split tie-breaking, pruning, prediction and evidence outputs
retain the Python implementation. Purity and legacy scoring also retain the
Python path. This is a port of the vote-scoring kernel, not the complete set
of ``fgrt_fast`` optimizations; end-to-end gains depend on the workload.

``backend="python"`` remains the default. Selecting ``"cython"`` without a
built extension raises an installation error at ``fit()``. The backend flag
works with estimator cloning and ``set_params``; all existing calls remain valid.

Evidential predictions
======================

Every activated rule supplies a Dempster--Shafer mass. The committed mass is
distributed across classes using the rule consequent; the remaining mass is
assigned to the full class frame as ignorance.

.. code-block:: python

   betp, belief, plausibility, ignorance = model.predict_credal(X_test)
   prediction_sets = model.predict_set(X_test)

``betp`` is the pignistic probability used for a point decision. ``belief``
and ``plausibility`` bound the support for each class. ``prediction_sets`` is a
boolean array with one column per entry in ``model.classes_``; FERL keeps each
class whose plausibility is at least the largest class belief.

FERL prediction sets are native evidential outputs and require no held-out
calibration set. They do not provide a finite-sample coverage guarantee. Use
:class:`ex_fuzzy.ConformalFuzzyClassifier` when guaranteed marginal coverage is
the primary requirement.

Compact and learned-split models
================================

``split_mode="fixed"`` is the default. It searches human-readable fuzzy terms
such as low, medium, and high. This usually produces the most compact model.

``split_mode="learned"`` learns a data-driven split location and represents it
with two soft ramps. With ``learned_width="bootstrap"``, repeated bootstrap cut
estimates determine the ramp width: stable cuts become sharper and uncertain
cuts remain wider.

.. code-block:: python

   deep_model = FERL(
       split_mode="learned",
       max_rules=100,
       max_depth=12,
       min_improvement=0.0,
       learned_n_boot=25,
       random_state=0,
   ).fit(X_train, y_train, patience=16)

   # Learned-split FERL uses leaves-only evidence by default here.
   betp, belief, plausibility, ignorance = deep_model.predict_credal(X_test)

``predict_credal`` and ``predict_set`` automatically combine only leaves for a
learned-split model, avoiding repeated evidence from strongly nested internal
rules. Pass ``leaves_only`` explicitly to override that choice. The lower-level
``predict_ds`` method exposes the same evidential calculation with all options.

Partition choices
=================

FERL constructs Type-1 fuzzy partitions during ``fit`` unless custom
``fuzzy_partitions`` are supplied.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Setting
     - Behavior
   * - ``partition="quantile"``
     - Unsupervised, fixed-count partitions controlled by ``n_partitions``.
   * - ``partition="mdlp"``
     - Supervised Fayyad--Irani MDLP cuts converted to overlapping trapezoids;
       overlap is controlled by ``overlap_frac``.
   * - ``fuzzy_partitions=...``
     - User-supplied Ex-Fuzzy variables, one per input feature.

Missing features
================

Prediction methods accept an ``observed_mask`` with the same shape as ``X``.
An unobserved condition contributes uniform membership instead of requiring
imputation.

.. code-block:: python

   import numpy as np

   observed = np.ones_like(X_test, dtype=bool)
   observed[:, 0] = False
   probabilities = model.predict_proba(X_test, observed_mask=observed)

Inspecting the model
====================

Use ``model.print_tree()`` for a readable hierarchy,
``model.get_tree_stats()`` for structural statistics, and ``model.n_rules()``
for the learned rule count. ``model.fuzzy_partitions_`` contains the fitted
linguistic variables and ``model.node_activation_matrix(X)`` exposes the
per-rule firing matrix and consequents.

See :doc:`../examples/ferl` for a complete example.
