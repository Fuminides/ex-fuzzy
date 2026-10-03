===================
Regression Examples
===================

Ex-Fuzzy learns interpretable Type-1 fuzzy rules for continuous targets with
:class:`ex_fuzzy.BaseFuzzyRulesRegressor`. The estimator provides the familiar
``fit``, ``predict``, and ``score`` methods; ``score`` returns :math:`R^2`.

Train/Test Workflow
===================

This example uses crisp zero-order Takagi-Sugeno consequents, where each rule
learns a numeric output:

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesRegressor
   from sklearn.datasets import make_friedman1
   from sklearn.model_selection import train_test_split

   X, y = make_friedman1(
       n_samples=500,
       n_features=5,
       noise=0.5,
       random_state=0,
   )
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0
   )

   regressor = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       consequent_type="crisp",
       backend="pymoo",
   )
   regressor.fit(X_train, y_train, n_gen=50, pop_size=50)

   predictions = regressor.predict(X_test)
   print(f"Test R2: {regressor.score(X_test, y_test):.3f}")
   regressor.print_rules()

Predictions are the firing-strength-weighted average of active rule
consequents. If no rule clears the tolerance, the model returns the training
target mean.

Mamdani Consequents
===================

Use ``consequent_type="fuzzy"`` when rules should name linguistic output sets.
The optimizer learns the trapezoidal output sets as part of the chromosome,
and predictions are centroid-defuzzified values.

.. code-block:: python

   mamdani = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       consequent_type="fuzzy",
       n_output_lvs=4,
       n_universe_points=101,
       backend="pymoo",
   )
   mamdani.fit(X_train, y_train, n_gen=50, pop_size=50)
   mamdani.print_rules()
   # Rule 1: IF x0 IS Low AND x2 IS High THEN output IS Output_3

GPU-Accelerated EvoX
====================

Install the optional EvoX dependencies and select the backend on the same
estimator:

.. code-block:: bash

   python -m pip install "ex-fuzzy[evox]"

.. code-block:: python

   gpu_regressor = BaseFuzzyRulesRegressor(
       nRules=30,
       nAnts=4,
       consequent_type="crisp",  # "fuzzy" is accelerated too
       backend="evox",
       verbose=True,
   )
   gpu_regressor.fit(X_train, y_train, n_gen=50, pop_size=100)

   print(gpu_regressor.optimization_device_)  # "cuda" or "cpu"
   print(gpu_regressor.gpu_accelerated_)      # True only on CUDA

EvoX evolves the population while Ex-Fuzzy evaluates the complete regression
objective in PyTorch. CUDA is selected automatically when available; otherwise
the same implementation runs on CPU. Population and sample batching adapts to
available memory.

Rule Modes
==========

``rule_mode="additive"`` lets every active rule contribute to the prediction.
``rule_mode="sufficient"`` keeps only the strongest rule for each sample:

.. code-block:: python

   winner_takes_all = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       rule_mode="sufficient",
       tolerance=0.05,
       backend="evox",
   )

See :doc:`../user-guide/regression` for modeling guidance and
:doc:`../api/regression` for the complete parameter reference.
