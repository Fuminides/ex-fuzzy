================
Fuzzy Regression
================

The :mod:`ex_fuzzy.evolutionary_fit_regression` module learns interpretable
Type-1 fuzzy rules for continuous targets. During optimization it precomputes
the fixed input-partition memberships and scores candidate rule bases through a
vectorized NumPy inference path with PyMoo, or a batched PyTorch path with the
optional EvoX backend.

Two consequent styles are available, selected with ``consequent_type``:

``crisp`` (default)
   Zero-order Takagi-Sugeno. Each rule carries a number, and a prediction is
   the firing-strength-weighted average of those numbers. Best numeric
   resolution.

``fuzzy``
   Mamdani. Each rule names an output fuzzy set whose trapezoid is evolved
   alongside the rules; predictions are the centroid of the clipped and
   max-aggregated consequents. Rules read fully linguistically, at some cost in
   resolution.

``rule_mode`` controls how many rules speak per sample. ``additive`` (default)
lets every rule contribute. ``sufficient`` keeps only each sample's strongest
rule, and falls back to the training-target mean when even that rule fires at
or below ``tolerance`` -- giving piecewise-constant, winner-takes-all output.

.. currentmodule:: ex_fuzzy.evolutionary_fit_regression

BaseFuzzyRulesRegressor
=======================

.. autoclass:: BaseFuzzyRulesRegressor
   :members:
   :show-inheritance:

FitRuleBaseRegression
=====================

.. autoclass:: FitRuleBaseRegression
   :members:

RuleBaseT1Regression
====================

.. autoclass:: RuleBaseT1Regression
   :members:

RuleBaseT1MamdaniRegression
===========================

.. autoclass:: RuleBaseT1MamdaniRegression
   :members:

Example
-------

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesRegressor
   from sklearn.datasets import make_regression
   from sklearn.model_selection import train_test_split

   X, y = make_regression(n_samples=200, n_features=4, random_state=0)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0
   )

   regressor = BaseFuzzyRulesRegressor(nRules=20, nAnts=3)
   regressor.fit(X_train, y_train, n_gen=50, pop_size=50)

   predictions = regressor.predict(X_test)
   print(regressor.score(X_test, y_test))
   regressor.print_rules()
   # Rule 1: IF x0 IS Low AND x2 IS High THEN output = 41.8203

GPU optimization
================

Install ``ex-fuzzy[evox]`` and select the EvoX backend to evaluate complete
populations on CUDA. If CUDA is unavailable, EvoX runs the same PyTorch path on
the CPU.

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=30, nAnts=4, backend="evox", verbose=True
   )
   regressor.fit(X_train, y_train, n_gen=50, pop_size=100)
   print(regressor.optimization_device_)
   print(regressor.gpu_accelerated_)

Linguistic consequents
======================

Switch to Mamdani inference when a rule should name an output *label* rather
than a number:

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       consequent_type="fuzzy",
       n_output_lvs=4,          # how many output sets to evolve
       n_universe_points=101,   # centroid integration grid
   )
   regressor.fit(X_train, y_train, n_gen=50, pop_size=50)
   regressor.print_rules()
   # Rule 1: IF x0 IS Low AND x2 IS High THEN output IS Output_3

Every chromosome decodes to a valid trapezoid because each output set's four
breakpoints are sorted before being scaled into the target range.

Winner-takes-all rules
======================

.. code-block:: python

   regressor = BaseFuzzyRulesRegressor(
       nRules=20, nAnts=3, rule_mode="sufficient", tolerance=0.05
   )

With ``sufficient`` only the strongest rule fires per sample, so a crisp-
consequent model becomes piecewise constant: every prediction is one rule's
consequent, or the target mean where nothing clears the tolerance.

The genetic search optimizes training-set :math:`R^2`. To estimate
generalization with cross-validation, evaluate the complete estimator through
scikit-learn's ``cross_val_score`` or ``cross_validate``; the estimator must be
refitted independently inside each fold.
