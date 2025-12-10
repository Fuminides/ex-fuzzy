==========================
Fuzzy Regression (API)
==========================

.. currentmodule:: ex_fuzzy.evolutionary_fit_regression

The regression module provides tools for building fuzzy rule-based regression models with support for both crisp numeric outputs and fuzzy set outputs with Mamdani-style inference.

Overview
========

Ex-Fuzzy's regression framework supports two types of consequents:

1. **Crisp Consequents**: Rules output numeric values that are combined using weighted averaging
2. **Fuzzy Consequents**: Rules output fuzzy sets that are aggregated and defuzzified using centroid method

Additionally, rules can operate in two modes:

- **Additive Mode**: All rules contribute to every prediction (default for regression)
- **Sufficient Mode**: Only the rule with highest activation fires (similar to classification)

Main Classes
============

BaseFuzzyRulesRegressor
------------------------

.. autoclass:: BaseFuzzyRulesRegressor
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

   The main regression class that handles training and prediction for fuzzy regression models.

   **Key Parameters:**

   - ``consequent_type``: 'crisp' (numeric outputs) or 'fuzzy' (fuzzy set outputs with defuzzification)
   - ``rule_mode``: 'additive' (all rules contribute) or 'sufficient' (winner-takes-all)
   - ``output_fuzzy_sets``: Precomputed output fuzzy sets for Mamdani inference (optional)
   - ``n_output_linguistic_variables``: Number of output fuzzy sets when evolving them (default: 3)
   - ``universe_points``: Discretization points for Mamdani defuzzification (default: 100)

   **Example:**

   .. code-block:: python

       from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
       from sklearn.datasets import fetch_california_housing
       from sklearn.model_selection import train_test_split
       import pandas as pd

       # Load dataset
       data = fetch_california_housing()
       X = pd.DataFrame(data.data, columns=data.feature_names)
       y = data.target

       # Split data
       X_train, X_test, y_train, y_test = train_test_split(
           X, y, test_size=0.2, random_state=42
       )

       # Crisp regression (numeric outputs)
       regressor = BaseFuzzyRulesRegressor(
           nRules=15,
           nAnts=3,
           consequent_type='crisp',
           rule_mode='additive',
           linguistic_variables=None,
           n_linguistic_variables=3
       )

       regressor.fit(X_train, y_train, n_gen=30, pop_size=50)
       predictions = regressor.predict(X_test)

       # Print rules
       print(regressor)
       regressor.print_rules_regression()

FitRuleBaseRegression
---------------------

.. autoclass:: FitRuleBaseRegression
   :members:
   :undoc-members:
   :show-inheritance:

   Internal optimization problem class for evolutionary fitting of regression rule bases.

Key Methods
===========

Fitting
-------

.. automethod:: BaseFuzzyRulesRegressor.fit

   Train the fuzzy regression model using evolutionary algorithms.

   **Parameters:**

   - ``X``: Training features (numpy array or pandas DataFrame)
   - ``y``: Target values (continuous)
   - ``n_gen``: Number of generations for optimization
   - ``pop_size``: Population size for evolutionary algorithm
   - ``random_state``: Random seed for reproducibility

Prediction
----------

.. automethod:: BaseFuzzyRulesRegressor.predict

   Make predictions on new data.

   Automatically handles both crisp and fuzzy consequent types:

   - **Crisp**: Weighted average of rule consequents
   - **Fuzzy**: Mamdani inference with centroid defuzzification

Rule Display
------------

.. automethod:: BaseFuzzyRulesRegressor.print_rules_regression

   Print regression rules in a readable format.

   Shows antecedents and consequents (numeric values or fuzzy set names).

.. automethod:: BaseFuzzyRulesRegressor.__str__

   String representation using print().

   .. code-block:: python

       regressor = BaseFuzzyRulesRegressor(...)
       regressor.fit(X_train, y_train)
       
       # Simply print
       print(regressor)

Examples
========

Crisp Consequents (Additive Mode)
----------------------------------

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    import numpy as np

    # Generate synthetic data
    X = np.random.rand(200, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

    # Create regressor
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='crisp',
        rule_mode='additive'
    )

    # Train
    regressor.fit(X, y, n_gen=20, pop_size=30)

    # Predict and evaluate
    predictions = regressor.predict(X)
    from sklearn.metrics import r2_score, mean_squared_error
    print(f"R² = {r2_score(y, predictions):.4f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y, predictions)):.4f}")

    # Show rules
    regressor.print_rules_regression()

Fuzzy Consequents (Mamdani Inference)
--------------------------------------

With precomputed output fuzzy sets:

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    from ex_fuzzy import fuzzy_sets as fs
    import numpy as np

    # Create output fuzzy sets
    y_min, y_max = 0, 50
    output_fs = [
        fs.FS('Low', [y_min, y_min, 15, 25], [y_min, y_max]),
        fs.FS('Medium', [20, 25, 25, 30], [y_min, y_max]),
        fs.FS('High', [25, 35, y_max, y_max], [y_min, y_max])
    ]

    # Create regressor with fuzzy consequents
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='fuzzy',
        rule_mode='additive',
        output_fuzzy_sets=output_fs,
        universe_points=100
    )

    regressor.fit(X, y, n_gen=20, pop_size=30)
    predictions = regressor.predict(X)

    # Rules show fuzzy set names
    regressor.print_rules_regression()
    # Output: "IF x1 is LOW THEN output is Low"

With evolved output fuzzy sets:

.. code-block:: python

    # Let the algorithm optimize output fuzzy sets
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='fuzzy',
        rule_mode='additive',
        output_fuzzy_sets=None,  # Evolve them
        n_output_linguistic_variables=5,  # Number of output FSs
        universe_points=100
    )

    regressor.fit(X, y, n_gen=30, pop_size=50)

Sufficient vs Additive Modes
-----------------------------

.. code-block:: python

    # Additive: All rules contribute (weighted average)
    regressor_additive = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        rule_mode='additive'
    )

    # Sufficient: Only strongest rule fires (winner-takes-all)
    regressor_sufficient = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        rule_mode='sufficient'
    )

Technical Details
=================

Crisp Inference
---------------

For crisp consequents, the output is computed as:

.. math::

    y = \\frac{\\sum_{i=1}^{R} \\mu_i(x) \\cdot c_i}{\\sum_{i=1}^{R} \\mu_i(x)}

where:
- :math:`R` is the number of rules
- :math:`\\mu_i(x)` is the firing strength of rule :math:`i`
- :math:`c_i` is the crisp consequent of rule :math:`i`

Mamdani Inference
-----------------

For fuzzy consequents:

1. **Clipping**: Each rule's output fuzzy set is clipped by its firing strength:

   .. math::

       \\mu'_i(z) = \\min(\\mu_i(x), \\mu_{C_i}(z))

2. **Aggregation**: All clipped fuzzy sets are combined using MAX:

   .. math::

       \\mu_{out}(z) = \\max_{i=1}^{R} \\mu'_i(z)

3. **Defuzzification**: Centroid method:

   .. math::

       y = \\frac{\\int z \\cdot \\mu_{out}(z) \\, dz}{\\int \\mu_{out}(z) \\, dz}

   Discretized as:

   .. math::

       y = \\frac{\\sum_{j=1}^{N} z_j \\cdot \\mu_{out}(z_j)}{\\sum_{j=1}^{N} \\mu_{out}(z_j)}

Gene Encoding
-------------

The evolutionary algorithm encodes rules as genes:

**Crisp consequents:**

.. code-block:: text

    [antecedents | antecedent_params | input_MFs | consequent_values]

**Fuzzy consequents:**

.. code-block:: text

    [antecedents | antecedent_params | input_MFs | consequent_indices | output_MFs?]

where ``output_MFs`` are only included if evolving output fuzzy sets.

See Also
========

- :doc:`../user-guide/regression` - User guide for regression
- :doc:`../examples/regression` - Regression examples
- :doc:`evolutionary_fit` - Classification module
- :doc:`fuzzy_sets` - Fuzzy set definitions

References
==========

.. [1] Takagi, T., & Sugeno, M. (1985). Fuzzy identification of systems and its applications to modeling and control.
.. [2] Mamdani, E. H., & Assilian, S. (1975). An experiment in linguistic synthesis with a fuzzy logic controller.
