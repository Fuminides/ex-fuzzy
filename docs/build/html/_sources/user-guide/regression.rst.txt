==================
Fuzzy Regression
==================

This guide covers fuzzy regression with Ex-Fuzzy, including crisp and fuzzy consequents, rule modes, and practical examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

Introduction
============

Fuzzy regression models use fuzzy rules to predict continuous values. Unlike traditional regression that outputs a single crisp value, fuzzy regression can leverage linguistic interpretability while maintaining prediction accuracy.

Ex-Fuzzy supports two types of regression consequents:

1. **Crisp Consequents**: Rules output numeric values (like traditional regression)
2. **Fuzzy Consequents**: Rules output fuzzy sets that are defuzzified to produce predictions

Basic Workflow
==============

1. Prepare your data (features X and target y)
2. Create a ``BaseFuzzyRulesRegressor`` instance
3. Train the model using ``.fit()``
4. Make predictions with ``.predict()``
5. Evaluate and visualize results

Quick Start Example
===================

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    import pandas as pd

    # Load dataset
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Create and train regressor
    regressor = BaseFuzzyRulesRegressor(
        nRules=15,
        nAnts=3,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3
    )

    regressor.fit(X_train, y_train, n_gen=30, pop_size=50)

    # Predict and evaluate
    y_pred = regressor.predict(X_test)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.4f}")

    # Display rules
    print(regressor)

Consequent Types
================

Crisp Consequents
-----------------

Crisp consequents output numeric values that are combined using weighted averaging based on rule firing strengths.

**Advantages:**
- Simple and efficient
- Easy to interpret
- Similar to traditional regression

**Formula:**

.. math::

    y = \\frac{\\sum_{i=1}^{R} \\mu_i(x) \\cdot c_i}{\\sum_{i=1}^{R} \\mu_i(x)}

**Example:**

.. code-block:: python

    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='crisp',  # Numeric outputs
        rule_mode='additive'
    )

    regressor.fit(X_train, y_train, n_gen=20, pop_size=30)

    # Rules show numeric consequents
    regressor.print_rules_regression()
    # Output: "IF x1 is LOW AND x2 is HIGH THEN output = 25.73"

Fuzzy Consequents (Mamdani Inference)
--------------------------------------

Fuzzy consequents output fuzzy sets that are aggregated and defuzzified using the centroid method.

**Advantages:**
- More linguistically interpretable
- Captures uncertainty in outputs
- Follows classical Mamdani inference

**Process:**
1. Each rule outputs a fuzzy set
2. Rule firing strength clips the output fuzzy set
3. All clipped sets are aggregated using MAX operation
4. Centroid defuzzification produces the final crisp value

**Example with precomputed fuzzy sets:**

.. code-block:: python

    from ex_fuzzy import fuzzy_sets as fs

    # Define output fuzzy sets
    y_min, y_max = 0, 50
    output_fs = [
        fs.FS('Very_Low', [y_min, y_min, 5, 10], [y_min, y_max]),
        fs.FS('Low', [5, 12, 15, 20], [y_min, y_max]),
        fs.FS('Medium', [15, 22, 28, 32], [y_min, y_max]),
        fs.FS('High', [28, 35, 40, 45], [y_min, y_max]),
        fs.FS('Very_High', [40, 45, y_max, y_max], [y_min, y_max])
    ]

    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='fuzzy',
        rule_mode='additive',
        output_fuzzy_sets=output_fs,
        universe_points=100  # Discretization for defuzzification
    )

    regressor.fit(X_train, y_train, n_gen=20, pop_size=30)

    # Rules show fuzzy set names
    regressor.print_rules_regression()
    # Output: "IF x1 is LOW AND x2 is HIGH THEN output is Very_High"

**Example with evolved fuzzy sets:**

.. code-block:: python

    # Let the algorithm learn output fuzzy sets
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='fuzzy',
        rule_mode='additive',
        output_fuzzy_sets=None,  # Will be evolved
        n_output_linguistic_variables=5,  # Number of output FSs
        universe_points=100
    )

    regressor.fit(X_train, y_train, n_gen=30, pop_size=50)

Rule Modes
==========

Additive Mode (Default)
------------------------

All rules contribute to every prediction. The final output is a weighted combination of all rule outputs.

**When to use:**
- Default choice for regression
- When multiple rules should influence the prediction
- Similar to ensemble methods

**Example:**

.. code-block:: python

    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        rule_mode='additive'  # All rules contribute
    )

Sufficient Mode
---------------

Only the rule with the highest activation fires. This is a winner-takes-all approach.

**When to use:**
- When you want crisp decision boundaries
- Similar to classification behavior
- When rules represent mutually exclusive regions

**Example:**

.. code-block:: python

    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        rule_mode='sufficient'  # Only strongest rule fires
    )

**Comparison:**

.. code-block:: python

    import numpy as np
    from sklearn.metrics import r2_score

    X = np.random.rand(200, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

    # Additive mode
    reg_add = BaseFuzzyRulesRegressor(nRules=10, nAnts=2, rule_mode='additive')
    reg_add.fit(X, y, n_gen=20, pop_size=30)
    y_pred_add = reg_add.predict(X)

    # Sufficient mode
    reg_suff = BaseFuzzyRulesRegressor(nRules=10, nAnts=2, rule_mode='sufficient')
    reg_suff.fit(X, y, n_gen=20, pop_size=30)
    y_pred_suff = reg_suff.predict(X)

    print(f"Additive R²: {r2_score(y, y_pred_add):.4f}")
    print(f"Sufficient R²: {r2_score(y, y_pred_suff):.4f}")

Complete Example: California Housing
=====================================

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load and prepare data
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    # Create and train regressor
    regressor = BaseFuzzyRulesRegressor(
        nRules=15,
        nAnts=3,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3,
        tolerance=0.01
    )

    print("Training fuzzy regressor...")
    regressor.fit(
        X_train_scaled, 
        y_train,
        n_gen=30,
        pop_size=50,
        random_state=42
    )

    # Evaluate on test set
    y_pred = regressor.predict(X_test_scaled)

    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"\\nTest Set Performance:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")

    # Display learned rules
    print("\\nLearned Rules:")
    regressor.print_rules_regression()

    # Visualize predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], 
             [y_test.min(), y_test.max()], 
             'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Fuzzy Regression: Predicted vs True Values')
    plt.grid(True, alpha=0.3)
    plt.show()

More Features
=================

Precomputed Linguistic Variables
---------------------------------

You can provide custom fuzzy partitions for input features:

.. code-block:: python

    from ex_fuzzy import fuzzy_sets as fs

    # Create custom linguistic variables
    x1_lvs = [
        fs.FS('Very_Low', [0, 0, 2, 3], [0, 10]),
        fs.FS('Low', [2, 3, 4, 5], [0, 10]),
        fs.FS('Medium', [4, 5, 6, 7], [0, 10]),
        fs.FS('High', [6, 7, 8, 9], [0, 10]),
        fs.FS('Very_High', [8, 9, 10, 10], [0, 10])
    ]
    x1_var = fs.fuzzyVariable('x1', x1_lvs)

    x2_lvs = [
        fs.FS('Low', [0, 0, 3, 5], [0, 10]),
        fs.FS('Medium', [3, 5, 5, 7], [0, 10]),
        fs.FS('High', [5, 7, 10, 10], [0, 10])
    ]
    x2_var = fs.fuzzyVariable('x2', x2_lvs)

    # Use precomputed partitions
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        linguistic_variables=[x1_var, x2_var],  # Custom partitions
        consequent_type='crisp',
        rule_mode='additive'
    )

GPU Acceleration (EvoX Backend)
--------------------------------

For large datasets, use GPU acceleration:

.. code-block:: python

    regressor = BaseFuzzyRulesRegressor(
        nRules=20,
        nAnts=4,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3
    )

    # Use EvoX backend for GPU acceleration
    regressor.fit(
        X_train, 
        y_train,
        n_gen=50,
        pop_size=100,
        backend='evox',  # GPU-accelerated
        device='cuda'
    )

See :doc:`../evox_backend` for more details on GPU acceleration.

Model Persistence
-----------------

Persist fuzzy variables and rule text for reporting or reuse:

.. code-block:: python

    from ex_fuzzy import persistence

    # Train and capture artifacts
    regressor.fit(X_train, y_train, n_gen=30, pop_size=50)
    rules_text = regressor.rule_base.print_rules_regression(
        return_rules=True,
        output_name="output",
    )
    variables_text = persistence.save_fuzzy_variables(regressor.lvs)

    # Store rules_text and variables_text as needed (files, database, etc.)



See Also
========

- :doc:`../api/regression` - API reference
- :doc:`../examples/regression` - More examples
- :doc:`../evox_backend` - GPU acceleration guide
- :doc:`validation-visualization` - Plotting fuzzy rules and partitions

Next Steps
==========

- Try the :doc:`../examples/regression` notebook
- Explore :doc:`validation-visualization` for rule visualization
- Check out :doc:`../evox_backend` for GPU acceleration
