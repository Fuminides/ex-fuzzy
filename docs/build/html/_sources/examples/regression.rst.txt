====================
Regression Examples
====================

This page provides comprehensive examples for fuzzy regression using Ex-Fuzzy.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Regression
================

Simple Crisp Regression
-----------------------

A minimal example to get started with fuzzy regression:

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    import numpy as np
    from sklearn.metrics import r2_score, mean_squared_error

    # Generate synthetic data
    np.random.seed(42)
    X = np.random.rand(200, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.5

    # Create and train regressor
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3
    )

    regressor.fit(X, y, n_gen=20, pop_size=30, random_state=42)

    # Evaluate
    predictions = regressor.predict(X)
    print(f"R² = {r2_score(y, predictions):.4f}")
    print(f"RMSE = {np.sqrt(mean_squared_error(y, predictions)):.4f}")

    # Display rules
    print("\\nLearned Fuzzy Rules:")
    regressor.print_rules_regression()

Output:

.. code-block:: text

    R² = 0.9823
    RMSE = 0.5124

    Learned Fuzzy Rules:
    ================================================================================
    Rule 1: IF x1 is LOW AND x2 is LOW THEN output = 5.23
    Rule 2: IF x1 is HIGH AND x2 is LOW THEN output = 9.87
    Rule 3: IF x1 is LOW AND x2 is HIGH THEN output = 8.45
    ...

California Housing Dataset
==========================

Complete Example with Real Data
--------------------------------

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # Load dataset
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = data.target

    print(f"Dataset shape: {X.shape}")
    print(f"Features: {list(X.columns)}")
    print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

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

    # Create regressor
    regressor = BaseFuzzyRulesRegressor(
        nRules=15,
        nAnts=3,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3,
        tolerance=0.01
    )

    # Train
    print("\\nTraining fuzzy regressor...")
    regressor.fit(
        X_train_scaled,
        y_train,
        n_gen=30,
        pop_size=50,
        random_state=42
    )

    # Evaluate on test set
    y_pred_train = regressor.predict(X_train_scaled)
    y_pred_test = regressor.predict(X_test_scaled)

    # Metrics
    print("\\nPerformance Metrics:")
    print("="*60)
    print("Training Set:")
    print(f"  R² Score: {r2_score(y_train, y_pred_train):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_train, y_pred_train)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_train, y_pred_train):.4f}")

    print("\\nTest Set:")
    print(f"  R² Score: {r2_score(y_test, y_pred_test):.4f}")
    print(f"  RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_test)):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test, y_pred_test):.4f}")

    # Display rules
    print("\\nLearned Fuzzy Rules:")
    print("="*60)
    regressor.print_rules_regression()

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Predicted vs True
    axes[0].scatter(y_test, y_pred_test, alpha=0.5)
    axes[0].plot([y_test.min(), y_test.max()],
                 [y_test.min(), y_test.max()],
                 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predictions')
    axes[0].set_title('Predicted vs True Values')
    axes[0].grid(True, alpha=0.3)

    # Residuals
    residuals = y_test - y_pred_test
    axes[1].scatter(y_pred_test, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Mamdani Fuzzy Inference
========================

Fuzzy Consequents with Precomputed Sets
----------------------------------------

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    from ex_fuzzy import fuzzy_sets as fs
    import numpy as np

    # Generate data
    X = np.random.rand(200, 2) * 10
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 2

    # Define output fuzzy sets
    y_min, y_max = y.min(), y.max()
    output_fs = [
        fs.FS('Very_Low', [y_min, y_min, 10, 15], [y_min, y_max]),
        fs.FS('Low', [10, 15, 20, 25], [y_min, y_max]),
        fs.FS('Medium', [20, 25, 30, 35], [y_min, y_max]),
        fs.FS('High', [30, 35, 40, 45], [y_min, y_max]),
        fs.FS('Very_High', [40, 45, y_max, y_max], [y_min, y_max])
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

    # Rules now show fuzzy set names
    regressor.print_rules_regression()

Output:

.. code-block:: text

    ================================================================================
    Rule 1: IF x1 is LOW AND x2 is LOW THEN output is Low
    Rule 2: IF x1 is HIGH AND x2 is LOW THEN output is Medium
    Rule 3: IF x1 is LOW AND x2 is HIGH THEN output is High
    ...

Evolved Output Fuzzy Sets
--------------------------

Let the algorithm learn the output fuzzy sets:

.. code-block:: python

    # Evolve output fuzzy sets during optimization
    regressor = BaseFuzzyRulesRegressor(
        nRules=10,
        nAnts=2,
        consequent_type='fuzzy',
        rule_mode='additive',
        output_fuzzy_sets=None,  # Will be evolved
        n_output_linguistic_variables=5,  # Number of output FSs
        universe_points=100
    )

    regressor.fit(X, y, n_gen=30, pop_size=50)

    # Access evolved output fuzzy sets
    print("\\nEvolved Output Fuzzy Sets:")
    for i, fs in enumerate(regressor.rule_base.output_fuzzy_sets):
        print(f"  {i}: {fs.name} - {fs.membership_parameters}")

Comparing Rule Modes
====================

Additive vs Sufficient
-----------------------

.. code-block:: python

    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor
    import numpy as np
    from sklearn.metrics import r2_score
    import matplotlib.pyplot as plt

    # Generate data with piecewise structure
    X = np.random.rand(300, 2) * 10
    y = np.where(X[:, 0] < 5,
                 2 * X[:, 0] + X[:, 1],
                 5 * X[:, 0] - 2 * X[:, 1])
    y += np.random.randn(300) * 0.5

    # Additive mode
    reg_add = BaseFuzzyRulesRegressor(
        nRules=10, nAnts=2, rule_mode='additive'
    )
    reg_add.fit(X, y, n_gen=20, pop_size=30, random_state=42)
    y_pred_add = reg_add.predict(X)

    # Sufficient mode
    reg_suff = BaseFuzzyRulesRegressor(
        nRules=10, nAnts=2, rule_mode='sufficient'
    )
    reg_suff.fit(X, y, n_gen=20, pop_size=30, random_state=42)
    y_pred_suff = reg_suff.predict(X)

    # Compare
    print(f"Additive R²: {r2_score(y, y_pred_add):.4f}")
    print(f"Sufficient R²: {r2_score(y, y_pred_suff):.4f}")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    axes[0].scatter(y, y_pred_add, alpha=0.5)
    axes[0].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[0].set_title(f'Additive Mode (R²={r2_score(y, y_pred_add):.4f})')
    axes[0].set_xlabel('True')
    axes[0].set_ylabel('Predicted')
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(y, y_pred_suff, alpha=0.5)
    axes[1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    axes[1].set_title(f'Sufficient Mode (R²={r2_score(y, y_pred_suff):.4f})')
    axes[1].set_xlabel('True')
    axes[1].set_ylabel('Predicted')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

Advanced Techniques
===================

Custom Linguistic Variables
----------------------------

Define your own fuzzy partitions:

.. code-block:: python

    from ex_fuzzy import fuzzy_sets as fs
    from ex_fuzzy.evolutionary_fit_regression import BaseFuzzyRulesRegressor

    # Create custom partitions for each feature
    x1_lvs = [
        fs.FS('Very_Low', [0, 0, 1, 2], [0, 10]),
        fs.FS('Low', [1, 2, 3, 4], [0, 10]),
        fs.FS('Medium', [3, 4, 6, 7], [0, 10]),
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
        linguistic_variables=[x1_var, x2_var],
        consequent_type='crisp',
        rule_mode='additive'
    )

    regressor.fit(X, y, n_gen=20, pop_size=30)

GPU Acceleration
----------------

Use EvoX backend for faster training on large datasets:

.. code-block:: python

    # Train with GPU acceleration
    regressor = BaseFuzzyRulesRegressor(
        nRules=20,
        nAnts=4,
        consequent_type='crisp',
        rule_mode='additive',
        n_linguistic_variables=3
    )

    # Use EvoX backend
    regressor.fit(
        X_train,
        y_train,
        n_gen=50,
        pop_size=100,
        backend='evox',  # GPU-accelerated
        device='cuda'
    )

Model Persistence
-----------------

Persist fuzzy variables and human-readable rules:

.. code-block:: python

    from ex_fuzzy import persistence

    # Train and capture artifacts
    regressor.fit(X_train, y_train, n_gen=30, pop_size=50)
    rules_text = regressor.rule_base.print_rules_regression(
        return_rules=True,
        output_name="output",
    )
    variables_text = persistence.save_fuzzy_variables(regressor.lvs)

Cross-Validation
----------------

.. code-block:: python

    from sklearn.model_selection import KFold
    from sklearn.metrics import r2_score
    import numpy as np

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in kfold.split(X):
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]

        regressor = BaseFuzzyRulesRegressor(
            nRules=10, nAnts=2,
            consequent_type='crisp',
            rule_mode='additive'
        )
        regressor.fit(X_train_fold, y_train_fold, n_gen=20, pop_size=30)

        y_pred = regressor.predict(X_val_fold)
        score = r2_score(y_val_fold, y_pred)
        scores.append(score)

    print(f"Cross-validation R² scores: {scores}")
    print(f"Mean R²: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")

See Also
========

- :doc:`../user-guide/regression` - Complete regression guide
- :doc:`../api/regression` - API reference
- :doc:`classification` - Classification examples
- :doc:`../user-guide/validation-visualization` - Visualizing results

Next Steps
==========

Try these examples in your own projects and explore:

- Different rule modes (additive vs sufficient)
- Fuzzy vs crisp consequents
- Custom linguistic variables
- GPU acceleration for large datasets
