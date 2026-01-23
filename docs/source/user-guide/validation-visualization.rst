====================================
Validation and Visualization
====================================

This guide covers tools for validating your fuzzy models, visualizing results, and analyzing model robustness.

Related Guides
==============

- :doc:`../user-guide/results-io`
- :doc:`../user-guide/recipes`
- :doc:`../user-guide/troubleshooting`

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

Ex-Fuzzy provides comprehensive tools for:

- **Model Validation**: Bootstrapping, cross-validation, and statistical testing
- **Visualization**: Plot fuzzy sets, rules, and decision boundaries
- **Pattern Stability**: Analyze consistency of learned patterns across runs
- **Robustness Analysis**: Evaluate model performance under different conditions

Visualization Tools
===================

Plotting Fuzzy Variables
-------------------------

Visualize fuzzy partitions and membership functions:

.. code-block:: python

    from ex_fuzzy import vis_rules
    import matplotlib.pyplot as plt

    # After training a classifier/regressor
    classifier.fit(X_train, y_train)

    # Plot fuzzy partitions for each feature
    for idx, fv in enumerate(classifier.rule_base.antecedents):
        vis_rules.plot_fuzzy_variable(fv)
        plt.title(f'Fuzzy Partition: {fv.name}')
        plt.show()

Visualizing Rules
-----------------

Display learned rules in various formats:

.. code-block:: python

    from ex_fuzzy import eval_tools

    # Create evaluator
    evaluator = eval_tools.FuzzyEvaluator(classifier)

    # Print rules in readable format
    evaluator.eval_fuzzy_model(
        X_train, y_train, X_test, y_test,
        print_rules=True,
        plot_rules=True,
        plot_partitions=True
    )

Rule Network Visualization
--------------------------

Visualize rule structure as a network:

.. code-block:: python

    from ex_fuzzy import vis_rules

    # Requires networkx: pip install networkx
    vis_rules.plot_rule_network(classifier.rule_base)

Decision Boundaries
-------------------

For 2D problems, visualize decision regions:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    # Create mesh grid
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, 0.1),
        np.arange(y_min, y_max, 0.1)
    )

    # Predict on mesh
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.8)
    plt.title('Decision Boundaries')
    plt.show()

Pattern Stability Analysis
==========================

Analyze consistency of learned patterns across multiple training runs.

Running Stability Analysis
--------------------------

.. code-block:: python

    from ex_fuzzy import pattern_stability
    from sklearn.model_selection import train_test_split

    # Prepare data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Run pattern stability analysis
    stability_report = pattern_stability.pattern_stabilizer(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        nRules=15,
        nAnts=4,
        n_runs=10,  # Number of independent runs
        n_gen=50,
        pop_size=30,
        tolerance=0.001
    )

Interpreting Results
--------------------

The stability report includes:

- **Rule Frequency**: How often each rule appears across runs
- **Performance Consistency**: Variance in accuracy/R² across runs
- **Feature Importance**: Which features appear most consistently in rules
- **Confidence Scores**: Statistical confidence in learned patterns

.. code-block:: python

    # Access stability metrics
    print("Average accuracy:", stability_report['mean_accuracy'])
    print("Std deviation:", stability_report['std_accuracy'])
    print("Most stable rules:", stability_report['top_rules'])

Visualizing Stability
---------------------

.. code-block:: python

    # Plot stability metrics
    pattern_stability.plot_stability_report(stability_report)
    
    # Feature importance plot
    pattern_stability.plot_feature_importance(stability_report)

Bootstrapping and Statistical Tests
====================================

Bootstrap Validation
--------------------

Assess model confidence using bootstrapping:

.. code-block:: python

    from ex_fuzzy import bootstrapping_test

    # Train model
    classifier.fit(X_train, y_train)

    # Bootstrap validation
    bootstrap_results = bootstrapping_test.bootstrap_evaluation(
        classifier,
        X_test,
        y_test,
        n_iterations=1000,
        confidence_level=0.95
    )

    print(f"Mean accuracy: {bootstrap_results['mean_accuracy']:.4f}")
    print(f"95% CI: [{bootstrap_results['ci_lower']:.4f}, {bootstrap_results['ci_upper']:.4f}]")

Permutation Test
----------------

Test statistical significance of results:

.. code-block:: python

    from ex_fuzzy import permutation_test

    # Compare model against null hypothesis
    p_value = permutation_test.permutation_test(
        classifier,
        X_test,
        y_test,
        n_permutations=1000,
        metric='accuracy'
    )

    print(f"P-value: {p_value:.4f}")
    if p_value < 0.05:
        print("Results are statistically significant!")

Cross-Validation
================

K-Fold Cross-Validation
-----------------------

.. code-block:: python

    from sklearn.model_selection import cross_val_score
    from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier

    # Initialize classifier
    classifier = BaseFuzzyRulesClassifier(nRules=15, nAnts=4)

    # Perform 5-fold cross-validation
    scores = cross_val_score(
        classifier,
        X, y,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    print(f"Cross-validation scores: {scores}")
    print(f"Mean accuracy: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

Stratified K-Fold
-----------------

For imbalanced datasets:

.. code-block:: python

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Stratified cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(
        classifier,
        X, y,
        cv=skf,
        scoring='accuracy'
    )

Performance Metrics
===================

Classification Metrics
----------------------

.. code-block:: python

    from ex_fuzzy import eval_tools
    from sklearn.metrics import classification_report, confusion_matrix

    # Predictions
    y_pred = classifier.predict(X_test)

    # Detailed classification report
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # Using FuzzyEvaluator
    evaluator = eval_tools.FuzzyEvaluator(classifier)
    accuracy = evaluator.get_metric("accuracy_score", X_test, y_test)
    f1_weighted = evaluator.get_metric("f1_score", X_test, y_test, average="weighted")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1_weighted:.4f}")

Regression Metrics
------------------

.. code-block:: python

    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
    import numpy as np

    # Predictions
    y_pred = regressor.predict(X_test)

    # Compute metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)

    print(f"R² Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")

Model Comparison
================

Comparing Multiple Models
-------------------------

.. code-block:: python

    from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.metrics import accuracy_score
    import time

    models = {
        'Fuzzy Classifier': BaseFuzzyRulesClassifier(nRules=15, nAnts=4),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(random_state=42)
    }

    results = {}
    for name, model in models.items():
        # Train
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Test
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'accuracy': accuracy,
            'train_time': train_time
        }

    # Display results
    import pandas as pd
    df_results = pd.DataFrame(results).T
    print(df_results)

Visualization of Comparisons
----------------------------

.. code-block:: python

    import matplotlib.pyplot as plt

    # Bar plot of accuracies
    names = list(results.keys())
    accuracies = [results[n]['accuracy'] for n in names]

    plt.figure(figsize=(10, 6))
    plt.bar(names, accuracies)
    plt.ylabel('Accuracy')
    plt.title('Model Comparison')
    plt.ylim([0, 1])
    plt.grid(axis='y', alpha=0.3)
    plt.show()

Advanced Visualization
======================


Interactive Plots
-----------------

For interactive exploration (requires plotly):

.. code-block:: python

    # Install: pip install plotly
    import plotly.express as px

    # Interactive scatter plot with predictions
    import pandas as pd
    df = pd.DataFrame(X_test, columns=['Feature 1', 'Feature 2'])
    df['True Label'] = y_test
    df['Predicted'] = y_pred
    df['Correct'] = df['True Label'] == df['Predicted']

    fig = px.scatter(
        df, 
        x='Feature 1', 
        y='Feature 2',
        color='Predicted',
        symbol='Correct',
        title='Classification Results'
    )
    fig.show()

Best Practices
==============

1. **Multiple Runs**: Always run pattern stability analysis with at least 10-20 runs
2. **Bootstrapping**: Use bootstrap confidence intervals for final model evaluation
3. **Visualization**: Always visualize learned fuzzy partitions to check interpretability
4. **Validate fuzzy Partitions**: Use the tools present in utils module to validate fuzzy partitions.




See Also
========

- :doc:`../user-guide/results-io` - Save rules, plots, and metrics
- :doc:`../user-guide/recipes` - Quick evaluation workflows
- :doc:`../api/index` - API reference
- :doc:`../examples/index` - Worked examples

Next Steps
==========

- Explore :doc:`../persistence` for saving validated models
- Check :doc:`../examples/index` for real-world validation workflows
- Review :doc:`../user-guide/troubleshooting` if you hit errors
