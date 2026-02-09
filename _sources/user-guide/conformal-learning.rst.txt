.. _conformal-learning:

Conformal Learning
==================

Ex-Fuzzy includes conformal prediction utilities that wrap fuzzy classifiers to provide
set-valued predictions with statistically valid coverage guarantees.
This is useful when you want reliable uncertainty quantification alongside interpretable rules.

What You Get
------------

- A wrapper classifier that outputs prediction sets instead of single labels.
- Coverage guarantees controlled by a miscoverage rate ``alpha`` (e.g., 0.1 -> 90% target coverage).
- Rule-aware prediction sets to preserve interpretability.
- Evaluation utilities to measure empirical coverage and set sizes.

Key Concepts
------------

**Calibration set**
  A held-out dataset used to calibrate the conformal predictor. It must be separate from training data.

**Miscoverage rate (``alpha``)**
  The target probability that the true label is *not* in the predicted set. Lower ``alpha`` yields larger sets.

**Prediction set**
  A set of labels returned per sample that should contain the true label with probability ``1 - alpha``.

Core API
--------

- :class:`ex_fuzzy.conformal.ConformalFuzzyClassifier`
- :func:`ex_fuzzy.conformal.evaluate_conformal_coverage`

Typical Workflow
----------------

1. Split data into train, calibration, and test sets.
2. Fit a fuzzy classifier.
3. Calibrate the conformal wrapper on the calibration set.
4. Predict sets and evaluate coverage.

Example: Train and Calibrate
----------------------------

.. code-block:: python

    from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage
    from sklearn.model_selection import train_test_split

    # Split into train, calibration, and test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=0
    )

    # Train + calibrate
    conf_clf = ConformalFuzzyClassifier(nRules=20, nAnts=4, backend="pymoo")
    conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50, pop_size=50)

    # Predict sets (alpha = 0.1 -> 90% target coverage)
    pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

    # Predict sets with rule explanations
    pred_sets_with_rules = conf_clf.predict_set_with_rules(X_test, alpha=0.1)

    # Evaluate empirical coverage
    metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)

Wrapping an Existing Classifier
-------------------------------

You can also wrap a trained classifier and calibrate it:

.. code-block:: python

    from ex_fuzzy.conformal import ConformalFuzzyClassifier

    base_clf = BaseFuzzyRulesClassifier(nRules=20, nAnts=4)
    base_clf.fit(X_train, y_train, n_gen=50, pop_size=50)

    conf_clf = ConformalFuzzyClassifier(base_clf)
    conf_clf.calibrate(X_cal, y_cal)
    pred_sets = conf_clf.predict_set(X_test, alpha=0.1)

Practical Notes
---------------

- Calibration quality depends on the representativeness and size of the calibration set.
- Smaller ``alpha`` yields larger, more conservative prediction sets.
- Use ``predict_set_with_rules`` when you need explanations for why each label appears in the set.
