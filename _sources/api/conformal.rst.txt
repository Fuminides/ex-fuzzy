Conformal Module
================

The :mod:`ex_fuzzy.conformal` module provides conformal prediction utilities for fuzzy classifiers.

.. currentmodule:: ex_fuzzy.conformal

Overview
--------

This module wraps fuzzy classifiers to output prediction sets with coverage guarantees and
supports evaluation of empirical coverage.

Classes
-------

ConformalFuzzyClassifier
------------------------

.. autoclass:: ConformalFuzzyClassifier
   :members:
   :exclude-members: rule_base, nclasses_
   :show-inheritance:

   A conformal wrapper that provides prediction sets and calibration utilities.

evaluate_conformal_coverage
---------------------------

.. autofunction:: evaluate_conformal_coverage

Examples
--------

.. code-block:: python

   from ex_fuzzy.conformal import ConformalFuzzyClassifier, evaluate_conformal_coverage
   from sklearn.model_selection import train_test_split

   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=0)
   X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=0)

   conf_clf = ConformalFuzzyClassifier(nRules=20, nAnts=4)
   conf_clf.fit(X_train, y_train, X_cal, y_cal, n_gen=50, pop_size=50)

   pred_sets = conf_clf.predict_set(X_test, alpha=0.1)
   metrics = evaluate_conformal_coverage(conf_clf, X_test, y_test, alpha=0.1)
