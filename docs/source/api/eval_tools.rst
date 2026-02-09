Evaluation Tools Module
=======================

The :mod:`ex_fuzzy.eval_tools` module provides evaluation and analysis tools for fuzzy classification models.

.. currentmodule:: ex_fuzzy.eval_tools

Overview
--------

This module includes performance metrics, statistical analysis, and model evaluation tools specifically designed for fuzzy rule-based systems.

Classes
-------

Main Function
-------------

Model Evaluation
~~~~~~~~~~~~~~~~

.. autofunction:: eval_fuzzy_model

Core Class
----------

FuzzyEvaluator
~~~~~~~~~~~~~~

.. autoclass:: FuzzyEvaluator
   :members:
   :show-inheritance:

Examples
--------

Basic Model Evaluation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.eval_tools as et
   from ex_fuzzy.classifiers import RuleMineClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   # Load data and train classifier
   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   classifier = RuleMineClassifier(nRules=15, nAnts=3, verbose=True)
   classifier.fit(X_train, y_train)

   # Comprehensive evaluation
   report = et.eval_fuzzy_model(
       fl_classifier=classifier,
       X_train=X_train,
       y_train=y_train,
       X_test=X_test,
       y_test=y_test,
       plot_rules=True,
       plot_partitions=True,
       bootstrap_results_print=True
   )

   print(report)

Using FuzzyEvaluator Class
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create evaluator
   evaluator = et.FuzzyEvaluator(classifier)
   
   # Get predictions
   y_pred = evaluator.predict(X_test)
   
   # Get specific metrics
   accuracy = evaluator.get_metric('accuracy_score', X_test, y_test)
   f1_score = evaluator.get_metric('f1_score', X_test, y_test, average='macro')
   
   print(f"Accuracy: {accuracy:.3f}")
   print(f"F1-score: {f1_score:.3f}")

   # Detailed evaluation
   evaluator.eval_fuzzy_model(
       X_train, y_train, X_test, y_test,
       plot_rules=True,
       print_rules=True,
       plot_partitions=True
   )

See Also
--------

* :mod:`ex_fuzzy.classifiers` : Fuzzy classification algorithms
* :mod:`ex_fuzzy.vis_rules` : Rule visualization utilities
* :mod:`sklearn.metrics` : Standard classification metrics
