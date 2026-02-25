Classifiers Module
==================

The :mod:`ex_fuzzy.classifiers` module provides the main classification interface for the ex-fuzzy library.

.. currentmodule:: ex_fuzzy.classifiers

Overview
--------

This module contains the high-level classifier that combines rule mining and genetic optimization for fuzzy classification tasks.

Classes
-------

RuleMineClassifier
------------------

.. autoclass:: RuleMineClassifier
   :members:
   :show-inheritance:

   The main classifier that mines candidate rules and then optimizes them using genetic algorithms.

Examples
--------

Basic Usage
~~~~~~~~~~~

.. code-block:: python

   from ex_fuzzy.classifiers import RuleMineClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   # Load data
   X, y = load_iris(return_X_y=True)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

   # Create and train classifier
   classifier = RuleMineClassifier(nRules=20, nAnts=4, verbose=True)
   classifier.fit(X_train, y_train)

   # Make predictions
   y_pred = classifier.predict(X_test)
   accuracy = classifier.score(X_test, y_test)
   print(f"Accuracy: {accuracy:.3f}")

See Also
--------

* :mod:`ex_fuzzy.evolutionary_fit` : Underlying genetic optimization 
* :mod:`ex_fuzzy.rule_mining` : Rule mining functionality
* :mod:`ex_fuzzy.fuzzy_sets` : Fuzzy set definitions
