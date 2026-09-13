Classifiers Module
==================

The :mod:`ex_fuzzy.classifiers` module provides the main classification interface for the ex-fuzzy library.

.. currentmodule:: ex_fuzzy.classifiers

Overview
--------

This module contains high-level classifiers built on fuzzy rule mining.
:class:`FuzzyRulesClassifier` mines fuzzy association rules on a capped feature
space and selects a compact rule base with a genetic algorithm (see
:doc:`../user-guide/fuzzy-association-rules`). :class:`RuleMineClassifier` and
:class:`RuleFineTuneClassifier` pass mined candidate rules to
:class:`~ex_fuzzy.BaseFuzzyRulesClassifier`.

Classes
-------

FuzzyRulesClassifier
--------------------

.. autoclass:: FuzzyRulesClassifier
   :members:
   :show-inheritance:

RuleMineClassifier
------------------

.. autoclass:: RuleMineClassifier
   :members:
   :show-inheritance:

   The main classifier that mines candidate rules and then optimizes them using genetic algorithms.

RuleFineTuneClassifier
----------------------

.. autoclass:: RuleFineTuneClassifier
   :members:
   :show-inheritance:

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
