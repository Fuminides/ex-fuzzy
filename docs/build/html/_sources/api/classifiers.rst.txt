Classifiers Module
==================

The :mod:`ex_fuzzy.classifiers` module provides high-level classification algorithms that combine rule mining, genetic optimization, and fuzzy inference for pattern classification tasks.

.. currentmodule:: ex_fuzzy.classifiers

Overview
--------

The classifiers module implements sophisticated two-stage optimization approaches:

1. **Rule Discovery**: Mine candidate rules from training data using support, confidence, and lift thresholds
2. **Rule Optimization**: Use evolutionary algorithms to find optimal rule combinations

All classifiers follow scikit-learn conventions with standard :meth:`fit` and :meth:`predict` interfaces.

Key Features
~~~~~~~~~~~~

* **Automatic Feature Fuzzification**: Optimal partitioning with various fuzzy set types
* **Rule Mining**: Support, confidence, and lift-based rule discovery
* **Multi-objective Optimization**: Balance accuracy and interpretability
* **Imbalanced Dataset Support**: Specialized fitness functions
* **Cross-validation Fitness**: Robust model evaluation
* **Fuzzy Set Integration**: Support for Type-1, Type-2, and GT2 fuzzy sets

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   RuleMineClassifier

RuleMineClassifier
------------------

.. autoclass:: RuleMineClassifier
   :members:
   :inherited-members:
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :nosignatures:

      ~RuleMineClassifier.fit
      ~RuleMineClassifier.predict
      ~RuleMineClassifier.predict_proba
      ~RuleMineClassifier.score

   **Constructor Parameters**

   .. automethod:: __init__

Examples
--------

Basic Classification
~~~~~~~~~~~~~~~~~~~~

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

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.fuzzy_sets as fs

   # Create custom fuzzy variables
   fuzzy_vars = [
       fs.fuzzyVariable("feature_1", X_train[:, 0], 3, fs.FUZZY_SETS.t1),
       fs.fuzzyVariable("feature_2", X_train[:, 1], 3, fs.FUZZY_SETS.t1),
       fs.fuzzyVariable("feature_3", X_train[:, 2], 3, fs.FUZZY_SETS.t1),
       fs.fuzzyVariable("feature_4", X_train[:, 3], 3, fs.FUZZY_SETS.t1)
   ]

   # Create classifier with custom settings
   classifier = RuleMineClassifier(
       nRules=30,
       nAnts=4,
       fuzzy_type=fs.FUZZY_SETS.t1,
       tolerance=0.1,
       linguistic_variables=fuzzy_vars,
       verbose=True
   )

   classifier.fit(X_train, y_train)

With Cross-Validation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import cross_val_score

   # Evaluate with cross-validation
   classifier = RuleMineClassifier(nRules=25, nAnts=3)
   scores = cross_val_score(classifier, X_train, y_train, cv=5)
   print(f"CV Accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

Parameter Tuning
~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import GridSearchCV

   # Define parameter grid
   param_grid = {
       'nRules': [10, 20, 30],
       'nAnts': [3, 4, 5],
       'tolerance': [0.0, 0.1, 0.2]
   }

   # Grid search
   classifier = RuleMineClassifier()
   grid_search = GridSearchCV(classifier, param_grid, cv=3, scoring='accuracy')
   grid_search.fit(X_train, y_train)

   print(f"Best parameters: {grid_search.best_params_}")
   print(f"Best score: {grid_search.best_score_:.3f}")

See Also
--------

* :mod:`ex_fuzzy.fuzzy_sets` : Fuzzy set definitions and operations
* :mod:`ex_fuzzy.rule_mining` : Rule discovery algorithms  
* :mod:`ex_fuzzy.evolutionary_fit` : Genetic optimization algorithms
* :mod:`ex_fuzzy.eval_tools` : Performance evaluation utilities

References
----------

.. [1] Alcalá-Fdez, J., et al. "KEEL: a software tool to assess evolutionary algorithms for data mining problems." 
       Soft Computing 13.3 (2009): 307-318.

.. [2] Herrera, F. "Genetic fuzzy systems: taxonomy, current research trends and prospects." 
       Evolutionary Intelligence 1.1 (2008): 27-46.
