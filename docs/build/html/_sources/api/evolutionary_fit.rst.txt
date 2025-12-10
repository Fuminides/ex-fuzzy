Evolutionary Fit Module
=======================

The :mod:`ex_fuzzy.evolutionary_fit` module implements genetic algorithm-based optimization for learning fuzzy rule bases.

.. currentmodule:: ex_fuzzy.evolutionary_fit

Overview
--------

This module provides automatic rule discovery, parameter tuning, and structure optimization for fuzzy inference systems using evolutionary computation techniques.

**Core Capabilities:**

* **Automatic Rule Learning**: Discover optimal rule antecedents and consequents
* **Multi-objective Optimization**: Balance accuracy vs. interpretability
* **Parallel Evaluation**: Efficient fitness computation using threading
* **Cross-validation**: Robust fitness evaluation with stratified CV
* **Pymoo Integration**: Leverages the powerful Pymoo optimization framework

**Optimization Targets:**

* Rule antecedents (variable and linguistic term selection)
* Rule consequents (output class assignments)  
* Rule structure (number of rules, complexity constraints)
* Membership function parameters (integration with other modules)

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FitRuleBase
   EvolutionaryFitness

Functions
---------

.. autosummary::
   :toctree: generated/

   evolutionary_fit
   evaluate_fitness
   create_initial_population
   genetic_operators

Optimization Problem
--------------------

FitRuleBase
~~~~~~~~~~~

.. autoclass:: FitRuleBase
   :members:
   :inherited-members:
   :show-inheritance:

   **Core Methods**

   .. autosummary::
      :nosignatures:

      ~FitRuleBase._evaluate
      ~FitRuleBase.decode_solution
      ~FitRuleBase.encode_rulebase

EvolutionaryFitness
~~~~~~~~~~~~~~~~~~~

.. autoclass:: EvolutionaryFitness
   :members:
   :inherited-members:
   :show-inheritance:

Main Functions
--------------

Evolutionary Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: evolutionary_fit

Fitness Evaluation
~~~~~~~~~~~~~~~~~~

.. autofunction:: evaluate_fitness

Examples
--------

Basic Rule Base Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.evolutionary_fit as evf
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np
   from sklearn.datasets import load_iris

   # Load data
   X, y = load_iris(return_X_y=True)

   # Create linguistic variables
   antecedents = [
       fs.fuzzyVariable(f"feature_{i}", X[:, i], 3, fs.FUZZY_SETS.t1)
       for i in range(X.shape[1])
   ]

   # Setup optimization problem
   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=10,
       n_class=3,
       fitness_function='accuracy'
   )

   # Run genetic algorithm
   result = evf.evolutionary_fit(
       problem=problem,
       n_gen=50,
       pop_size=100,
       verbose=True
   )

   # Extract best rule base
   best_rulebase = problem.decode_solution(result.X)
   print(f"Best fitness: {result.F}")

Multi-objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize for both accuracy and rule complexity
   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=15,
       n_class=3,
       fitness_function=['accuracy', 'complexity'],
       weights=[0.8, 0.2]  # 80% accuracy, 20% simplicity
   )

   # Use NSGA-II for multi-objective optimization
   from pymoo.algorithms.moo.nsga2 import NSGA2

   algorithm = NSGA2(
       pop_size=100,
       eliminate_duplicates=True
   )

   result = evf.evolutionary_fit(
       problem=problem,
       algorithm=algorithm,
       n_gen=100,
       verbose=True
   )

   # Analyze Pareto front
   pareto_solutions = result.X
   pareto_fitness = result.F

Cross-validation Based Fitness
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use cross-validation for robust fitness evaluation
   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=12,
       n_class=3,
       fitness_function='mcc',  # Matthews Correlation Coefficient
       cv_folds=5,
       stratified=True
   )

   # Configure genetic operators
   algorithm = GA(
       pop_size=80,
       sampling=IntegerRandomSampling(),
       crossover=SBX(eta=15, prob=0.9),
       mutation=PolynomialMutation(eta=20, prob=0.1),
       eliminate_duplicates=True
   )

   result = evf.evolutionary_fit(
       problem=problem,
       algorithm=algorithm,
       n_gen=75,
       seed=42
   )

Parallel Fitness Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from multiprocessing import cpu_count

   # Setup parallel evaluation
   n_threads = min(4, cpu_count())
   pool = ThreadPool(n_threads)
   runner = StarmapParallelization(pool.starmap)

   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=20,
       n_class=3,
       fitness_function='f1_macro',
       parallelization=runner
   )

   # Run with parallel evaluation
   result = evf.evolutionary_fit(
       problem=problem,
       n_gen=60,
       pop_size=120,
       verbose=True
   )

   pool.close()
   pool.join()

Custom Fitness Functions
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def custom_fitness(y_true, y_pred, rulebase=None):
       """Custom fitness combining accuracy and rule diversity."""
       from sklearn.metrics import accuracy_score
       
       accuracy = accuracy_score(y_true, y_pred)
       
       # Penalty for rule complexity
       if rulebase is not None:
           complexity_penalty = len(rulebase.rules) * 0.01
           return accuracy - complexity_penalty
       
       return accuracy

   # Use custom fitness
   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=15,
       n_class=3,
       fitness_function=custom_fitness
   )

Advanced Configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Advanced evolutionary parameters
   from pymoo.operators.selection.tournament import TournamentSelection
   from pymoo.operators.survival.rank_and_crowding import RankAndCrowding

   algorithm = GA(
       pop_size=150,
       sampling=IntegerRandomSampling(),
       selection=TournamentSelection(pressure=2),
       crossover=SBX(eta=10, prob=0.85),
       mutation=PolynomialMutation(eta=25, prob=0.15),
       survival=RankAndCrowding(),
       eliminate_duplicates=True
   )

   # Problem with complexity constraints
   problem = evf.FitRuleBase(
       antecedents=antecedents,
       X=X,
       y=y,
       n_rules=25,
       n_class=3,
       fitness_function='balanced_accuracy',
       max_complexity=0.8,  # Limit rule complexity
       min_support=0.05,    # Minimum rule support
       cv_folds=3
   )

   # Track evolution progress
   from pymoo.core.callback import Callback

   class ProgressCallback(Callback):
       def notify(self, algorithm):
           print(f"Generation {algorithm.n_gen}: Best = {algorithm.pop.get('F').min():.4f}")

   result = evf.evolutionary_fit(
       problem=problem,
       algorithm=algorithm,
       n_gen=100,
       callback=ProgressCallback(),
       verbose=False
   )

Hyperparameter Tuning
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from ex_fuzzy.classifiers import RuleMineClassifier

   # Define parameter ranges for evolutionary optimization
   param_grid = {
       'n_rules': [10, 15, 20, 25],
       'pop_size': [50, 100, 150],
       'n_gen': [30, 50, 75],
       'fitness_function': ['accuracy', 'f1_macro', 'mcc']
   }

   # Custom estimator using evolutionary fit
   class EvolutionaryClassifier(ClassifierMixin):
       def __init__(self, n_rules=15, pop_size=100, n_gen=50, fitness_function='accuracy'):
           self.n_rules = n_rules
           self.pop_size = pop_size  
           self.n_gen = n_gen
           self.fitness_function = fitness_function

       def fit(self, X, y):
           # Implementation using evolutionary_fit
           pass

       def predict(self, X):
           # Implementation using optimized rule base
           pass

   # Grid search over evolutionary parameters
   classifier = EvolutionaryClassifier()
   grid_search = GridSearchCV(classifier, param_grid, cv=3)
   grid_search.fit(X, y)

Fitness Functions
-----------------

Available Fitness Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The module supports multiple built-in fitness functions:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Function
     - Description
   * - ``'accuracy'``
     - Classification accuracy (default)
   * - ``'balanced_accuracy'``
     - Balanced accuracy for imbalanced datasets
   * - ``'f1_macro'``
     - Macro-averaged F1 score
   * - ``'f1_micro'``
     - Micro-averaged F1 score
   * - ``'f1_weighted'``
     - Weighted F1 score by class support
   * - ``'mcc'``
     - Matthews Correlation Coefficient
   * - ``'precision_macro'``
     - Macro-averaged precision
   * - ``'recall_macro'``
     - Macro-averaged recall
   * - ``'roc_auc'``
     - Area under ROC curve (binary only)
   * - ``'complexity'``
     - Rule base complexity measure

See Also
--------

* :mod:`ex_fuzzy.classifiers` : High-level classification interfaces
* :mod:`ex_fuzzy.rules` : Rule base classes and inference
* :mod:`ex_fuzzy.rule_mining` : Rule discovery algorithms
* :mod:`ex_fuzzy.eval_tools` : Performance evaluation utilities

References
----------

.. [1] Holland, J.H. "Adaptation in Natural and Artificial Systems." University of Michigan Press, 1975.

.. [2] Deb, K., et al. "A fast and elitist multiobjective genetic algorithm: NSGA-II." 
       IEEE Transactions on Evolutionary Computation 6.2 (2002): 182-197.

.. [3] Blank, J., and K. Deb. "Pymoo: Multi-Objective Optimization in Python." 
       IEEE Access 8 (2020): 89497-89509.
