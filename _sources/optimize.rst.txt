.. _ga:

Genetic algorithm details
=======================================

The genetic algorithm searches for the optimal rule base for a problem. Ex-Fuzzy supports two evolutionary optimization backends:

**PyMoo Backend (CPU)**:
  - Traditional CPU-based optimization
  - Supports checkpointing and resume
  - Best for small to medium datasets
  - Memory-efficient with automatic sample batching

**EvoX Backend (GPU-accelerated)**:
  - GPU acceleration using PyTorch
  - 2-10x faster on large datasets with CUDA GPU
  - Automatic population batching for memory management
  - Seamlessly falls back to CPU if GPU unavailable

You can select the backend when creating a classifier or regressor:

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier, BaseFuzzyRulesRegressor
   
   # PyMoo backend (default)
   clf_pymoo = BaseFuzzyRulesClassifier(backend='pymoo')
   
   # EvoX backend (GPU-accelerated)
   clf_evox = BaseFuzzyRulesClassifier(backend='evox')

   # Regression uses the same backend abstraction
   reg_evox = BaseFuzzyRulesRegressor(
       nRules=20,
       nAnts=3,
       consequent_type='crisp',
       backend='evox',
   )

For regression, the search maximizes training-set :math:`R^2`. EvoX evaluates
crisp Takagi-Sugeno and fuzzy Mamdani regression populations with PyTorch on
CUDA when available, and falls back to the same batched path on CPU. See
:doc:`user-guide/regression` for the full workflow.

For classification, the search maximizes the macro F1 of the rule base on the
training data, after pruning, minus two small penalties: one for the share of
classes left without rules and one for the share of the possible antecedent
conditions the rule base uses. `Fitness function`_ gives the details, and
`Single-objective or Pareto search`_ the alternative that keeps the whole
trade-off between accuracy and size.

It is possible to use previously computed rulesin order to fine tune them. There are two ways to do this using the ``ex_fuzzy.evolutionary_fit.BaseFuzzyRulesClassifier``:

1. Use the previously computed rules as the initial population for a new optimization problem. In that case, you can pass that rules to the ``initial_rules`` parameter the ``ex_fuzzy.rules.MasterRuleBase`` object.
2. Look for more efficient subsets of rules in the previously computed rules. In this case the genetic optimization will use those rules as the search space itself, and will try to optimize the best subset of them.  In that case, you can pass that rules to the ``candidate_rules`` parameter the ``ex_fuzzy.rules.MasterRuleBase`` object.

---------------------------------------
Limitations of the optimization process
---------------------------------------

- General Type 2 requires precomputed fuzzy partitions.
- When optimizing IV fuzzy partitions: Not all possible shapes of trapezoids all supported. Optimized trapezoids will always have max memberships for the lower and upper bounds in the same points. Height of the lower membership is optimized by scaling. Upper membership always reaches 1 at some point.

----------------
Fitness function
----------------

The default search maximizes a single fitness value:

.. code-block:: text

   fitness = macro F1 - beta  * (classes without rules / classes)
                      - alpha * (conditions / (nRules * nAnts))

Macro F1 averages the F1 score of every class, so small classes count as much as
large ones, and a class without rules scores 0. The penalties are deliberately
small, ``alpha=0.03`` and ``beta=0.05`` by default, so they mostly decide between
rule bases of similar accuracy. ``reparametrize_loss`` changes them:

.. code-block:: python

   clf = BaseFuzzyRulesClassifier(nRules=30, nAnts=4)
   clf.reparametrize_loss(alpha=0.1, beta=0.05)  # prefer smaller rule bases

On 43 KEEL datasets, the default penalties gave rule bases with about a fifth
fewer conditions than no penalties, at the same test macro F1, and
``alpha=0.1`` removed about a third of the conditions for a small loss of
accuracy. To optimize something else entirely, see :ref:`extending`.

---------------------------------
Single-objective or Pareto search
---------------------------------

The ``algorithm`` argument chooses how the search treats compactness:

- ``'ga'`` (default): the genetic algorithm maximizes the fitness above, with
  tournaments of ``tournament_size`` candidates.
- ``'nsga2'``: NSGA-II makes compactness a second objective instead of a
  penalty. It searches the trade-off between accuracy (the macro F1 with the
  class-coverage penalty) and the share of conditions used, and the classifier
  keeps the whole Pareto front.

.. code-block:: python

   clf = BaseFuzzyRulesClassifier(nRules=30, nAnts=4, algorithm='nsga2')
   clf.fit(X_train, y_train, n_gen=100, pop_size=100)

   for index, solution in enumerate(clf.pareto_front_):
       print(index, solution['fitness'], solution['rules'], solution['conditions'])

   clf.select_pareto_solution(-1)  # the most compact solution
   y_pred = clf.predict(X_test)

``pareto_front_`` is ordered from the most accurate solution on the training
data to the most compact. ``fit`` selects the most accurate one, at index 0, and
``select_pareto_solution`` switches the classifier to any other: its rule base,
its evaluation, ``performance`` and, when the partitions were optimized, its
fuzzy partitions. Each entry holds the training ``fitness``, the number of
``rules`` and ``conditions``, the fitted ``rule_base`` and its ``evaluation``.
Choose among the solutions with held-out data, not with their training fitness.

**What to expect.** Even the most accurate solution on the front is usually
smaller than the genetic algorithm's rule base. On 61 KEEL datasets, at the same
search budget and with the objective of an earlier version, it had about half
the conditions and a third fewer rules, for about one point less test accuracy.
The other solutions trade more accuracy for size.

**Cost and limits.** NSGA-II scores as many candidates as the genetic algorithm
and uses the same fitness cache and array evaluator, so a fit with optimized
partitions takes about as long. With fixed partitions the genetic algorithm can
also score whole populations at once, which NSGA-II cannot, so it may be faster
there. ``algorithm='nsga2'`` needs the PyMoo backend and the built-in loss, and
it ignores ``alpha`` and ``tournament_size``. It works with ``candidate_rules``.
