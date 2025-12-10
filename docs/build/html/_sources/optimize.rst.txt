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

You can select the backend when creating a classifier:

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier
   
   # PyMoo backend (default)
   clf_pymoo = BaseFuzzyRulesClassifier(backend='pymoo')
   
   # EvoX backend (GPU-accelerated)
   clf_evox = BaseFuzzyRulesClassifier(backend='evox')

The criteria used to determine optimal is the one mentioned in :ref:`step3`:

1. Matthew Correlation Coefficient: it is a metric that ranges from [-1, 1] that measures the quality of a classification performance. It less sensible to imbalance classification than the standard accuracy.
2. Less antecedents: the less antecedents per rule, the better. We compute this using the average number of antecedents per rule. We to normalize this by dividing the number of antecedents per rule by the maximum allowed in the optimization) 
3. Less rules: rule bases with less rules are prefered. We normalize this by dividing the number of rules present in the database with dominance score bigger than the minimum threshold by the possible number of rules allowed in the optimization.

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

By default, the fitness function is just Matthew Correlation Coefficient. You can add laplacian multiplier to penalize the number of rules and rule antecedent size preference. 
For more information about changing this fitness function check :ref:`extending`.
