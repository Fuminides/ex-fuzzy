.. _ga:

Genetic algorithm details
=======================================

The genetic algorithm searchs for the optimal rule base for a problem. The criteria used to determine optimal is the one mentioned in :ref:`step3`:

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

By default, the fitness function is the convex combination of the Matthew Correlation Coefficient (95%), to the rule size preference (2.5%) and to the rule antecedent size preference (2.5%). 
For more information about changing this fitness function check :ref:`extending`.
