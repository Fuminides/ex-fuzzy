Rule Mining Module
==================

The :mod:`ex_fuzzy.rule_mining` module provides comprehensive fuzzy rule mining capabilities for extracting meaningful rules from datasets.

.. currentmodule:: ex_fuzzy.rule_mining

Overview
--------

This module implements efficient algorithms for discovering frequent fuzzy patterns and generating fuzzy association rules using support-based itemset mining.

**Key Capabilities:**

* **Itemset Mining**: Discovery of frequent fuzzy itemsets using support thresholds
* **Rule Generation**: Conversion of frequent itemsets into fuzzy association rules  
* **Support Calculation**: Efficient computation of fuzzy support measures
* **Quality Filtering**: Support, confidence, and lift-based rule filtering
* **Evolutionary Integration**: Seamless compatibility with genetic optimization

**Workflow:**

1. **Pattern Discovery**: Mine frequent fuzzy itemsets from training data
2. **Rule Extraction**: Generate candidate rules from frequent patterns
3. **Quality Assessment**: Filter rules based on support, confidence, lift
4. **Optimization Ready**: Provide candidate rules for evolutionary algorithms

Functions
---------

.. autosummary::
   :toctree: generated/

   mine_fuzzy_rules
   compute_support
   generate_itemsets
   filter_rules_by_quality
   calculate_confidence
   calculate_lift

Core Functions
--------------

Rule Mining
~~~~~~~~~~~

.. autofunction:: mine_fuzzy_rules

Support Calculation
~~~~~~~~~~~~~~~~~~~

.. autofunction:: compute_support

Itemset Generation
~~~~~~~~~~~~~~~~~~

.. autofunction:: generate_itemsets

Quality Metrics
~~~~~~~~~~~~~~~

.. autofunction:: filter_rules_by_quality
.. autofunction:: calculate_confidence  
.. autofunction:: calculate_lift

Examples
--------

Basic Rule Mining
~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.rule_mining as rm
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np
   from sklearn.datasets import load_iris

   # Load and prepare data
   X, y = load_iris(return_X_y=True)

   # Create linguistic variables
   antecedents = [
       fs.fuzzyVariable(f"feature_{i}", X[:, i], 3, fs.FUZZY_SETS.t1)
       for i in range(X.shape[1])
   ]

   # Mine fuzzy rules
   candidate_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents,
       X=X,
       y=y,
       min_support=0.1,
       min_confidence=0.6,
       min_lift=1.0,
       max_antecedents=3
   )

   print(f"Discovered {len(candidate_rules)} candidate rules")

   # Display some rules
   for i, rule in enumerate(candidate_rules[:5]):
       print(f"Rule {i+1}: {rule}")

Advanced Rule Mining
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mine rules with specific quality thresholds
   high_quality_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents,
       X=X,
       y=y,
       min_support=0.15,      # Higher support threshold
       min_confidence=0.8,    # Higher confidence threshold
       min_lift=1.5,          # Higher lift threshold
       max_antecedents=2,     # Simpler rules
       target_classes=[0, 1]  # Focus on specific classes
   )

   # Additional filtering
   filtered_rules = rm.filter_rules_by_quality(
       high_quality_rules,
       X=X,
       y=y,
       antecedents=antecedents,
       min_coverage=0.05,     # Minimum rule coverage
       max_overlap=0.3        # Maximum rule overlap
   )

Support-based Itemset Mining
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Generate frequent itemsets
   frequent_itemsets = rm.generate_itemsets(
       antecedents=antecedents,
       X=X,
       min_support=0.1,
       max_size=3
   )

   # Examine itemset support values
   for itemset, support in frequent_itemsets.items():
       print(f"Itemset {itemset}: Support = {support:.3f}")

   # Compute support for specific patterns
   pattern = [(0, 1), (1, 2)]  # Variable 0, term 1 AND Variable 1, term 2
   support = rm.compute_support(pattern, antecedents, X)
   print(f"Pattern support: {support:.3f}")

Rule Quality Assessment
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate rule quality metrics
   for rule in candidate_rules[:10]:
       # Calculate confidence
       confidence = rm.calculate_confidence(
           rule=rule,
           antecedents=antecedents,
           X=X,
           y=y
       )
       
       # Calculate lift
       lift = rm.calculate_lift(
           rule=rule,
           antecedents=antecedents,
           X=X,
           y=y
       )
       
       print(f"Rule: {rule}")
       print(f"  Confidence: {confidence:.3f}")
       print(f"  Lift: {lift:.3f}")
       print()

Custom Mining Parameters
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Mine rules for each class separately
   class_specific_rules = {}
   
   for class_id in np.unique(y):
       # Filter data for current class
       class_mask = (y == class_id)
       X_class = X[class_mask]
       y_class = y[class_mask]
       
       # Mine rules for this class
       rules = rm.mine_fuzzy_rules(
           antecedents=antecedents,
           X=X_class,
           y=y_class,
           min_support=0.2,
           min_confidence=0.7,
           min_lift=1.2,
           max_antecedents=2
       )
       
       class_specific_rules[class_id] = rules
       print(f"Class {class_id}: {len(rules)} rules")

Integration with Classifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ex_fuzzy.classifiers import RuleMineClassifier

   # Use mined rules as initial population for genetic algorithm
   classifier = RuleMineClassifier(
       nRules=20,
       nAnts=3,
       linguistic_variables=antecedents,
       initial_rules=candidate_rules[:50],  # Use mined rules as starting point
       verbose=True
   )

   # Fit classifier
   classifier.fit(X, y)
   
   # The genetic algorithm will optimize the mined rules
   optimized_rules = classifier.get_rules()
   print(f"Optimized to {len(optimized_rules)} rules")

Large Dataset Mining
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient mining for large datasets
   def mine_large_dataset(X, y, antecedents, batch_size=1000):
       """Mine rules from large dataset using batching."""
       n_samples = X.shape[0]
       all_itemsets = {}
       
       # Process in batches
       for start_idx in range(0, n_samples, batch_size):
           end_idx = min(start_idx + batch_size, n_samples)
           X_batch = X[start_idx:end_idx]
           
           # Mine itemsets from batch
           batch_itemsets = rm.generate_itemsets(
               antecedents=antecedents,
               X=X_batch,
               min_support=0.05,
               max_size=2
           )
           
           # Merge with global itemsets
           for itemset, support in batch_itemsets.items():
               if itemset in all_itemsets:
                   all_itemsets[itemset] += support
               else:
                   all_itemsets[itemset] = support
       
       # Normalize supports
       for itemset in all_itemsets:
           all_itemsets[itemset] /= (n_samples // batch_size)
       
       return all_itemsets

   # Use for large dataset
   if X.shape[0] > 10000:
       itemsets = mine_large_dataset(X, y, antecedents)
   else:
       itemsets = rm.generate_itemsets(antecedents, X, min_support=0.1)

Rule Pattern Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze patterns in mined rules
   def analyze_rule_patterns(rules, antecedents):
       """Analyze patterns in rule antecedents."""
       variable_usage = {i: 0 for i in range(len(antecedents))}
       term_usage = {}
       
       for rule in rules:
           for var_idx, term_idx in rule.antecedents:
               variable_usage[var_idx] += 1
               
               if (var_idx, term_idx) not in term_usage:
                   term_usage[(var_idx, term_idx)] = 0
               term_usage[(var_idx, term_idx)] += 1
       
       return variable_usage, term_usage

   # Analyze mined rules
   var_usage, term_usage = analyze_rule_patterns(candidate_rules, antecedents)

   print("Variable usage frequency:")
   for var_idx, count in var_usage.items():
       print(f"  Variable {var_idx}: {count} rules")

   print("\\nMost common terms:")
   sorted_terms = sorted(term_usage.items(), key=lambda x: x[1], reverse=True)
   for (var_idx, term_idx), count in sorted_terms[:10]:
       print(f"  Var {var_idx}, Term {term_idx}: {count} rules")

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Optimize mining performance
   import time

   def benchmark_mining_parameters():
       """Compare different mining parameters for performance."""
       parameters = [
           {'min_support': 0.05, 'max_antecedents': 2},
           {'min_support': 0.1, 'max_antecedents': 3},
           {'min_support': 0.15, 'max_antecedents': 4},
       ]
       
       results = []
       for params in parameters:
           start_time = time.time()
           
           rules = rm.mine_fuzzy_rules(
               antecedents=antecedents,
               X=X,
               y=y,
               min_confidence=0.6,
               min_lift=1.0,
               **params
           )
           
           mining_time = time.time() - start_time
           results.append({
               'params': params,
               'n_rules': len(rules),
               'time': mining_time
           })
           
           print(f"Parameters {params}: {len(rules)} rules in {mining_time:.2f}s")
       
       return results

   # Run benchmark
   benchmark_results = benchmark_mining_parameters()

Algorithmic Details
-------------------

Support Calculation
~~~~~~~~~~~~~~~~~~~

The module uses fuzzy support measures that account for partial membership:

.. math::

   Support(A) = \\frac{1}{n} \\sum_{i=1}^{n} \\mu_A(x_i)

where :math:`\\mu_A(x_i)` is the membership degree of sample :math:`x_i` in fuzzy set :math:`A`.

Confidence Calculation
~~~~~~~~~~~~~~~~~~~~~~

Fuzzy confidence measures the strength of the implication:

.. math::

   Confidence(A \\rightarrow C) = \\frac{Support(A \\cup C)}{Support(A)}

Lift Calculation
~~~~~~~~~~~~~~~~

Fuzzy lift measures how much more likely the consequent is given the antecedent:

.. math::

   Lift(A \\rightarrow C) = \\frac{Confidence(A \\rightarrow C)}{Support(C)}

See Also
--------

* :mod:`ex_fuzzy.evolutionary_fit` : Genetic optimization of mined rules
* :mod:`ex_fuzzy.rules` : Rule representation and inference
* :mod:`ex_fuzzy.classifiers` : High-level classification interfaces
* :mod:`ex_fuzzy.fuzzy_sets` : Fuzzy variable and set definitions

References
----------

.. [1] Agrawal, R., and R. Srikant. "Fast algorithms for mining association rules." 
       Proceedings of the 20th VLDB Conference, 1994.

.. [2] Delgado, M., et al. "Fuzzy association rules: general model and applications." 
       IEEE Transactions on Fuzzy Systems 11.2 (2003): 214-225.

.. [3] Hong, T.P., et al. "Mining fuzzy association rules from quantitative data." 
       Intelligent Data Analysis 3.5 (1999): 363-376.
