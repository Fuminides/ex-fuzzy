Rule Mining Guide
================

This guide provides comprehensive coverage of fuzzy rule mining in ex-fuzzy, from basic concepts to advanced techniques for discovering high-quality rules from data.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
--------

Fuzzy rule mining is the process of automatically discovering fuzzy association rules from datasets. These rules capture relationships between input variables and output classes in a format that's both mathematically precise and human-interpretable.

What is Rule Mining?
~~~~~~~~~~~~~~~~~~~

Rule mining discovers patterns of the form:

.. code-block:: text

   IF (antecedent conditions) THEN (consequent) WITH (confidence)

Example:
  IF temperature is HIGH AND humidity is HIGH THEN comfort is LOW (confidence: 0.85)

Key Concepts
~~~~~~~~~~~

**Support**
  How frequently a rule pattern appears in the data

**Confidence**
  How often the rule is correct when its conditions are met

**Lift**
  How much better the rule performs compared to random chance

**Coverage**
  What percentage of the dataset the rule applies to

Getting Started
--------------

Basic Rule Mining
~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.rule_mining as rm
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np
   from sklearn.datasets import load_iris

   # Load data
   X, y = load_iris(return_X_y=True)

   # Create linguistic variables
   antecedents = []
   feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
   
   for i, name in enumerate(feature_names):
       var = fs.fuzzyVariable(name, X[:, i], 3, fs.FUZZY_SETS.t1)
       antecedents.append(var)

   # Mine rules with basic parameters
   rules = rm.mine_fuzzy_rules(
       antecedents=antecedents,
       X=X,
       y=y,
       min_support=0.1,     # Rule must apply to at least 10% of data
       min_confidence=0.6,  # Rule must be correct at least 60% of time
       min_lift=1.0,        # Rule must be better than random
       max_antecedents=3    # Limit rule complexity
   )

   print(f"Discovered {len(rules)} rules")

   # Examine first few rules
   for i, rule in enumerate(rules[:5]):
       print(f"Rule {i+1}: {rule}")

Understanding the Output
~~~~~~~~~~~~~~~~~~~~~~~

Each mined rule contains:

- **Antecedents**: List of (variable_index, term_index) pairs
- **Consequent**: Target class
- **Weight**: Rule confidence/strength
- **Support**: How often the rule pattern occurs
- **Lift**: Performance vs. random

.. code-block:: python

   # Detailed rule analysis
   for rule in rules[:3]:
       print(f"\\nRule: {rule}")
       print(f"Antecedents: {rule.antecedents}")
       print(f"Consequent: class {rule.consequent}")
       print(f"Weight: {rule.weight:.3f}")
       
       # Calculate additional metrics
       support = rm.compute_support(rule.antecedents, antecedents, X)
       confidence = rm.calculate_confidence(rule, antecedents, X, y)
       lift = rm.calculate_lift(rule, antecedents, X, y)
       
       print(f"Support: {support:.3f}")
       print(f"Confidence: {confidence:.3f}")
       print(f"Lift: {lift:.3f}")

Mining Parameters
----------------

Support Threshold
~~~~~~~~~~~~~~~~

Controls the minimum frequency a pattern must have:

.. code-block:: python

   # Conservative mining (fewer, more frequent rules)
   conservative_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents,
       X=X, y=y,
       min_support=0.2,      # High support requirement
       min_confidence=0.8,
       min_lift=1.5
   )

   # Exploratory mining (more rules, including rare patterns)
   exploratory_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents,
       X=X, y=y,
       min_support=0.05,     # Low support requirement
       min_confidence=0.6,
       min_lift=1.0
   )

   print(f"Conservative: {len(conservative_rules)} rules")
   print(f"Exploratory: {len(exploratory_rules)} rules")

Confidence Threshold
~~~~~~~~~~~~~~~~~~~

Controls rule accuracy requirements:

.. code-block:: python

   # Compare different confidence levels
   confidence_levels = [0.5, 0.7, 0.9]
   
   for conf in confidence_levels:
       rules = rm.mine_fuzzy_rules(
           antecedents=antecedents,
           X=X, y=y,
           min_support=0.1,
           min_confidence=conf,
           min_lift=1.0
       )
       
       avg_accuracy = np.mean([r.weight for r in rules])
       print(f"Confidence {conf}: {len(rules)} rules, avg accuracy {avg_accuracy:.3f}")

Lift Threshold  
~~~~~~~~~~~~~

Controls how much better than random the rules must be:

.. code-block:: python

   # No lift requirement (includes random-level rules)
   all_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.1, min_confidence=0.6, min_lift=0.0
   )

   # Strong lift requirement (only significantly better rules)
   strong_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.1, min_confidence=0.6, min_lift=2.0
   )

   print(f"All rules: {len(all_rules)}")
   print(f"Strong rules: {len(strong_rules)}")

Rule Complexity
~~~~~~~~~~~~~~

Control antecedent complexity:

.. code-block:: python

   # Simple rules (1-2 conditions)
   simple_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.1, min_confidence=0.7,
       max_antecedents=2
   )

   # Complex rules (up to 4 conditions)  
   complex_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.05, min_confidence=0.6,
       max_antecedents=4
   )

   # Analyze complexity distribution
   simple_lengths = [len(r.antecedents) for r in simple_rules]
   complex_lengths = [len(r.antecedents) for r in complex_rules]

   print(f"Simple rules: avg length {np.mean(simple_lengths):.1f}")
   print(f"Complex rules: avg length {np.mean(complex_lengths):.1f}")

Advanced Mining Techniques
--------------------------

Class-Specific Mining
~~~~~~~~~~~~~~~~~~~~

Mine rules for each class separately:

.. code-block:: python

   class_rules = {}
   class_names = ['setosa', 'versicolor', 'virginica']

   for class_id in range(3):
       # Filter data for current class
       class_mask = (y == class_id)
       X_class = X[class_mask]
       y_class = y[class_mask]
       
       # Mine class-specific rules
       rules = rm.mine_fuzzy_rules(
           antecedents=antecedents,
           X=X_class,
           y=y_class,
           min_support=0.15,
           min_confidence=0.7,
           max_antecedents=2
       )
       
       class_rules[class_id] = rules
       print(f"{class_names[class_id]}: {len(rules)} rules")

   # Analyze class-specific patterns
   for class_id, rules in class_rules.items():
       print(f"\\n{class_names[class_id]} patterns:")
       for rule in rules[:3]:
           print(f"  {rule}")

Hierarchical Mining
~~~~~~~~~~~~~~~~~~

Mine rules at different granularities:

.. code-block:: python

   # Coarse-grained partitions (fewer, broader terms)
   coarse_vars = [
       fs.fuzzyVariable(name, X[:, i], 2, fs.FUZZY_SETS.t1)  # 2 terms each
       for i, name in enumerate(feature_names)
   ]

   # Fine-grained partitions (more, specific terms)
   fine_vars = [
       fs.fuzzyVariable(name, X[:, i], 5, fs.FUZZY_SETS.t1)  # 5 terms each
       for i, name in enumerate(feature_names)
   ]

   # Mine at each level
   coarse_rules = rm.mine_fuzzy_rules(
       antecedents=coarse_vars, X=X, y=y,
       min_support=0.2, min_confidence=0.7
   )

   fine_rules = rm.mine_fuzzy_rules(
       antecedents=fine_vars, X=X, y=y,
       min_support=0.05, min_confidence=0.6
   )

   print(f"Coarse level: {len(coarse_rules)} rules")
   print(f"Fine level: {len(fine_rules)} rules")

Temporal Rule Mining
~~~~~~~~~~~~~~~~~~~

For time-series or sequential data:

.. code-block:: python

   # Example with temporal features
   def add_temporal_features(X, window_size=3):
       """Add temporal features like trends and moving averages."""
       X_temporal = X.copy()
       
       # Add moving averages
       for i in range(X.shape[1]):
           ma = np.convolve(X[:, i], np.ones(window_size)/window_size, mode='same')
           X_temporal = np.column_stack([X_temporal, ma])
       
       # Add trends (simple differences)
       for i in range(X.shape[1]):
           trend = np.gradient(X[:, i])
           X_temporal = np.column_stack([X_temporal, trend])
       
       return X_temporal

   # Create temporal features
   X_temporal = add_temporal_features(X)
   
   # Create linguistic variables for all features
   temporal_vars = []
   n_original = len(feature_names)
   
   for i in range(X_temporal.shape[1]):
       if i < n_original:
           name = feature_names[i]
       elif i < 2 * n_original:
           name = f"{feature_names[i - n_original]}_ma"
       else:
           name = f"{feature_names[i - 2 * n_original]}_trend"
       
       var = fs.fuzzyVariable(name, X_temporal[:, i], 3, fs.FUZZY_SETS.t1)
       temporal_vars.append(var)

   # Mine temporal rules
   temporal_rules = rm.mine_fuzzy_rules(
       antecedents=temporal_vars,
       X=X_temporal,
       y=y,
       min_support=0.1,
       min_confidence=0.6,
       max_antecedents=3
   )

Multi-Target Mining
~~~~~~~~~~~~~~~~~~

For problems with multiple output variables:

.. code-block:: python

   # Example: Predict both class and confidence
   def mine_multi_target(X, y_primary, y_secondary):
       """Mine rules for multiple targets."""
       all_rules = {}
       
       # Mine rules for primary target
       primary_rules = rm.mine_fuzzy_rules(
           antecedents=antecedents,
           X=X, y=y_primary,
           min_support=0.1, min_confidence=0.6
       )
       all_rules['primary'] = primary_rules
       
       # Mine rules for secondary target
       secondary_rules = rm.mine_fuzzy_rules(
           antecedents=antecedents,
           X=X, y=y_secondary,
           min_support=0.1, min_confidence=0.6
       )
       all_rules['secondary'] = secondary_rules
       
       return all_rules

   # Create secondary target (confidence levels)
   from sklearn.linear_model import LogisticRegression
   lr = LogisticRegression()
   lr.fit(X, y)
   y_proba = lr.predict_proba(X)
   y_confidence = np.digitize(np.max(y_proba, axis=1), [0.6, 0.8, 1.0])

   # Mine for both targets
   multi_rules = mine_multi_target(X, y, y_confidence)

Rule Quality Assessment
----------------------

Comprehensive Quality Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_rule_quality(rules, antecedents, X, y):
       """Comprehensive quality analysis of mined rules."""
       quality_metrics = []
       
       for i, rule in enumerate(rules):
           metrics = {}
           
           # Basic metrics
           metrics['rule_id'] = i
           metrics['n_antecedents'] = len(rule.antecedents)
           metrics['consequent'] = rule.consequent
           
           # Statistical metrics
           metrics['support'] = rm.compute_support(rule.antecedents, antecedents, X)
           metrics['confidence'] = rm.calculate_confidence(rule, antecedents, X, y)
           metrics['lift'] = rm.calculate_lift(rule, antecedents, X, y)
           
           # Coverage metrics
           rule_mask = np.ones(len(X), dtype=bool)
           for var_idx, term_idx in rule.antecedents:
               memberships = antecedents[var_idx][term_idx].evaluate(X[:, var_idx])
               rule_mask &= (memberships > 0.5)  # Strong membership
           
           metrics['coverage'] = np.mean(rule_mask)
           metrics['accuracy'] = np.mean(y[rule_mask] == rule.consequent) if np.any(rule_mask) else 0
           
           # Complexity metrics
           metrics['complexity'] = len(rule.antecedents) / len(antecedents)  # Relative complexity
           
           quality_metrics.append(metrics)
       
       return pd.DataFrame(quality_metrics)

   # Analyze rule quality
   import pandas as pd
   quality_df = analyze_rule_quality(rules[:20], antecedents, X, y)
   
   # Display top rules by different criteria
   print("Top rules by lift:")
   print(quality_df.nlargest(5, 'lift')[['rule_id', 'support', 'confidence', 'lift']])
   
   print("\\nTop rules by support:")
   print(quality_df.nlargest(5, 'support')[['rule_id', 'support', 'confidence', 'lift']])

Rule Filtering and Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def filter_rules_advanced(rules, antecedents, X, y, criteria=None):
       """Advanced rule filtering with multiple criteria."""
       if criteria is None:
           criteria = {
               'min_support': 0.1,
               'min_confidence': 0.6,
               'min_lift': 1.2,
               'max_complexity': 3,
               'min_coverage': 0.05
           }
       
       filtered_rules = []
       
       for rule in rules:
           # Calculate metrics
           support = rm.compute_support(rule.antecedents, antecedents, X)
           confidence = rm.calculate_confidence(rule, antecedents, X, y)
           lift = rm.calculate_lift(rule, antecedents, X, y)
           complexity = len(rule.antecedents)
           
           # Coverage calculation
           rule_mask = np.ones(len(X), dtype=bool)
           for var_idx, term_idx in rule.antecedents:
               memberships = antecedents[var_idx][term_idx].evaluate(X[:, var_idx])
               rule_mask &= (memberships > 0.5)
           coverage = np.mean(rule_mask)
           
           # Apply filters
           if (support >= criteria['min_support'] and
               confidence >= criteria['min_confidence'] and
               lift >= criteria['min_lift'] and
               complexity <= criteria['max_complexity'] and
               coverage >= criteria['min_coverage']):
               
               filtered_rules.append(rule)
       
       return filtered_rules

   # Apply advanced filtering
   high_quality_rules = filter_rules_advanced(
       rules, antecedents, X, y,
       criteria={
           'min_support': 0.15,
           'min_confidence': 0.8,
           'min_lift': 1.5,
           'max_complexity': 2,
           'min_coverage': 0.1
       }
   )

   print(f"Filtered to {len(high_quality_rules)} high-quality rules")

Rule Redundancy Removal
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def remove_redundant_rules(rules, antecedents, X, similarity_threshold=0.8):
       """Remove redundant rules based on antecedent similarity."""
       unique_rules = []
       
       for i, rule1 in enumerate(rules):
           is_redundant = False
           
           for rule2 in unique_rules:
               # Calculate Jaccard similarity of antecedents
               set1 = set(rule1.antecedents)
               set2 = set(rule2.antecedents)
               
               jaccard = len(set1.intersection(set2)) / len(set1.union(set2))
               
               if jaccard > similarity_threshold and rule1.consequent == rule2.consequent:
                   is_redundant = True
                   break
           
           if not is_redundant:
               unique_rules.append(rule1)
       
       return unique_rules

   # Remove redundant rules
   unique_rules = remove_redundant_rules(rules, antecedents, X, similarity_threshold=0.7)
   print(f"Reduced from {len(rules)} to {len(unique_rules)} unique rules")

Visualization and Analysis
-------------------------

Rule Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import matplotlib.pyplot as plt
   import seaborn as sns

   def analyze_rule_distribution(rules):
       """Analyze distribution of rule characteristics."""
       # Extract characteristics
       rule_lengths = [len(r.antecedents) for r in rules]
       rule_weights = [r.weight for r in rules]
       consequents = [r.consequent for r in rules]
       
       # Create visualizations
       fig, axes = plt.subplots(2, 2, figsize=(12, 10))
       
       # Rule length distribution
       axes[0, 0].hist(rule_lengths, bins=range(1, max(rule_lengths)+2), alpha=0.7)
       axes[0, 0].set_xlabel('Number of Antecedents')
       axes[0, 0].set_ylabel('Frequency')
       axes[0, 0].set_title('Rule Complexity Distribution')
       
       # Weight distribution
       axes[0, 1].hist(rule_weights, bins=20, alpha=0.7)
       axes[0, 1].set_xlabel('Rule Weight')
       axes[0, 1].set_ylabel('Frequency')
       axes[0, 1].set_title('Rule Weight Distribution')
       
       # Consequent distribution
       unique_consequents, counts = np.unique(consequents, return_counts=True)
       axes[1, 0].bar(unique_consequents, counts)
       axes[1, 0].set_xlabel('Consequent Class')
       axes[1, 0].set_ylabel('Number of Rules')
       axes[1, 0].set_title('Rules per Class')
       
       # Length vs Weight scatter
       axes[1, 1].scatter(rule_lengths, rule_weights, alpha=0.6)
       axes[1, 1].set_xlabel('Number of Antecedents')
       axes[1, 1].set_ylabel('Rule Weight')
       axes[1, 1].set_title('Complexity vs Quality')
       
       plt.tight_layout()
       plt.show()
       
       return {
           'avg_length': np.mean(rule_lengths),
           'avg_weight': np.mean(rule_weights),
           'class_distribution': dict(zip(unique_consequents, counts))
       }

   # Analyze mined rules
   distribution_stats = analyze_rule_distribution(rules[:50])
   print(f"Average rule length: {distribution_stats['avg_length']:.1f}")
   print(f"Average rule weight: {distribution_stats['avg_weight']:.3f}")
   print(f"Class distribution: {distribution_stats['class_distribution']}")

Variable Importance Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def analyze_variable_importance(rules, feature_names):
       """Analyze which variables appear most frequently in rules."""
       variable_counts = {i: 0 for i in range(len(feature_names))}
       term_counts = {}
       
       for rule in rules:
           for var_idx, term_idx in rule.antecedents:
               variable_counts[var_idx] += 1
               
               if (var_idx, term_idx) not in term_counts:
                   term_counts[(var_idx, term_idx)] = 0
               term_counts[(var_idx, term_idx)] += 1
       
       # Plot variable importance
       plt.figure(figsize=(12, 5))
       
       plt.subplot(1, 2, 1)
       vars_sorted = sorted(variable_counts.items(), key=lambda x: x[1], reverse=True)
       var_names = [feature_names[i] for i, _ in vars_sorted]
       var_counts = [count for _, count in vars_sorted]
       
       plt.bar(range(len(var_names)), var_counts)
       plt.xticks(range(len(var_names)), var_names, rotation=45)
       plt.ylabel('Frequency in Rules')
       plt.title('Variable Importance')
       
       # Plot most common terms
       plt.subplot(1, 2, 2)
       terms_sorted = sorted(term_counts.items(), key=lambda x: x[1], reverse=True)[:10]
       term_labels = [f"{feature_names[var]}_{term}" for (var, term), _ in terms_sorted]
       term_counts_list = [count for _, count in terms_sorted]
       
       plt.bar(range(len(term_labels)), term_counts_list)
       plt.xticks(range(len(term_labels)), term_labels, rotation=45)
       plt.ylabel('Frequency in Rules')
       plt.title('Most Common Terms')
       
       plt.tight_layout()
       plt.show()
       
       return variable_counts, term_counts

   # Analyze variable importance
   var_importance, term_importance = analyze_variable_importance(rules, feature_names)

Integration with Classifiers
----------------------------

Using Mined Rules for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from ex_fuzzy.classifiers import RuleMineClassifier

   # Use mined rules as initial population
   classifier = RuleMineClassifier(
       nRules=30,
       nAnts=3,
       linguistic_variables=antecedents,
       initial_rules=high_quality_rules,  # Use pre-mined rules
       verbose=True
   )

   # Fit classifier (will optimize the mined rules)
   classifier.fit(X, y)

   # Compare performance
   from sklearn.model_selection import cross_val_score

   # Classifier with mined rules
   scores_mined = cross_val_score(classifier, X, y, cv=5)

   # Classifier without mined rules (random initialization)
   classifier_random = RuleMineClassifier(nRules=30, nAnts=3, linguistic_variables=antecedents)
   scores_random = cross_val_score(classifier_random, X, y, cv=5)

   print(f"With mined rules: {scores_mined.mean():.3f} (+/- {scores_mined.std() * 2:.3f})")
   print(f"Random initialization: {scores_random.mean():.3f} (+/- {scores_random.std() * 2:.3f})")

Iterative Rule Refinement
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def iterative_rule_mining(X, y, antecedents, n_iterations=3):
       """Iteratively refine rule mining parameters."""
       best_rules = []
       best_score = 0
       
       # Start with conservative parameters
       support_values = [0.2, 0.15, 0.1, 0.05]
       confidence_values = [0.9, 0.8, 0.7, 0.6]
       
       for iteration in range(n_iterations):
           print(f"\\nIteration {iteration + 1}")
           
           for support in support_values:
               for confidence in confidence_values:
                   # Mine rules with current parameters
                   rules = rm.mine_fuzzy_rules(
                       antecedents=antecedents,
                       X=X, y=y,
                       min_support=support,
                       min_confidence=confidence,
                       min_lift=1.0,
                       max_antecedents=3
                   )
                   
                   if len(rules) == 0:
                       continue
                   
                   # Evaluate rules quickly
                   classifier = RuleMineClassifier(
                       nRules=min(20, len(rules)),
                       nAnts=3,
                       linguistic_variables=antecedents,
                       initial_rules=rules[:20]
                   )
                   
                   try:
                       scores = cross_val_score(classifier, X, y, cv=3)
                       avg_score = scores.mean()
                       
                       if avg_score > best_score:
                           best_score = avg_score
                           best_rules = rules
                           print(f"  New best: support={support}, confidence={confidence}, score={avg_score:.3f}")
                   except:
                       continue
           
           # Adjust parameters for next iteration
           support_values = [s * 0.8 for s in support_values]  # More permissive
       
       return best_rules, best_score

   # Run iterative refinement
   refined_rules, best_score = iterative_rule_mining(X, y, antecedents)
   print(f"\\nFinal best rules: {len(refined_rules)} rules, score: {best_score:.3f}")

Large Dataset Strategies
-----------------------

Sampling-Based Mining
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def sample_based_mining(X, y, antecedents, sample_size=1000, n_samples=5):
       """Mine rules from multiple random samples for large datasets."""
       all_rules = []
       
       for i in range(n_samples):
           # Random sample
           sample_indices = np.random.choice(len(X), size=min(sample_size, len(X)), replace=False)
           X_sample = X[sample_indices]
           y_sample = y[sample_indices]
           
           # Mine rules from sample
           sample_rules = rm.mine_fuzzy_rules(
               antecedents=antecedents,
               X=X_sample,
               y=y_sample,
               min_support=0.1,
               min_confidence=0.6,
               max_antecedents=3
           )
           
           all_rules.extend(sample_rules)
           print(f"Sample {i+1}: {len(sample_rules)} rules")
       
       # Remove duplicates and filter
       unique_rules = remove_redundant_rules(all_rules, antecedents, X)
       
       # Validate on full dataset
       final_rules = filter_rules_advanced(unique_rules, antecedents, X, y)
       
       return final_rules

   # For large datasets
   if len(X) > 10000:
       sampled_rules = sample_based_mining(X, y, antecedents)
   else:
       sampled_rules = rules

Incremental Mining
~~~~~~~~~~~~~~~~~

.. code-block:: python

   class IncrementalRuleMiner:
       """Incrementally update rules as new data arrives."""
       
       def __init__(self, antecedents, min_support=0.1, min_confidence=0.6):
           self.antecedents = antecedents
           self.min_support = min_support
           self.min_confidence = min_confidence
           self.rules = []
           self.seen_data = []
           self.seen_labels = []
       
       def update(self, X_new, y_new):
           """Update rules with new data."""
           # Add new data to history
           if len(self.seen_data) == 0:
               self.seen_data = X_new
               self.seen_labels = y_new
           else:
               self.seen_data = np.vstack([self.seen_data, X_new])
               self.seen_labels = np.concatenate([self.seen_labels, y_new])
           
           # Re-mine rules with updated dataset
           new_rules = rm.mine_fuzzy_rules(
               antecedents=self.antecedents,
               X=self.seen_data,
               y=self.seen_labels,
               min_support=self.min_support,
               min_confidence=self.min_confidence,
               max_antecedents=3
           )
           
           # Update rule base
           self.rules = new_rules
           
           return len(self.rules)
       
       def get_rules(self):
           return self.rules

   # Use incremental miner
   incremental_miner = IncrementalRuleMiner(antecedents)

   # Simulate streaming data
   batch_size = 30
   for i in range(0, len(X), batch_size):
       X_batch = X[i:i+batch_size]
       y_batch = y[i:i+batch_size]
       
       n_rules = incremental_miner.update(X_batch, y_batch)
       print(f"Batch {i//batch_size + 1}: {n_rules} total rules")

Best Practices
--------------

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   def recommend_parameters(X, y):
       """Recommend mining parameters based on dataset characteristics."""
       n_samples, n_features = X.shape
       n_classes = len(np.unique(y))
       
       recommendations = {}
       
       # Support threshold based on dataset size
       if n_samples < 100:
           recommendations['min_support'] = 0.2
       elif n_samples < 1000:
           recommendations['min_support'] = 0.1
       else:
           recommendations['min_support'] = 0.05
       
       # Confidence based on class balance
       class_counts = np.bincount(y)
       balance_ratio = np.min(class_counts) / np.max(class_counts)
       
       if balance_ratio > 0.8:  # Balanced
           recommendations['min_confidence'] = 0.6
       elif balance_ratio > 0.5:  # Slightly imbalanced
           recommendations['min_confidence'] = 0.7
       else:  # Highly imbalanced
           recommendations['min_confidence'] = 0.8
       
       # Complexity based on feature count
       if n_features <= 4:
           recommendations['max_antecedents'] = min(n_features, 3)
       elif n_features <= 10:
           recommendations['max_antecedents'] = 3
       else:
           recommendations['max_antecedents'] = 2
       
       # Lift threshold
       recommendations['min_lift'] = 1.0 + (0.5 / n_classes)  # Harder for more classes
       
       return recommendations

   # Get recommendations
   params = recommend_parameters(X, y)
   print("Recommended parameters:")
   for param, value in params.items():
       print(f"  {param}: {value}")

Common Pitfalls and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: Too many rules**

.. code-block:: python

   # Solution: Increase thresholds or add post-filtering
   conservative_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.15,      # Higher support
       min_confidence=0.8,    # Higher confidence
       min_lift=1.5,          # Higher lift
       max_antecedents=2      # Simpler rules
   )

**Problem: Too few rules**

.. code-block:: python

   # Solution: Relax thresholds or increase partitions
   permissive_rules = rm.mine_fuzzy_rules(
       antecedents=antecedents, X=X, y=y,
       min_support=0.05,      # Lower support
       min_confidence=0.5,    # Lower confidence
       min_lift=0.8,          # Lower lift
       max_antecedents=4      # More complex rules
   )

**Problem: Low-quality rules**

.. code-block:: python

   # Solution: Apply rigorous post-filtering
   quality_filtered = filter_rules_advanced(
       rules, antecedents, X, y,
       criteria={
           'min_support': 0.1,
           'min_confidence': 0.8,
           'min_lift': 1.5,
           'max_complexity': 2,
           'min_coverage': 0.1
       }
   )

Next Steps
----------

After mastering rule mining:

1. **Integration**: Learn to integrate mined rules with evolutionary optimization
2. **Evaluation**: Use comprehensive evaluation tools to assess rule quality
3. **Visualization**: Explore rule visualization techniques
4. **Applications**: Apply to real-world datasets and domains

Related Guides:

- :doc:`evolutionary-optimization` - Optimize mined rules
- :doc:`evaluation-metrics` - Evaluate rule quality
- :doc:`visualization` - Visualize rules and partitions
- :doc:`../examples/index` - Real-world applications
