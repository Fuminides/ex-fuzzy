Evaluation Tools Module
=======================

The :mod:`ex_fuzzy.eval_tools` module provides comprehensive evaluation and analysis tools for fuzzy classification models.

.. currentmodule:: ex_fuzzy.eval_tools

Overview
--------

This module includes performance metrics, statistical analysis, visualization capabilities, and model interpretation tools specifically designed for fuzzy rule-based systems.

**Core Capabilities:**

* **Performance Metrics**: Standard and fuzzy-specific evaluation measures
* **Statistical Analysis**: Bootstrap confidence intervals and significance testing
* **Rule Analysis**: Coverage, importance, and interpretability metrics
* **Visualization Integration**: Rule plotting and partition visualization
* **Model Comparison**: Statistical comparison of different fuzzy models
* **Comprehensive Reporting**: Detailed evaluation reports with statistical significance

**Key Features:**

* Scikit-learn compatible metric evaluation
* Fuzzy-specific measures (rule coverage, dominance scores)
* Bootstrap statistical analysis for robust assessment
* Integration with visualization tools
* Support for multi-class and imbalanced datasets

Classes
-------

.. autosummary::
   :toctree: generated/
   :template: class.rst

   FuzzyEvaluator
   ModelComparator
   RuleAnalyzer

Functions
---------

.. autosummary::
   :toctree: generated/

   eval_fuzzy_model
   compute_performance_metrics
   bootstrap_evaluation
   compare_models
   rule_importance_analysis

Main Functions
--------------

Model Evaluation
~~~~~~~~~~~~~~~~

.. autofunction:: eval_fuzzy_model

Performance Metrics
~~~~~~~~~~~~~~~~~~~

.. autofunction:: compute_performance_metrics

Statistical Analysis
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: bootstrap_evaluation

Core Classes
------------

FuzzyEvaluator
~~~~~~~~~~~~~~

.. autoclass:: FuzzyEvaluator
   :members:
   :inherited-members:
   :show-inheritance:

   **Core Methods**

   .. autosummary::
      :nosignatures:

      ~FuzzyEvaluator.evaluate
      ~FuzzyEvaluator.bootstrap_analysis
      ~FuzzyEvaluator.rule_analysis
      ~FuzzyEvaluator.generate_report

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

Performance Metrics Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detailed performance metrics
   evaluator = et.FuzzyEvaluator(classifier)
   
   metrics = evaluator.compute_performance_metrics(X_test, y_test)
   print(f"Accuracy: {metrics['accuracy']:.3f}")
   print(f"F1-score (macro): {metrics['f1_macro']:.3f}")
   print(f"F1-score (weighted): {metrics['f1_weighted']:.3f}")
   print(f"Matthews Correlation: {metrics['mcc']:.3f}")

   # Class-wise performance
   for class_id, class_metrics in metrics['per_class'].items():
       print(f"\\nClass {class_id}:")
       print(f"  Precision: {class_metrics['precision']:.3f}")
       print(f"  Recall: {class_metrics['recall']:.3f}")
       print(f"  F1-score: {class_metrics['f1']:.3f}")

Bootstrap Statistical Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Bootstrap confidence intervals
   bootstrap_results = evaluator.bootstrap_analysis(
       X_test=X_test,
       y_test=y_test,
       n_bootstrap=1000,
       confidence_level=0.95,
       metrics=['accuracy', 'f1_macro', 'mcc']
   )

   for metric, results in bootstrap_results.items():
       mean_val = results['mean']
       ci_lower = results['ci_lower']
       ci_upper = results['ci_upper']
       print(f"{metric}: {mean_val:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")

Rule Analysis and Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze rule importance and coverage
   rule_analyzer = et.RuleAnalyzer(classifier)
   
   # Rule coverage analysis
   coverage_stats = rule_analyzer.compute_rule_coverage(X_test, y_test)
   
   print("Rule Coverage Analysis:")
   for rule_id, stats in coverage_stats.items():
       print(f"Rule {rule_id}:")
       print(f"  Coverage: {stats['coverage']:.3f}")
       print(f"  Accuracy: {stats['accuracy']:.3f}")
       print(f"  Samples covered: {stats['n_samples']}")

   # Rule importance ranking
   importance_scores = rule_analyzer.rule_importance(X_test, y_test)
   
   print("\\nRule Importance Ranking:")
   for rule_id, importance in sorted(importance_scores.items(), 
                                   key=lambda x: x[1], reverse=True):
       print(f"Rule {rule_id}: {importance:.3f}")

Model Comparison
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compare multiple models
   from sklearn.ensemble import RandomForestClassifier
   
   # Train baseline models
   rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
   rf_classifier.fit(X_train, y_train)
   
   fuzzy_classifier2 = RuleMineClassifier(nRules=25, nAnts=4)
   fuzzy_classifier2.fit(X_train, y_train)
   
   # Compare models
   comparator = et.ModelComparator()
   
   comparison_results = comparator.compare_models(
       models={
           'Fuzzy_15_rules': classifier,
           'Fuzzy_25_rules': fuzzy_classifier2,
           'Random_Forest': rf_classifier
       },
       X_test=X_test,
       y_test=y_test,
       metrics=['accuracy', 'f1_macro', 'mcc'],
       statistical_test=True
   )
   
   # Display comparison
   comparator.display_comparison(comparison_results)

Advanced Evaluation Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive evaluation workflow
   class FuzzyModelEvaluator:
       def __init__(self, classifier):
           self.classifier = classifier
           self.evaluator = et.FuzzyEvaluator(classifier)
           self.rule_analyzer = et.RuleAnalyzer(classifier)
       
       def full_evaluation(self, X_train, y_train, X_test, y_test):
           results = {}
           
           # Basic performance
           results['performance'] = self.evaluator.compute_performance_metrics(X_test, y_test)
           
           # Bootstrap analysis
           results['bootstrap'] = self.evaluator.bootstrap_analysis(
               X_test, y_test, n_bootstrap=500
           )
           
           # Rule analysis
           results['rule_coverage'] = self.rule_analyzer.compute_rule_coverage(X_test, y_test)
           results['rule_importance'] = self.rule_analyzer.rule_importance(X_test, y_test)
           
           # Model complexity
           results['complexity'] = {
               'n_rules': len(self.classifier.get_rules()),
               'avg_rule_length': np.mean([len(rule.antecedents) 
                                         for rule in self.classifier.get_rules()]),
               'total_conditions': sum(len(rule.antecedents) 
                                     for rule in self.classifier.get_rules())
           }
           
           return results
       
       def generate_report(self, results):
           report = []
           report.append("=== Fuzzy Model Evaluation Report ===\\n")
           
           # Performance section
           perf = results['performance']
           report.append(f"Accuracy: {perf['accuracy']:.4f}")
           report.append(f"F1-score (macro): {perf['f1_macro']:.4f}")
           report.append(f"Matthews Correlation: {perf['mcc']:.4f}\\n")
           
           # Bootstrap confidence intervals
           bootstrap = results['bootstrap']
           report.append("Bootstrap Confidence Intervals (95%):")
           for metric, stats in bootstrap.items():
               ci_str = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
               report.append(f"  {metric}: {stats['mean']:.3f} {ci_str}")
           
           # Model complexity
           comp = results['complexity']
           report.append(f"\\nModel Complexity:")
           report.append(f"  Number of rules: {comp['n_rules']}")
           report.append(f"  Average rule length: {comp['avg_rule_length']:.1f}")
           report.append(f"  Total conditions: {comp['total_conditions']}")
           
           return "\\n".join(report)

   # Use advanced evaluator
   evaluator = FuzzyModelEvaluator(classifier)
   results = evaluator.full_evaluation(X_train, y_train, X_test, y_test)
   report = evaluator.generate_report(results)
   print(report)

Cross-Validation Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import cross_validate, StratifiedKFold

   # Cross-validation evaluation
   def cross_validate_fuzzy_model(classifier, X, y, cv=5):
       cv_results = {}
       skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
       
       # Standard cross-validation
       scoring = ['accuracy', 'f1_macro', 'precision_macro', 'recall_macro']
       cv_scores = cross_validate(classifier, X, y, cv=skf, scoring=scoring)
       
       # Aggregate results
       for metric in scoring:
           scores = cv_scores[f'test_{metric}']
           cv_results[metric] = {
               'mean': np.mean(scores),
               'std': np.std(scores),
               'scores': scores
           }
       
       return cv_results

   # Run cross-validation
   cv_results = cross_validate_fuzzy_model(
       RuleMineClassifier(nRules=20, nAnts=3),
       X, y, cv=5
   )

   # Display results
   print("Cross-Validation Results:")
   for metric, stats in cv_results.items():
       mean_score = stats['mean']
       std_score = stats['std']
       print(f"{metric}: {mean_score:.3f} (+/- {std_score * 2:.3f})")

Visualization Integration
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Evaluation with visualization
   def evaluate_with_plots(classifier, X_train, y_train, X_test, y_test):
       # Generate evaluation report
       report = et.eval_fuzzy_model(
           fl_classifier=classifier,
           X_train=X_train,
           y_train=y_train,
           X_test=X_test,
           y_test=y_test,
           plot_rules=True,        # Plot rule structure
           plot_partitions=True,   # Plot fuzzy partitions
           print_rules=True,       # Print rule descriptions
           bootstrap_results_print=True
       )
       
       # Additional custom plots
       import matplotlib.pyplot as plt
       
       # Prediction confidence distribution
       y_proba = classifier.predict_proba(X_test)
       max_proba = np.max(y_proba, axis=1)
       
       plt.figure(figsize=(10, 6))
       plt.subplot(1, 2, 1)
       plt.hist(max_proba, bins=20, alpha=0.7)
       plt.xlabel('Maximum Prediction Confidence')
       plt.ylabel('Frequency')
       plt.title('Prediction Confidence Distribution')
       
       # Confusion matrix
       from sklearn.metrics import confusion_matrix
       import seaborn as sns
       
       y_pred = classifier.predict(X_test)
       cm = confusion_matrix(y_test, y_pred)
       
       plt.subplot(1, 2, 2)
       sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
       plt.xlabel('Predicted')
       plt.ylabel('Actual')
       plt.title('Confusion Matrix')
       
       plt.tight_layout()
       plt.show()
       
       return report

   # Run evaluation with plots
   report = evaluate_with_plots(classifier, X_train, y_train, X_test, y_test)

Performance Monitoring
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Monitor performance over time or different datasets
   class PerformanceMonitor:
       def __init__(self):
           self.history = []
       
       def log_performance(self, classifier, X_test, y_test, dataset_name):
           evaluator = et.FuzzyEvaluator(classifier)
           metrics = evaluator.compute_performance_metrics(X_test, y_test)
           
           entry = {
               'timestamp': pd.Timestamp.now(),
               'dataset': dataset_name,
               'accuracy': metrics['accuracy'],
               'f1_macro': metrics['f1_macro'],
               'mcc': metrics['mcc'],
               'n_rules': len(classifier.get_rules())
           }
           
           self.history.append(entry)
       
       def get_performance_trend(self):
           df = pd.DataFrame(self.history)
           return df.groupby('dataset').agg({
               'accuracy': ['mean', 'std'],
               'f1_macro': ['mean', 'std'],
               'mcc': ['mean', 'std']
           })
       
       def plot_performance_trend(self):
           df = pd.DataFrame(self.history)
           
           plt.figure(figsize=(12, 8))
           for i, metric in enumerate(['accuracy', 'f1_macro', 'mcc']):
               plt.subplot(2, 2, i+1)
               for dataset in df['dataset'].unique():
                   dataset_data = df[df['dataset'] == dataset]
                   plt.plot(dataset_data['timestamp'], dataset_data[metric], 
                           marker='o', label=dataset)
               
               plt.xlabel('Time')
               plt.ylabel(metric.replace('_', ' ').title())
               plt.legend()
               plt.xticks(rotation=45)
           
           plt.tight_layout()
           plt.show()

   # Use performance monitor
   monitor = PerformanceMonitor()
   monitor.log_performance(classifier, X_test, y_test, 'iris_test')

Available Metrics
-----------------

Standard Classification Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``accuracy``
     - Overall classification accuracy
   * - ``balanced_accuracy``
     - Balanced accuracy for imbalanced datasets
   * - ``f1_macro``
     - Macro-averaged F1 score
   * - ``f1_micro``
     - Micro-averaged F1 score
   * - ``f1_weighted``
     - Weighted F1 score by class support
   * - ``precision_macro``
     - Macro-averaged precision
   * - ``recall_macro``
     - Macro-averaged recall
   * - ``mcc``
     - Matthews Correlation Coefficient

Fuzzy-Specific Metrics
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Metric
     - Description
   * - ``rule_coverage``
     - Average rule coverage across dataset
   * - ``rule_dominance``
     - Average rule dominance scores
   * - ``model_complexity``
     - Number of rules and conditions
   * - ``interpretability_score``
     - Combined interpretability measure

See Also
--------

* :mod:`ex_fuzzy.classifiers` : Fuzzy classification algorithms
* :mod:`ex_fuzzy.rules` : Rule representation and inference
* :mod:`ex_fuzzy.vis_rules` : Rule visualization utilities
* :mod:`sklearn.metrics` : Standard classification metrics

References
----------

.. [1] Sokolova, M., and G. Lapalme. "A systematic analysis of performance measures for classification tasks." 
       Information Processing & Management 45.4 (2009): 427-437.

.. [2] Matthews, B.W. "Comparison of the predicted and observed secondary structure of T4 phage lysozyme." 
       Biochimica et Biophysica Acta 405.2 (1975): 442-451.

.. [3] Efron, B., and R.J. Tibshirani. "An Introduction to the Bootstrap." CRC Press, 1994.
