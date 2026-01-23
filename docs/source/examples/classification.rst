Classification Examples
=======================

This section provides comprehensive examples of using ex-fuzzy for classification tasks, from basic usage to advanced techniques.

.. contents:: Table of Contents
   :local:
   :depth: 2

Basic Classification
--------------------

Iris Classification
~~~~~~~~~~~~~~~~~~~

A complete walkthrough using the classic Iris dataset:

.. code-block:: python

   import pandas as pd
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
   
   import ex_fuzzy.fuzzy_sets as fs
   import ex_fuzzy.classifiers as clf
   import ex_fuzzy.utils as utils
   import ex_fuzzy.eval_tools as eval_tools

   # Load the Iris dataset
   iris = load_iris()
   X = pd.DataFrame(iris.data, columns=iris.feature_names)
   y = iris.target
   feature_names = iris.feature_names
   class_names = iris.target_names

   print(f"Dataset shape: {X.shape}")
   print(f"Classes: {class_names}")
   print(f"Features: {feature_names}")

   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.3, random_state=42, stratify=y
   )

   # Compute linguistic variables (fuzzy partitions) for each feature
   precomputed_partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

   # Create and train the rule-mining fuzzy classifier
   classifier = clf.RuleMineClassifier(
       nRules=15,                                    # Number of rules to optimize
       nAnts=4,                                      # Maximum antecedents per rule
       fuzzy_type=fs.FUZZY_SETS.t1,                 # Type-1 fuzzy sets
       tolerance=0.01,                              # Tolerance for rule dominance
       linguistic_variables=precomputed_partitions  # Precomputed fuzzy partitions
   )

   # Fit the classifier with genetic algorithm parameters
   print("Training fuzzy classifier...")
   classifier.fit(X_train, y_train, n_gen=30, pop_size=50)

   # Make predictions
   y_pred = classifier.predict(X_test)

   # Evaluate performance
   accuracy = accuracy_score(y_test, y_pred)
   print(f"\\nTest Accuracy: {accuracy:.3f}")

   # Detailed evaluation
   print("\\nClassification Report:")
   print(classification_report(y_test, y_pred, target_names=class_names))

   # Confusion matrix
   cm = confusion_matrix(y_test, y_pred)
   print("\\nConfusion Matrix:")
   print(pd.DataFrame(cm, columns=class_names, index=class_names))

Expected output:

.. code-block:: text

   Dataset shape: (150, 4)
   Classes: ['setosa' 'versicolor' 'virginica']
   Features: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
   
   Training fuzzy classifier...
   Test Accuracy: 0.956
   
   Classification Report:
                precision    recall  f1-score   support
        setosa       1.00      1.00      1.00        15
    versicolor       0.93      0.93      0.93        15
     virginica       0.93      0.93      0.93        15
   
      accuracy                           0.96        45
     macro avg       0.95      0.95      0.95        45
  weighted avg       0.96      0.96      0.96        45

Custom Fuzzy Variables
Advanced Usage
~~~~~~~~~~~~~~

Using the comprehensive evaluation function:

.. code-block:: python

   # Use the comprehensive evaluation function
   evaluation_report = eval_tools.eval_fuzzy_model(
       classifier, X_train, y_train, X_test, y_test,
       plot_rules=True,           # Generate rule plots
       print_rules=True,          # Print rule text
       plot_partitions=True,      # Plot fuzzy partitions
       return_rules=True,         # Return rule text
       bootstrap_results_print=True  # Statistical analysis
   )
   
   print(evaluation_report)

   # Or use the FuzzyEvaluator class directly for more control
   evaluator = eval_tools.FuzzyEvaluator(classifier)
   
   # Get specific metrics
   accuracy = evaluator.get_metric('accuracy_score', X_test, y_test)
   f1_score = evaluator.get_metric('f1_score', X_test, y_test, average='weighted')
   precision = evaluator.get_metric('precision_score', X_test, y_test, average='weighted')
   recall = evaluator.get_metric('recall_score', X_test, y_test, average='weighted')
   
   print(f"Accuracy: {accuracy:.3f}")
   print(f"F1-Score: {f1_score:.3f}")
   print(f"Precision: {precision:.3f}")
   print(f"Recall: {recall:.3f}")

Alternative Classifiers
~~~~~~~~~~~~~~~~~~~~~~~

Ex-fuzzy provides multiple classifier options:

.. code-block:: python

   # Two-stage classifier with rule mining and optimization
   fuzzy_classifier = clf.FuzzyRulesClassifier(
       nRules=20,
       nAnts=4,
       fuzzy_type=fs.FUZZY_SETS.t1,
       tolerance=0.01,
       linguistic_variables=precomputed_partitions
   )
   fuzzy_classifier.fit(X_train, y_train, n_gen=40, pop_size=60)

   # Rule fine-tuning classifier
   fine_tune_classifier = clf.RuleFineTuneClassifier(
       nRules=15,
       nAnts=3,
       fuzzy_type=fs.FUZZY_SETS.t1,
       tolerance=0.01,
       linguistic_variables=precomputed_partitions
   )
   fine_tune_classifier.fit(X_train, y_train, n_gen=30, pop_size=50)

   # Strategy 2: SMOTE + Fuzzy classifier
   smote = SMOTE(random_state=42)
   X_train_smote, y_train_smote = smote.fit_resample(X_train_imb, y_train_imb)

   smote_classifier = clf.RuleMineClassifier(
       nRules=25,
       nAnts=3,
       verbose=False
   )

   smote_classifier.fit(X_train_smote, y_train_smote)
   smote_score = smote_classifier.score(X_test_imb, y_test_imb)

   print(f"Weighted classifier accuracy: {weighted_score:.3f}")
   print(f"SMOTE + Fuzzy accuracy: {smote_score:.3f}")

   # Compare with balanced accuracy
   from sklearn.metrics import balanced_accuracy_score

   weighted_balanced = balanced_accuracy_score(
       y_test_imb, weighted_classifier.predict(X_test_imb)
   )
   smote_balanced = balanced_accuracy_score(
       y_test_imb, smote_classifier.predict(X_test_imb)
   )

   print(f"Weighted balanced accuracy: {weighted_balanced:.3f}")
   print(f"SMOTE balanced accuracy: {smote_balanced:.3f}")

Multi-Class Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

Working with datasets having many classes:

.. code-block:: python

   from sklearn.datasets import load_digits

   # Load digits dataset (10 classes)
   digits = load_digits()

Example Output
~~~~~~~~~~~~~~

Expected performance on the Iris dataset:

.. code-block:: text

   Training fuzzy classifier...
   ------------
   ACCURACY
   Train performance: 0.943
   Test performance: 0.956
   ------------
   MATTHEW CORRCOEF
   Train performance: 0.914
   Test performance: 0.933
   ------------

   Rule 1: IF petal length (cm) is low AND petal width (cm) is low THEN setosa
   Rule 2: IF petal length (cm) is medium AND petal width (cm) is medium THEN versicolor
   Rule 3: IF petal length (cm) is high AND petal width (cm) is high THEN virginica
   ...

Working with Different Fuzzy Set Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ex-fuzzy supports different types of fuzzy sets:

.. code-block:: python

   # Type-1 fuzzy sets (default)
   classifier_t1 = clf.RuleMineClassifier(
       nRules=15,
       nAnts=4,
       fuzzy_type=fs.FUZZY_SETS.t1,  # Type-1 fuzzy sets
       tolerance=0.01
   )

   # Type-2 fuzzy sets
   classifier_t2 = clf.RuleMineClassifier(
       nRules=15,
       nAnts=4,
       fuzzy_type=fs.FUZZY_SETS.t2,  # Type-2 fuzzy sets
       tolerance=0.01
   )

   # General Type-2 fuzzy sets
   classifier_gt2 = clf.RuleMineClassifier(
       nRules=15,
       nAnts=4,
       fuzzy_type=fs.FUZZY_SETS.gt2,  # General Type-2 fuzzy sets
       tolerance=0.01
   )

Direct Use of BaseFuzzyRulesClassifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For more direct control, you can use the underlying classifier:

.. code-block:: python

   import ex_fuzzy.evolutionary_fit as evf

   # Create linguistic variables manually
   precomputed_partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

   # Create the base classifier directly
   base_classifier = evf.BaseFuzzyRulesClassifier(
       nRules=15,
       linguistic_variables=precomputed_partitions,
       nAnts=4,
       n_linguistic_variables=3,
       fuzzy_type=fs.FUZZY_SETS.t1,
       verbose=True,
       tolerance=0.01
   )

   # Fit with genetic algorithm parameters
   base_classifier.fit(X_train, y_train, n_gen=50, pop_size=30)

   # Evaluate
   y_pred = base_classifier.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Base classifier accuracy: {accuracy:.3f}")

   # Print and visualize rules
   rule_text = base_classifier.print_rules(return_rules=True)
   print(rule_text)

   # Plot fuzzy variables
   base_classifier.plot_fuzzy_variables()

Available Evaluation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

Ex-fuzzy provides comprehensive evaluation capabilities:

.. code-block:: python

   # Create evaluator
   evaluator = eval_tools.FuzzyEvaluator(classifier)

   # Get individual metrics
   metrics_to_compute = [
       'accuracy_score',
       'precision_score',
       'recall_score', 
       'f1_score',
       'matthews_corrcoef'
   ]

   for metric in metrics_to_compute:
       if metric in ['precision_score', 'recall_score', 'f1_score']:
           score = evaluator.get_metric(metric, X_test, y_test, average='weighted')
       else:
           score = evaluator.get_metric(metric, X_test, y_test)
       print(f"{metric}: {score:.3f}")
   print("IF age is elderly AND cholesterol is high THEN high_risk")
   print("IF blood_pressure is high AND heart_rate is low THEN high_risk")
   print("IF age is young AND cholesterol is normal THEN low_risk")

Customer Segmentation
~~~~~~~~~~~~~~~~~~~~~

Business application for customer classification:

.. code-block:: python

   def create_customer_data():
       """Create synthetic customer dataset."""
       np.random.seed(42)
       n_customers = 1000
       
       # Customer features
       age = np.random.normal(40, 15, n_customers)
       income = np.random.lognormal(10, 0.5, n_customers)
       spending_score = np.random.normal(50, 20, n_customers)
       tenure = np.random.exponential(3, n_customers)
       
       # Create customer segments based on business logic
       segments = []
       for i in range(n_customers):
           if income[i] > 75000 and spending_score[i] > 60:
               segments.append(2)  # Premium
           elif income[i] > 50000 and spending_score[i] > 40:
               segments.append(1)  # Standard
           else:
               segments.append(0)  # Basic
       
       X = np.column_stack([age, income/1000, spending_score, tenure])  # Scale income
       feature_names = ['age', 'income_k', 'spending_score', 'tenure_years']
       
       return X, np.array(segments), feature_names

   # Create customer data
   X_cust, y_cust, cust_features = create_customer_data()
   segment_names = ['Basic', 'Standard', 'Premium']

   print(f"Customer dataset: {X_cust.shape}")
   print(f"Segment distribution: {np.bincount(y_cust)}")

   # Split data
   X_train_cust, X_test_cust, y_train_cust, y_test_cust = train_test_split(
       X_cust, y_cust, test_size=0.3, random_state=42, stratify=y_cust
   )

   # Create business-relevant fuzzy variables
   age_segments = fs.fuzzyVariable(
       "age", X_train_cust[:, 0], 4, fs.FUZZY_SETS.t1,
       terms=['young', 'adult', 'middle_aged', 'senior']
   )

   income_segments = fs.fuzzyVariable(
       "income", X_train_cust[:, 1], 4, fs.FUZZY_SETS.t1,
       terms=['low', 'medium', 'high', 'very_high']
   )

   spending_segments = fs.fuzzyVariable(
       "spending", X_train_cust[:, 2], 3, fs.FUZZY_SETS.t1,
       terms=['low_spender', 'moderate_spender', 'high_spender']
   )

   tenure_segments = fs.fuzzyVariable(
       "tenure", X_train_cust[:, 3], 3, fs.FUZZY_SETS.t1,
       terms=['new', 'established', 'loyal']
   )

   business_variables = [age_segments, income_segments, spending_segments, tenure_segments]

   # Train business classifier
   business_classifier = clf.RuleMineClassifier(
       nRules=20,
       nAnts=3,
       linguistic_variables=business_variables,
       verbose=True
   )

   business_classifier.fit(X_train_cust, y_train_cust)

   # Evaluate business classifier
   cust_accuracy = business_classifier.score(X_test_cust, y_test_cust)
   print(f"Customer segmentation accuracy: {cust_accuracy:.3f}")

   y_pred_cust = business_classifier.predict(X_test_cust)
   print("\\nBusiness Classification Report:")
   print(classification_report(y_test_cust, y_pred_cust, target_names=segment_names))

   # Business insights from rules (example interpretations)
   print("\\nBusiness Insights from Fuzzy Rules:")
   print("- High income + High spending → Premium segment")
   print("- Long tenure + Moderate spending → Standard segment")  
   print("- Young age + Low income → Basic segment")
   print("- Senior age + High income → Premium segment")

Model Interpretation and Analysis
---------------------------------

Rule Analysis
~~~~~~~~~~~~~

Understanding what the model learned:

.. code-block:: python

   # Comprehensive model evaluation
   comprehensive_report = eval_tools.eval_fuzzy_model(
       fl_classifier=classifier,
       X_train=X_train,
       y_train=y_train,
       X_test=X_test,
       y_test=y_test,
       plot_rules=True,
       print_rules=True,
       plot_partitions=True,
       bootstrap_results_print=True
   )

   print(comprehensive_report)

Feature Importance
~~~~~~~~~~~~~~~~~~

Analyzing which features are most important:

.. code-block:: python

   def analyze_feature_importance(classifier, feature_names):
       """Analyze feature importance in fuzzy rules."""
       rules = classifier.get_rules()
       feature_usage = {name: 0 for name in feature_names}
       
       for rule in rules:
           for var_idx, term_idx in rule.antecedents:
               feature_usage[feature_names[var_idx]] += 1
       
       # Normalize by total number of rule conditions
       total_conditions = sum(feature_usage.values())
       importance_scores = {
           name: count / total_conditions 
           for name, count in feature_usage.items()
       }
       
       return importance_scores

   # Analyze feature importance
   importance = analyze_feature_importance(classifier, feature_names)
   
   print("Feature Importance in Fuzzy Rules:")
   for feature, score in sorted(importance.items(), key=lambda x: x[1], reverse=True):
       print(f"  {feature}: {score:.3f}")

Prediction Explanation
~~~~~~~~~~~~~~~~~~~~~~

Explaining individual predictions:

.. code-block:: python

   def explain_prediction(classifier, X_sample, feature_names, class_names):
       """Explain a single prediction."""
       # Get prediction and probability
       prediction = classifier.predict([X_sample])[0]
       probabilities = classifier.predict_proba([X_sample])[0]
       
       print(f"Input: {dict(zip(feature_names, X_sample))}")
       print(f"Predicted class: {class_names[prediction]} (confidence: {probabilities[prediction]:.3f})")
       print(f"All probabilities: {dict(zip(class_names, probabilities))}")
       
       # Get fired rules (this would require access to the rule firing mechanism)
       # This is a simplified example - actual implementation would depend on classifier internals
       print("\\nMost relevant rules (conceptual):")
       print(f"- IF {feature_names[2]} is high AND {feature_names[3]} is wide THEN {class_names[prediction]}")
       
       return prediction, probabilities

   # Explain a few test samples
   for i in range(3):
       print(f"\\n--- Sample {i+1} ---")
       explain_prediction(classifier, X_test[i], feature_names, class_names)

Cross-Validation and Robustness
-------------------------------

Robust Evaluation
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.model_selection import StratifiedKFold, cross_validate

   # Define multiple metrics
   scoring = {
       'accuracy': 'accuracy',
       'f1_macro': 'f1_macro',
       'f1_weighted': 'f1_weighted',
       'precision_macro': 'precision_macro',
       'recall_macro': 'recall_macro'
   }

   # Perform comprehensive cross-validation
   cv_results = cross_validate(
       classifier, X, y,
       cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
       scoring=scoring,
       return_train_score=True,
       n_jobs=-1
   )

   # Display results
   print("Cross-Validation Results:")
   for metric in scoring.keys():
       test_scores = cv_results[f'test_{metric}']
       train_scores = cv_results[f'train_{metric}']
       
       print(f"{metric}:")
       print(f"  Test:  {test_scores.mean():.3f} (+/- {test_scores.std() * 2:.3f})")
       print(f"  Train: {train_scores.mean():.3f} (+/- {train_scores.std() * 2:.3f})")

Model Comparison
~~~~~~~~~~~~~~~~

Comparing fuzzy classifier with other methods:

.. code-block:: python

   from sklearn.ensemble import RandomForestClassifier
   from sklearn.svm import SVC
   from sklearn.naive_bayes import GaussianNB
   from sklearn.linear_model import LogisticRegression

   # Define models to compare
   models = {
       'Fuzzy Classifier': clf.RuleMineClassifier(nRules=15, nAnts=3, verbose=False),
       'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
       'SVM': SVC(random_state=42, probability=True),
       'Naive Bayes': GaussianNB(),
       'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
   }

   # Compare models
   results = {}
   for name, model in models.items():
       scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
       results[name] = scores
       print(f"{name}: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")

   # Statistical significance testing
   from scipy import stats

   fuzzy_scores = results['Fuzzy Classifier']
   for name, scores in results.items():
       if name != 'Fuzzy Classifier':
           statistic, p_value = stats.ttest_rel(fuzzy_scores, scores)
           significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
           print(f"Fuzzy vs {name}: p-value = {p_value:.4f} {significance}")

Performance Optimization
------------------------

Efficient Training
~~~~~~~~~~~~~~~~~~

Tips for faster training on large datasets:

.. code-block:: python

   # For large datasets, consider these optimizations:

   # 1. Reduce feature dimensionality
   from sklearn.feature_selection import SelectKBest, f_classif

   selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
   X_selected = selector.fit_transform(X, y)
   
   # 2. Use fewer fuzzy partitions
   efficient_classifier = clf.RuleMineClassifier(
       nRules=10,           # Fewer rules
       nAnts=2,             # Simpler rules
       fuzzy_partitions=3,  # Fewer partitions per variable
       verbose=False
   )

   # 3. Sample-based training for very large datasets
   if len(X) > 10000:
       sample_size = 5000
       sample_indices = np.random.choice(len(X), sample_size, replace=False)
       X_sample = X[sample_indices]
       y_sample = y[sample_indices]
       
       efficient_classifier.fit(X_sample, y_sample)
   else:
       efficient_classifier.fit(X, y)

Memory-Efficient Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For memory-constrained environments
   def batch_predict(classifier, X, batch_size=1000):
       """Predict in batches to save memory."""
       predictions = []
       
       for i in range(0, len(X), batch_size):
           batch = X[i:i+batch_size]
           batch_pred = classifier.predict(batch)
           predictions.extend(batch_pred)
       
       return np.array(predictions)

   # Use batch prediction for large test sets
   if len(X_test) > 5000:
       y_pred_batch = batch_predict(classifier, X_test, batch_size=500)
   else:
       y_pred_batch = classifier.predict(X_test)

Best Practices Summary
----------------------

1. **Data Preparation**
   - Normalize features if they have different scales
   - Handle missing values appropriately
   - Consider feature selection for high-dimensional data

2. **Model Configuration**
   - Start with 3-5 fuzzy partitions per variable
   - Use 10-30 rules depending on problem complexity
   - Limit rule antecedents to 2-4 for interpretability

3. **Evaluation**
   - Always use cross-validation for robust estimates
   - Consider multiple metrics (accuracy, F1, precision, recall)
   - Use stratified splits for imbalanced data

4. **Interpretability**
   - Design meaningful linguistic variable names
   - Analyze rule importance and coverage
   - Visualize fuzzy partitions and rules

5. **Performance**
   - Use hyperparameter tuning for optimal results
   - Consider ensemble methods for complex problems
   - Monitor training time vs. accuracy trade-offs

Next Steps
----------

After mastering basic classification:

1. **Advanced Techniques**: Explore Type-2 and General Type-2 fuzzy sets
2. **Ensemble Methods**: Combine multiple fuzzy classifiers
3. **Online Learning**: Adapt classifiers to streaming data
4. **Domain Applications**: Apply to specific domains (medical, financial, etc.)

Related Examples:

- :doc:`regression` - Fuzzy regression techniques
- :doc:`../user-guide/validation-visualization` - Validation and visualization workflows
- :doc:`../api/rule_mining` - Rule mining utilities
- :doc:`../evox_backend` - GPU-accelerated optimization
