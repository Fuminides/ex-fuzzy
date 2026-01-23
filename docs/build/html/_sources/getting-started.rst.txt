===============
Getting Started
===============

Welcome to Ex-Fuzzy! This guide will help you get up and running with fuzzy logic classification in just a few minutes.

What is Ex-Fuzzy?
==================

Ex-Fuzzy is a Python library designed for building explainable fuzzy rule-based classifiers. Unlike traditional "black box" machine learning models, Ex-Fuzzy generates human-readable fuzzy rules that clearly explain how decisions are made.

.. note::
    **Why Choose Ex-Fuzzy?**
    
    - 🔍 **Explainable**: Generate interpretable fuzzy rules
    - ⚡ **Fast**: Optimized for performance with multiprocessing support
    - 🎯 **Accurate**: Competitive classification performance
    - 📊 **Visual**: Rich visualization capabilities
    - 🔧 **Flexible**: Highly customizable fuzzy systems

Installation
============

Install Ex-Fuzzy using pip:

.. code-block:: bash

    pip install ex-fuzzy

Or from source:

.. code-block:: bash

    git clone https://github.com/fuminides/ex-fuzzy.git
    cd ex-fuzzy
    pip install -e .

Dependencies
------------

Ex-Fuzzy requires:

- Python 3.8+
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- PyMoo (for evolutionary optimization)

Your First Fuzzy Classifier
===========================

Let's start with a simple example using the famous Iris dataset:

.. code-block:: python

    import ex_fuzzy.evolutionary_fit as evf
    import ex_fuzzy.fuzzy_sets as fs
    import ex_fuzzy.utils as utils
    import ex_fuzzy.eval_tools as eval_tools
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42
    )

    # Create linguistic variables (fuzzy partitions) from the data
    linguistic_variables = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)

    # Create a fuzzy classifier
    classifier = evf.BaseFuzzyRulesClassifier(
        nRules=15,                            # Number of rules to evolve
        nAnts=4,                              # Maximum antecedents per rule
        linguistic_variables=linguistic_variables,  # Pre-computed fuzzy variables
        fuzzy_type=fs.FUZZY_SETS.t1,         # Type-1 fuzzy sets
        verbose=True                          # Show training progress
    )

    # Train the classifier
    classifier.fit(X_train, y_train, n_gen=30, pop_size=50)

    # Make predictions
    y_pred = classifier.predict(X_test)
    
    # Calculate accuracy manually since score method requires X input too
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

Understanding the Output
========================

After training, you can examine the fuzzy rules that were learned:

.. code-block:: python

    # Print the learned rules
    rule_text = classifier.print_rules(return_rules=True)
    print(rule_text)

    # Use comprehensive evaluation
    evaluation_report = eval_tools.eval_fuzzy_model(
        classifier, X_train, y_train, X_test, y_test,
        plot_rules=True,      # Show rule structure  
        print_rules=True,     # Print rules in readable format
        plot_partitions=True, # Show fuzzy partitions
        return_rules=True     # Return rule text
    )

Example Output
--------------

Your fuzzy rules might look like this:

.. code-block:: text

    Rule 1: IF sepal length (cm) is Low AND petal width (cm) is Low THEN setosa (DS: 0.85)
    Rule 2: IF petal length (cm) is High AND petal width (cm) is High THEN virginica (DS: 0.92)
    Rule 3: IF petal length (cm) is Medium AND sepal width (cm) is High THEN versicolor (DS: 0.78)

Where DS is the Dominance Score indicating rule quality.

Key Concepts
============

Fuzzy Variables
---------------

Fuzzy variables define how crisp values are mapped to linguistic terms like "Low", "Medium", "High":

.. code-block:: python

    import ex_fuzzy.fuzzy_sets as fs
    import ex_fuzzy.utils as utils
    import numpy as np
    
    # Create sample data
    data = np.array([[4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5]]).T
    
    # Create fuzzy variables automatically from data
    fuzzy_vars = utils.construct_partitions(data, fs.FUZZY_SETS.t1, n_partitions=3)
    
    # This creates linguistic variables with terms like "Low", "Medium", "High"

Fuzzy Rules
-----------

Rules combine multiple conditions using fuzzy logic:

.. code-block:: python

    # A rule has antecedents (IF part) and consequent (THEN part)
    # IF sepal_length is Low AND petal_width is Low THEN class is setosa

Classification Process
----------------------

1. **Fuzzification**: Convert crisp inputs to fuzzy values
2. **Rule Evaluation**: Calculate rule activation strengths
3. **Aggregation**: Combine evidence from all rules
4. **Defuzzification**: Convert fuzzy output to crisp classification

Advanced Features
=================

Type-2 Fuzzy Sets
-----------------

For handling uncertainty in fuzzy membership:

.. code-block:: python

    classifier = evf.BaseFuzzyRulesClassifier(
        nRules=10,
        fuzzy_type=evf.fs.FUZZY_SETS.t2,  # Type-2 fuzzy sets
        verbose=True
    )

Pattern Stability Analysis
--------------------------

Analyze the consistency of discovered patterns:

.. code-block:: python

    import ex_fuzzy.pattern_stability as ps
    
    # Analyze pattern stability across multiple runs
    stabilizer = ps.pattern_stabilizer(X, y)
    stabilizer.stability_report(n=20, n_gen=30, pop_size=50)

Custom Fuzzy Sets
-----------------

Define your own membership functions:

.. code-block:: python

   # Create custom Gaussian fuzzy sets
   custom_var = fs.fuzzyVariable(
       name="custom_feature",
       fuzzy_sets=[
           fs.gaussianFS("Very Low", [1.0, 0.8], [0, 10]),
           fs.gaussianFS("Low", [3.0, 0.8], [0, 10]),
           fs.gaussianFS("Medium", [5.0, 0.8], [0, 10]),
           fs.gaussianFS("High", [7.0, 0.8], [0, 10]),
           fs.gaussianFS("Very High", [9.0, 0.8], [0, 10]),
       ],
   )


Next Steps
==========

Now that you've got the basics, explore more advanced features:

- :doc:`user-guide/core-concepts`: Learn about different types of fuzzy sets
- :doc:`api/evolutionary_fit`: Understand the evolutionary optimization process
- :doc:`user-guide/validation-visualization`: Create visualizations of your models
- :doc:`user-guide/results-io`: Save rules, variables, and plots to files
- :doc:`user-guide/recipes`: Quick workflows you can reuse
- :doc:`user-guide/project-layout`: Suggested folder structure for outputs
- :doc:`user-guide/troubleshooting`: Common issues and fixes
- :doc:`user-guide/glossary`: Definitions for Ex-Fuzzy terms
- :doc:`examples/index`: See real-world applications and case studies

.. tip::
    Check out the :doc:`examples/classification` for more detailed examples with different datasets!
