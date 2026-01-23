Rules Module
============

The :mod:`ex_fuzzy.rules` module contains the core classes and functions for fuzzy rule definition, management, and inference.

.. currentmodule:: ex_fuzzy.rules

Overview
--------

This module implements a complete fuzzy inference system supporting:

* **Type-1, Type-2, and General Type-2** fuzzy sets
* **Multiple inference methods**: Mamdani and Takagi-Sugeno
* **Various t-norm operations**: Product, minimum, and other aggregation functions
* **Defuzzification methods**: Centroid, height, and other methods
* **Rule quality assessment**: Dominance scores and performance metrics

Architecture
~~~~~~~~~~~~

The module follows a hierarchical structure:

1. **Individual Rules**: :class:`RuleSimple` for single rule representation
2. **Rule Collections**: :class:`RuleBaseT1`, :class:`RuleBaseT2`, :class:`RuleBaseGT2` 
3. **Multi-class Systems**: :class:`MasterRuleBase` for complete fuzzy systems
4. **Inference Support**: Functions for membership computation and aggregation

Functions
---------

- :func:`ex_fuzzy.rules.compute_antecedents_memberships`
- :func:`ex_fuzzy.rules.construct_rule_base`
- :func:`ex_fuzzy.rules.list_rules_to_matrix`
- :func:`ex_fuzzy.rules.generate_rule_string`
- :func:`ex_fuzzy.rules.generate_rule_string_regression`

Rule Classes
------------

RuleSimple
~~~~~~~~~~

.. autoclass:: RuleSimple
   :members:
   :inherited-members:
   :show-inheritance:

RuleBase Classes
~~~~~~~~~~~~~~~~

.. autoclass:: RuleBaseT1
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: RuleBaseT2
   :members:
   :inherited-members:
   :show-inheritance:

.. autoclass:: RuleBaseGT2
   :members:
   :inherited-members:
   :show-inheritance:

MasterRuleBase
~~~~~~~~~~~~~~

.. autoclass:: MasterRuleBase
   :members:
   :inherited-members:
   :show-inheritance:

Core Functions
--------------

Membership Computation
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: compute_antecedents_memberships

Constants and Modifiers
-----------------------

Rule Modifiers
~~~~~~~~~~~~~~

The module supports linguistic hedges that modify fuzzy set membership:

.. data:: modifiers_names

   Dictionary mapping modifier powers to linguistic terms:
   
   * ``0.5``: "Somewhat"
   * ``1.0``: "" (no modifier)
   * ``1.3``: "A little"
   * ``1.7``: "Slightly"  
   * ``2.0``: "Very"
   * ``3.0``: "Extremely"
   * ``4.0``: "Very very"

Examples
--------

Creating Simple Rules
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.rules as rules
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np

   # Create fuzzy variables
   temp_var = fs.fuzzyVariable("Temperature", [0, 50], 3, fs.FUZZY_SETS.t1)
   humidity_var = fs.fuzzyVariable("Humidity", [0, 100], 3, fs.FUZZY_SETS.t1)
   
   # Create a simple rule: IF Temperature is High AND Humidity is Low THEN Comfort is Good
   rule = rules.RuleSimple(
       antecedents=[temp_var[2], humidity_var[0]],  # High temp, Low humidity
       consequent=1,  # Good comfort class
       weight=1.0
   )

   # Evaluate rule for input
   input_values = np.array([35, 25])  # 35°C, 25% humidity
   membership = rule.eval_rule([temp_var, humidity_var], input_values)
   print(f"Rule activation: {membership}")

Building Rule Bases
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create rule base for Type-1 fuzzy sets
   rule_base = rules.RuleBaseT1()
   
   # Add multiple rules
   rules_list = [
       rules.RuleSimple([temp_var[0], humidity_var[0]], 0, 1.0),  # Low temp, Low humidity -> Class 0
       rules.RuleSimple([temp_var[1], humidity_var[1]], 1, 0.8),  # Med temp, Med humidity -> Class 1
       rules.RuleSimple([temp_var[2], humidity_var[2]], 2, 0.9),  # High temp, High humidity -> Class 2
   ]
   
   rule_base.add_rules(rules_list)
   rule_base.antecedents = [temp_var, humidity_var]

Multi-class Systems
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create master rule base for multi-class classification
   master_rb = rules.MasterRuleBase()
   
   # Add rule bases for each class
   for class_id in range(3):
       class_rules = rules.RuleBaseT1()
       # Add class-specific rules...
       master_rb.add_rulebase(class_rules, class_id)
   
   # Evaluate complete system
   input_data = np.array([[35, 25], [20, 80], [40, 90]])
   predictions = master_rb.predict(input_data)

Computing Memberships
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Compute antecedent memberships for multiple inputs
   antecedents = [temp_var, humidity_var]
   input_values = np.array([[25, 60], [35, 30], [15, 80]])
   
   memberships = rules.compute_antecedents_memberships(antecedents, input_values)
   
   # Access membership for first variable, second sample, first fuzzy set
   first_var_memberships = memberships[0]
   print(f"Memberships for variable 1: {first_var_memberships}")

Rule String Formatting
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Render a human-readable rule string
   rule_text = rules.generate_rule_string(rule_base[0], rule_base.antecedents)
   print(rule_text)

See Also
--------

* :mod:`ex_fuzzy.fuzzy_sets` : Fuzzy set definitions and operations
* :mod:`ex_fuzzy.centroid` : Defuzzification algorithms
* :mod:`ex_fuzzy.classifiers` : High-level classification interfaces
* :mod:`ex_fuzzy.eval_tools` : Rule evaluation and performance metrics
