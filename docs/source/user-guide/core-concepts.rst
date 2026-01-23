Core Concepts
=============

(This is just a small intro to fuzzy logic. For a better introduction I recommend checking the specialized literature)

This guide introduces the fundamental concepts underlying the ex-fuzzy library, providing a solid foundation for understanding fuzzy logic, fuzzy sets, and fuzzy rule-based systems.

Related Guides
========================================

- :doc:`../getting-started`
- :doc:`../user-guide/recipes`
- :doc:`../user-guide/glossary`

.. contents:: Table of Contents
   :local:
   :depth: 2

What is Fuzzy Logic?
----------------------------------------

Fuzzy logic is an extension of classical binary logic that allows for degrees of truth between completely true and completely false. Unlike traditional logic where statements are either true (1) or false (0), fuzzy logic permits partial truth values between 0 and 1.

Why Fuzzy Logic?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real-world problems often involve:

* **Uncertainty**: Imprecise or incomplete information
* **Vagueness**: Concepts without sharp boundaries (e.g., "tall", "hot", "many")
* **Gradual transitions**: Natural phenomena rarely have sharp cutoffs
* **Human reasoning**: People naturally think in terms of degrees rather than absolutes

Example: Temperature Classification
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Consider classifying temperature:

* **Classical logic**: Temperature ≥ 25°C is "hot", < 25°C is "not hot"
* **Fuzzy logic**: Temperature has degrees of "hotness"

  - 20°C: 0.0 hot (completely not hot)
  - 22°C: 0.2 hot (slightly hot)
  - 25°C: 0.5 hot (moderately hot)
  - 30°C: 0.8 hot (very hot)
  - 35°C: 1.0 hot (completely hot)

Fuzzy Sets
----------------------------------------

A fuzzy set is a collection of objects where each object has a degree of membership between 0 and 1.

Mathematical Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fuzzy set A in universe X is defined by a membership function:

.. math::

   \\mu_A: X \\rightarrow [0, 1]

where :math:`\\mu_A(x)` represents the degree to which element x belongs to set A.

Types of Fuzzy Sets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Ex-fuzzy supports three types of fuzzy sets:

Type-1 Fuzzy Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Standard fuzzy sets where membership is a crisp value between 0 and 1.

.. code-block:: python

   import ex_fuzzy.fuzzy_sets as fs
   import ex_fuzzy.utils as utils
   import numpy as np
   import pandas as pd

   # Create sample data
   data = np.array([[20, 30, 40], [25, 35, 45], [30, 40, 50]])
   
   # Create Type-1 fuzzy variables automatically from data
   fuzzy_variables_t1 = utils.construct_partitions(data, fs.FUZZY_SETS.t1, n_partitions=3)
   
   # Each variable contains fuzzy sets for that feature
   temperature_var = fuzzy_variables_t1[0]  # First feature
   print(f"Variable name: {temperature_var.name}")
   print(f"Number of fuzzy sets: {len(temperature_var.linguistic_variables)}")

Type-2 Fuzzy Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Fuzzy sets where the membership function itself is fuzzy, represented by upper and lower bounds.

.. code-block:: python

   # Create Type-2 fuzzy variables with uncertainty
   fuzzy_variables_t2 = utils.construct_partitions(data, fs.FUZZY_SETS.t2, n_partitions=3)
   
   # Type-2 sets have upper and lower membership functions
   # to model uncertainty in the membership itself
   temperature_t2 = fuzzy_variables_t2[0]

General Type-2 Fuzzy Sets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Most general form where membership is a fuzzy set in three dimensions.

.. code-block:: python

   # Create General Type-2 fuzzy variables
   fuzzy_variables_gt2 = utils.construct_partitions(data, fs.FUZZY_SETS.gt2, n_partitions=3)

Membership Functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common membership function shapes supported:

**Triangular**
  Symmetric triangular shape with three parameters (left, center, right)

**Trapezoidal**
  Flat-topped with four parameters (left, left-top, right-top, right)

**Gaussian**
  Bell-shaped curve with center and width parameters

Linguistic Variables
----------------------------------------

Linguistic variables represent concepts that can be described using natural language terms.

Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A linguistic variable consists of:

1. **Name**: The variable name (e.g., "temperature", "speed", "quality")
2. **Universe**: The range of possible values (e.g., 0-100°C)
3. **Terms**: Linguistic terms that describe the variable (e.g., "cold", "warm", "hot")
4. **Membership functions**: Mathematical functions defining each term

Creating Linguistic Variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import ex_fuzzy.utils as utils
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np

   # Create sample data
   data = np.array([[20, 25, 30], [35, 40, 45], [50, 55, 60]])
   
   # Automatic partitioning creates linguistic variables from data
   fuzzy_variables = utils.construct_partitions(
       data, 
       fz_type_studied=fs.FUZZY_SETS.t1, 
       n_partitions=3
   )
   
   # Each variable contains fuzzy sets representing linguistic terms
   # like "low", "medium", "high" for each feature

Fuzzy Rules
----------------------------------------

Fuzzy rules encode human knowledge in IF-THEN format using linguistic variables.

Rule Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Basic Form:**
  IF antecedent(s) THEN consequent

**Components:**
  - **Antecedents**: Conditions (can be combined with AND/OR)
  - **Consequent**: Conclusion or action
  - **Confidence**: Rule strength or support

Ex-fuzzy uses rule classes like ``RuleSimple`` and rule base classes like ``RuleBaseT1``, ``RuleBaseT2``, and ``RuleBaseGT2`` to manage collections of rules for different fuzzy set types.

.. code-block:: python

   # Simple rule: IF temperature is high THEN comfort is low
   rule1 = rules.RuleSimple(
       antecedents=[2],  # high temperature label index
       consequent=0,  # low comfort class
   )

   # Complex rule: IF temperature is high AND humidity is high THEN comfort is very_low
   rule2 = rules.RuleSimple(
       antecedents=[2, 2],  # high temp AND high humidity
       consequent=0,  # very low comfort
   )

Types of Fuzzy Rules
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Mamdani Rules**
  Output is a fuzzy set

.. code-block:: python

   # IF temperature is hot THEN fan_speed is high
   # Both antecedent and consequent are fuzzy sets

**Takagi-Sugeno Rules**
  Output is a crisp function

.. code-block:: python

   # IF temperature is hot THEN fan_speed = 0.8 * temperature + 10
   # Consequent is a mathematical function

Rule Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rules are evaluated in several steps:

1. **Fuzzification**: Convert crisp inputs to fuzzy membership degrees
2. **Rule firing**: Calculate how strongly each rule applies
3. **Aggregation**: Combine outputs from multiple rules
4. **Defuzzification**: Convert fuzzy output to crisp value

.. code-block:: python

   import numpy as np
   import ex_fuzzy.fuzzy_sets as fs
   import ex_fuzzy.rules as rules
   import ex_fuzzy.utils as utils

   # Evaluate rule for specific input
   input_values = np.array([[35, 80]])  # 35°C, 80% humidity

   # Create linguistic variables from data
   data = np.array([[20, 40], [25, 60], [35, 80]])
   fuzzy_variables = utils.construct_partitions(data, fs.FUZZY_SETS.t1, n_partitions=3)
   temp_var, humid_var = fuzzy_variables[0], fuzzy_variables[1]

   # Rule: IF temp is high AND humidity is high THEN comfort is low
   rule = rules.RuleSimple([2, 2], 0)

   # Evaluate rule strength via a temporary rule base
   rule_base = rules.RuleBaseT1()
   rule_base.antecedents = [temp_var, humid_var]
   rule_base.add_rules([rule])
   strengths = rule_base.compute_rule_antecedent_memberships(input_values)
   print(f"Rule fires with strength: {strengths[0, 0]:.3f}")

Fuzzy Inference Systems
----------------------------------------

A fuzzy inference system (FIS) combines multiple fuzzy rules to make decisions or predictions.

System Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   Input → Fuzzification → Rule Evaluation → Aggregation → Defuzzification → Output
      ↓           ↓              ↓             ↓              ↓
   Crisp      Fuzzy        Rule Strengths   Combined      Crisp
   Values     Degrees                       Output        Result

Components
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Fuzzifier**: Converts crisp inputs to fuzzy degrees
2. **Rule Base**: Collection of fuzzy rules
3. **Inference Engine**: Evaluates rules and combines outputs  
4. **Defuzzifier**: Converts fuzzy output to crisp result

Building a Complete System
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import ex_fuzzy.fuzzy_sets as fs
   import ex_fuzzy.rules as rules
   import ex_fuzzy.utils as utils

   # 1. Define linguistic variables from data
   data = np.array([[20, 40], [25, 60], [35, 80]])
   temperature, humidity = utils.construct_partitions(
       data,
       fs.FUZZY_SETS.t1,
       n_partitions=3,
   )
   
   # 2. Create rules
   rule_list = [
       rules.RuleSimple([temperature[0], humidity[0]], 2, 1.0),  # Low temp, low humid → high comfort
       rules.RuleSimple([temperature[0], humidity[1]], 1, 0.8),  # Low temp, med humid → med comfort
       rules.RuleSimple([temperature[1], humidity[1]], 1, 0.9),  # Med temp, med humid → med comfort
       rules.RuleSimple([temperature[2], humidity[2]], 0, 1.0),  # High temp, high humid → low comfort
   ]
   
   # 3. Create rule base
   rule_base = rules.RuleBaseT1()
   rule_base.add_rules(rule_list)
   rule_base.antecedents = [temperature, humidity]
   
   # 4. Evaluate system
   test_inputs = np.array([[25, 60], [35, 80], [15, 40]])
   predictions = rule_base.predict(test_inputs)

Classification vs. Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Fuzzy Classification**
  Predicts discrete class labels

.. code-block:: python

   from ex_fuzzy.classifiers import RuleMineClassifier
   
   classifier = RuleMineClassifier(nRules=15, nAnts=3)
   classifier.fit(X_train, y_train)
   predictions = classifier.predict(X_test)

**Fuzzy Regression**
  Predicts continuous values

.. code-block:: python

   # Fuzzy regression with continuous outputs
   # (Advanced topic covered in separate guides)

Rule Learning and Optimization
----------------------------------------

Ex-fuzzy provides automated methods to learn fuzzy rules from data.

Two-Stage Approach
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Rule Mining**: Discover candidate rules from data
2. **Evolutionary Optimization**: Select optimal rule combinations

.. code-block:: python

   import ex_fuzzy.fuzzy_sets as fs
   import ex_fuzzy.rule_mining as rm
   import ex_fuzzy.utils as utils
   import numpy as np

   data = np.array([[20, 40], [25, 60], [35, 80]])
   X_train = data
   y_train = np.array([0, 1, 1])
   linguistic_vars = utils.construct_partitions(data, fs.FUZZY_SETS.t1, n_partitions=3)

   # Stage 1: Mine candidate rules
   candidate_rules = rm.multiclass_mine_rulebase(
       X_train,
       y_train,
       linguistic_vars,
       support_threshold=0.05,
       max_depth=3,
   )

   # Stage 2: Optimize rule selection
   import ex_fuzzy.evolutionary_fit as evf

   classifier = evf.BaseFuzzyRulesClassifier(
       nRules=20,
       nAnts=3,
       linguistic_variables=linguistic_vars,
   )
   classifier.fit(X_train, y_train, candidate_rules=candidate_rules, n_gen=50, pop_size=100)

Quality Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Rules are evaluated using multiple criteria:

**Support**
  How often the rule applies to the data

**Confidence**  
  How often the rule is correct when it applies

**Lift**
  How much better the rule is than random

**Accuracy**
  Overall correctness of the rule

**Complexity**
  Number of conditions in the rule

Multi-Objective Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Balance multiple criteria simultaneously when choosing the final rule base,
such as accuracy, coverage, and interpretability.

Interpretability and Explainability
----------------------------------------

One of the key advantages of fuzzy systems is their interpretability.

Rule Readability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fuzzy rules can be expressed in natural language:

.. code-block:: text

   Rule 1: IF temperature is high AND humidity is high THEN comfort is low (weight: 0.95)
   Rule 2: IF temperature is low AND humidity is low THEN comfort is high (weight: 0.87)
   Rule 3: IF temperature is medium THEN comfort is medium (weight: 0.72)

System Transparency
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Users can understand:

- Which rules fired for a given input
- How strongly each rule contributed
- Why a particular decision was made

.. code-block:: python

   # Analyze rule contributions
   from ex_fuzzy import eval_rules

   rule_eval = eval_rules.evalRuleBase(classifier.rule_base, X_test, y_test)
   rule_eval.add_full_evaluation()

   for rule_id, rule in enumerate(classifier.rule_base.get_rules()):
       print(f"Rule {rule_id}:")
       print(f"  Dominance score: {rule.score}")
       print(f"  Accuracy: {rule.accuracy:.3f}")

Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Visual representation aids understanding:

.. code-block:: python

   # Plot fuzzy partitions
   import ex_fuzzy.vis_rules as vis
   
   vis.plot_fuzzy_variable(temperature)
   vis.visualize_rulebase(rule_base)

Common Patterns and Best Practices
----------------------------------------

Variable Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Number of Terms**
  - 3-5 terms usually sufficient
  - More terms = higher precision but lower interpretability
  - Odd numbers (3, 5, 7) often work well

**Overlap**
  - Adjacent terms should overlap by 10-50%
  - No overlap = discontinuous behavior
  - Too much overlap = reduced discrimination

**Coverage**
  - Ensure complete coverage of the universe
  - No gaps between adjacent terms

Rule Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Rule Complexity**
  - Keep rules simple (2-4 antecedents maximum)
  - Complex rules are harder to interpret and may overfit

**Rule Consistency**
  - Avoid contradictory rules with high weights
  - Similar antecedents should lead to similar consequents

**Completeness**
  - Cover all important regions of the input space
  - Sparse rule bases may have poor generalization

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Efficiency**
  - Fewer rules = faster inference
  - Pre-compute membership values for batch processing
  - Use appropriate data structures for large rule bases

**Accuracy vs. Interpretability**
  - More rules generally improve accuracy
  - Balance performance with understandability
  - Use multi-objective optimization

Common Pitfalls
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Over-partitioning**
  Too many fuzzy sets reduces interpretability

**Under-smoothing**
  Sharp membership functions may cause discontinuities

**Rule Explosion**
  Combinatorial growth with multiple variables and terms

**Ignoring Domain Knowledge**
  Purely data-driven approaches may miss important constraints

Next Steps
----------------------------------------

Now that you understand the core concepts, explore:

1. :doc:`../getting-started` - Hands-on tutorial
2. :doc:`../user-guide/index` - Detailed user guides
3. :doc:`../examples/index` - Real-world examples
4. :doc:`../api/index` - Complete API reference

For specific applications:

- **Classification**: See :doc:`../examples/classification`
- **Rule Mining**: See :doc:`../api/rule_mining`
- **Optimization**: See :doc:`../api/evolutionary_fit`
- **Visualization**: See :doc:`../user-guide/validation-visualization`
