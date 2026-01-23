==========
Fuzzy Sets
==========

.. currentmodule:: ex_fuzzy.fuzzy_sets

The ``fuzzy_sets`` module provides the fundamental building blocks for fuzzy logic systems in Ex-Fuzzy.

Overview
========

Fuzzy sets extend classical set theory by allowing partial membership. Instead of binary membership (in/out), 
fuzzy sets assign membership degrees between 0 and 1, enabling more nuanced representation of concepts.

Core Classes
============

Base Fuzzy Set
--------------

.. autoclass:: FS
   :members:
   :show-inheritance:

   The base class for all fuzzy sets in Ex-Fuzzy. Provides fundamental operations
   and the interface that all fuzzy set types must implement.

   **Key Methods:**

   - :meth:`ex_fuzzy.fuzzy_sets.FS.membership`

Gaussian Fuzzy Set
------------------

.. autoclass:: gaussianFS
   :members:
   :show-inheritance:

   Implements Gaussian (normal) membership functions. These are the most commonly
   used fuzzy sets due to their smooth shape and mathematical properties.

   **Example:**

   .. code-block:: python

      import ex_fuzzy.fuzzy_sets as fs
      
      # Create a Gaussian fuzzy set
      gaussian_set = fs.gaussianFS(
          name="Medium",
          membership_parameters=[5.0, 1.5],
          domain=[0, 10],
      )
      
      # Calculate membership
      membership = gaussian_set.membership(4.5)
      print(f"Membership of 4.5: {membership:.3f}")

Interval-Valued Gaussian Fuzzy Set
-----------------------------------

.. autoclass:: gaussianIVFS
   :members:
   :show-inheritance:

   Extends Gaussian fuzzy sets to handle uncertainty in the membership function
   itself using interval-valued membership degrees.

Categorical Fuzzy Set
---------------------

.. autoclass:: categoricalFS
   :members:
   :show-inheritance:

   Handles discrete, categorical variables by mapping categories to membership
   degrees. Useful for non-numeric data.

Fuzzy Variable
==============

.. autoclass:: fuzzyVariable
   :members:
   :show-inheritance:

   Represents a complete fuzzy variable with multiple linguistic terms (fuzzy sets).
   This is the main class users interact with for defining input and output variables.

   **Example:**

   .. code-block:: python

      import ex_fuzzy.fuzzy_sets as fs
      
      # Create a fuzzy variable for temperature
      temperature = fs.fuzzyVariable(
          name="temperature",
          fuzzy_sets=[
              fs.gaussianFS("Cold", [5.0, 3.0], [0, 40]),
              fs.gaussianFS("Warm", [20.0, 4.0], [0, 40]),
              fs.gaussianFS("Hot", [32.0, 3.0], [0, 40]),
          ],
          units="C",
      )
      
      print(temperature.linguistic_variable_names())
      
      # Evaluate membership for a specific value
      memberships = temperature.membership(25.0)
      print(f"Memberships for 25°C: {memberships}")

Constants and Enums
===================

Fuzzy Set Types
---------------

.. autoclass:: FUZZY_SETS
   :members:

   Enumeration of available fuzzy set types:

   - ``t1``: Type-1 fuzzy sets (crisp membership values)
   - ``t2``: Type-2 interval-valued fuzzy sets
   - ``gt2``: General Type-2 fuzzy sets

Linguistic Naming
-----------------

Fuzzy set naming is user-defined. Provide human-readable names when you
instantiate :class:`FS` subclasses and when you build :class:`fuzzyVariable`.

   - ``standard``: Low, Medium, High
   - ``numeric``: Var1, Var2, Var3
   - ``custom``: User-defined names

Examples
========

Basic Usage
-----------

.. code-block:: python

   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np
   import matplotlib.pyplot as plt
   
   # Create a fuzzy variable for age
   age = fs.fuzzyVariable(
       name="age",
       fuzzy_sets=[
           fs.gaussianFS("Young", [20, 10], [0, 100]),
           fs.gaussianFS("Middle-aged", [50, 10], [0, 100]),
           fs.gaussianFS("Old", [80, 10], [0, 100]),
       ],
   )
   
   # Plot the membership functions
   x = np.linspace(0, 100, 1000)
   plt.figure(figsize=(10, 6))
   
   for i, term_name in enumerate(age.linguistic_variable_names()):
       y = [age[i].membership(val) for val in x]
       plt.plot(x, y, label=term_name, linewidth=2)
   
   plt.xlabel('Age')
   plt.ylabel('Membership Degree')
   plt.title('Age Fuzzy Variable')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.show()

Advanced Usage
--------------

.. code-block:: python

   # Create custom fuzzy sets with specific parameters
   young = fs.gaussianFS(mean=20, std=10, domain=[0, 100], name="Young")
   middle = fs.gaussianFS(mean=50, std=15, domain=[0, 100], name="Middle-aged") 
   old = fs.gaussianFS(mean=80, std=12, domain=[0, 100], name="Old")
   
   # Create variable with custom sets
   age_custom = fs.fuzzyVariable(
       domain=[0, 100],
       name="age_custom", 
       fuzzy_sets=[young, middle, old]
   )
   
   # Evaluate multiple values
   ages_to_test = [15, 35, 55, 75]
   for age_val in ages_to_test:
       memberships = age_custom.membership(age_val)
       print(f"Age {age_val}: {dict(zip(age_custom.linguistic_variable_names(), memberships))}")

Type-2 Fuzzy Sets
-----------------

.. code-block:: python

   import ex_fuzzy.utils as utils
   import ex_fuzzy.fuzzy_sets as fs
   import numpy as np

   # Create Type-2 fuzzy variables for handling uncertainty
   data = np.array([[10, 20], [15, 25], [20, 30]])
   uncertain_vars = utils.construct_partitions(data, fs.FUZZY_SETS.t2, n_partitions=3)
   uncertain_temp = uncertain_vars[0]

   # Type-2 membership returns intervals
   temp_value = 25.0
   memberships = uncertain_temp.membership(np.array([temp_value]))
   print(f"Type-2 memberships for {temp_value}°C:")
   for i, name in enumerate(uncertain_temp.linguistic_variable_names()):
       lower, upper = memberships[i][0]
       print(f"  {name}: [{lower:.3f}, {upper:.3f}]")

See Also
========

- :doc:`../user-guide/core-concepts`: Detailed guide on fuzzy sets and variables
- :doc:`../examples/classification`: Example workflows using fuzzy sets
- :doc:`evolutionary_fit`: Using fuzzy variables in classification
- :doc:`rules`: Creating rules with fuzzy variables
