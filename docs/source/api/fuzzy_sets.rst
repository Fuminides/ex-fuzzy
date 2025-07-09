===========
Fuzzy Sets
===========

.. currentmodule:: ex_fuzzy.fuzzy_sets

The ``fuzzy_sets`` module provides the fundamental building blocks for fuzzy logic systems in Ex-Fuzzy.

Overview
========

Fuzzy sets extend classical set theory by allowing partial membership. Instead of binary membership (in/out), 
fuzzy sets assign membership degrees between 0 and 1, enabling more nuanced representation of concepts.

Key Classes
===========

.. autosummary::
   :toctree: generated/

   FS
   gaussianFS
   gaussianIVFS
   categoricalFS
   fuzzyVariable

Enumerations
============

.. autosummary::
   :toctree: generated/

   FUZZY_SETS
   LINGUISTIC_TYPES

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

   .. autosummary::
      :toctree: generated/
      
      FS.membership
      FS.centroid
      FS.defuzzify

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
          mean=5.0,
          std=1.5,
          domain=[0, 10],
          name="Medium"
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
          domain=[0, 40],  # Celsius
          name="temperature",
          n_linguistic_vars=3  # Cold, Warm, Hot
      )
      
      # The variable automatically creates linguistic terms
      print(temperature.linguistic_variable_names())
      # Output: ['Cold', 'Warm', 'Hot']
      
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
   - ``t2``: Type-2 fuzzy sets (fuzzy membership values)  
   - ``gt2``: General Type-2 fuzzy sets
   - ``it2``: Interval Type-2 fuzzy sets

Linguistic Types
----------------

.. autoclass:: LINGUISTIC_TYPES
   :members:

   Enumeration of linguistic variable naming conventions:

   - ``standard``: Low, Medium, High
   - ``numeric``: Var1, Var2, Var3
   - ``custom``: User-defined names

Functions
=========

Utility Functions
-----------------

.. autosummary::
   :toctree: generated/

   create_fuzzy_variable
   load_fuzzy_variables
   validate_domain
   normalize_membership

.. autofunction:: create_fuzzy_variable

   Convenience function for creating fuzzy variables with common configurations.

.. autofunction:: load_fuzzy_variables

   Load pre-configured fuzzy variables from file.

.. autofunction:: validate_domain

   Validate that a domain specification is correct.

.. autofunction:: normalize_membership

   Normalize membership values to ensure they sum to 1.

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
       domain=[0, 100],
       name="age",
       linguistic_variable_names=["Young", "Middle-aged", "Old"]
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

   # Create Type-2 fuzzy variable for handling uncertainty
   uncertain_temp = fs.fuzzyVariable(
       domain=[0, 40],
       name="uncertain_temperature",
       fuzzy_type=fs.FUZZY_SETS.t2,
       n_linguistic_vars=3
   )
   
   # Type-2 membership returns intervals
   temp_value = 25.0
   memberships = uncertain_temp.membership(temp_value)
   print(f"Type-2 memberships for {temp_value}°C:")
   for i, name in enumerate(uncertain_temp.linguistic_variable_names()):
       lower, upper = memberships[i]
       print(f"  {name}: [{lower:.3f}, {upper:.3f}]")

See Also
========

- :doc:`../user-guide/fuzzy-sets`: Detailed guide on using fuzzy sets
- :doc:`../examples/custom-fuzzy-sets`: Examples of creating custom fuzzy sets
- :doc:`evolutionary_fit`: Using fuzzy variables in classification
- :doc:`rules`: Creating rules with fuzzy variables
