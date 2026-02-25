Rules Module
============

The :mod:`ex_fuzzy.rules` module contains rule and rule-base abstractions used
by Ex-Fuzzy inference engines.

.. currentmodule:: ex_fuzzy.rules

Core Classes
------------

.. autoclass:: RuleSimple
   :members:
   :show-inheritance:

.. autoclass:: RuleBaseT1
   :members:
   :show-inheritance:

.. autoclass:: RuleBaseT2
   :members:
   :show-inheritance:

.. autoclass:: RuleBaseGT2
   :members:
   :show-inheritance:

.. autoclass:: MasterRuleBase
   :members:
   :show-inheritance:

Core Functions
--------------

.. autofunction:: compute_antecedents_memberships
.. autofunction:: generate_rule_string

See Also
--------

* :mod:`ex_fuzzy.fuzzy_sets`
* :mod:`ex_fuzzy.evolutionary_fit`
* :mod:`ex_fuzzy.classifiers`
