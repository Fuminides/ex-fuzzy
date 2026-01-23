========
Glossary
========

Short definitions for commonly used Ex-Fuzzy terms.

Related Guides
==============

- :doc:`../user-guide/core-concepts`
- :doc:`../user-guide/recipes`
- :doc:`../user-guide/validation-visualization`

Accuracy (ACC)
==============

Rule-level accuracy reported alongside dominance scores in rule listings.

Alpha-Cut
=========

A slice of a Type-2 or GT2 fuzzy set at a fixed membership degree. Used to evaluate
interval uncertainty.

Consequent
==========

The output part of a rule, usually the predicted class label (classification) or
a numeric value (regression).

Dominance Score (DS)
====================

A rule strength score used to rank or filter rules. Higher values typically indicate
more influential rules.

Fuzzy Partition
===============

The set of membership functions for a single input variable (e.g., Low/Medium/High).

GT2 (General Type-2)
====================

General Type-2 fuzzy sets with full membership uncertainty, represented via multiple
alpha-cuts.

IV (Interval-Valued)
====================

Interval-Valued (Type-2) fuzzy sets where membership is an interval instead of a
single value.

MasterRuleBase
==============

A container holding one rule base per class (classification), or a single base for
regression.

Membership Function
===================

Function that maps a numeric input to a membership degree in a fuzzy set.

Rule Base
=========

A collection of rules sharing the same consequents and membership functions.

T1 (Type-1)
===========

Standard fuzzy sets with a single membership value for each input.

Weight (WGHT)
=============

Optional rule weight included in printed rule strings for some training modes.

Next Steps
==========

- Learn the concepts in context in :doc:`../user-guide/core-concepts`
- Apply terms in :doc:`../user-guide/recipes`
