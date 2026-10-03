=================================
Fuzzy association rule classifier
=================================

:class:`ex_fuzzy.FuzzyRulesClassifier` learns a compact linguistic rule base by
mining candidate fuzzy association rules and selecting a subset of them with a
genetic algorithm. It follows the FARC-HD family of fuzzy association rule
classifiers (Alcalá-Fdez, Alcalá and Herrera, *IEEE Transactions on Fuzzy
Systems* 19(5), 2011). :class:`ex_fuzzy.BaseFuzzyRulesClassifier`, which evolves
rules directly, is a separate estimator and is unchanged.

How it learns
=============

1. **Feature cap.** Only the ``max_features`` most relevant features are kept,
   scored by mutual information by default. This keeps rule generation
   tractable and removes noisy inputs. With ``feature_selection="per_class"``,
   features are scored one class against the rest and each class keeps its own
   ``max_features``. The rule base can then draw on more features overall,
   while each class's rule search stays small.
2. **Fixed partitions.** Each kept numerical feature is split into
   ``n_linguistic_variables`` quantile-based fuzzy terms. Categorical features
   get one term per category. Pass ``linguistic_variables`` to use your own.
3. **Candidate mining.** Rules with up to ``nAnts`` conditions are scored per
   class (see `Longer rules`_ for four or five conditions). A rule is kept when its class support, confidence and penalized
   certainty factor pass their thresholds. Every class keeps some candidates,
   even when no rule passes them.
4. **Prescreening.** A covering-based subgroup-discovery step keeps up to
   ``candidates_per_class`` rules per class. It favours rules with high
   weighted relative accuracy on class samples that are not yet covered.
5. **Genetic rule selection.** A genetic algorithm chooses the subset of
   prescreened rules that maximizes training accuracy, with a small penalty
   per rule. The firing strength of each candidate is computed once, so whole
   populations are scored with array operations.

Quick start
===========

.. code-block:: python

   from ex_fuzzy import FuzzyRulesClassifier
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split

   X, y = load_iris(return_X_y=True, as_frame=True)
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=0, stratify=y
   )

   model = FuzzyRulesClassifier(max_features=8, rule_mode="additive", random_state=0)
   model.fit(X_train, y_train)

   labels = model.predict(X_test)
   probabilities = model.predict_proba(X_test)
   model.print_rules()

The estimator follows the scikit-learn API. It works with pipelines, cloning
and cross-validation, and accepts DataFrames or arrays with any label type.

Additive and sufficient rules
=============================

Each rule is weighted by its penalized certainty factor. ``rule_mode`` follows
the same convention as :class:`ex_fuzzy.BaseFuzzyRulesRegressor`. With
``rule_mode="additive"`` (the default), the weighted firing of each class's
rules is summed and the largest total wins, so several moderately matching
rules can outvote one strong rule. With ``rule_mode="sufficient"``, only each
sample's strongest weighted rule decides, which makes every prediction
traceable to a single rule. A sample that fires no rule takes the
majority class of the training data. ``predict_proba`` normalizes the class
scores, and gives the training class distribution to samples that fire no
rule.

Controlling size and cost
=========================

- ``max_features`` and ``nAnts`` bound the search space and rule length.
  ``class_features_`` records the features each class's rules were mined on.
  ``max_features="auto"`` keeps 8 features for problems with up to five
  classes and 16 for problems with more. ``nAnts="auto"`` allows four
  conditions when a class's rules come from at most six features, and three
  otherwise. The resolved values are stored in ``max_features_`` and
  ``n_conditions_``. The
  number of candidates grows with the number of feature combinations of up to
  ``nAnts`` features.
- ``n_linguistic_variables`` sets the number of terms per numerical feature.
  ``"auto"`` uses three terms for binary problems and five otherwise.
- ``nRules`` caps the number of selected rules, and ``rules_per_class`` caps it
  relative to the number of classes. ``nRules=None`` removes the absolute cap,
  which problems with many classes may need. ``rule_penalty`` trades training
  accuracy for fewer rules.
- ``candidates_per_class`` and ``expansion_factor`` set how many prescreened
  rules the genetic search can choose from.
- ``n_gen``, ``pop_size`` and ``patience`` bound the genetic search.

Longer rules
============

Rules have at most three conditions by default. To allow longer rules, for
example when classes are defined by interactions of several features, follow
these steps.

1. **Set** ``nAnts=4`` **or** ``nAnts=5``. Beyond three conditions, candidate
   mining is pruned by support. Adding a condition can only lower a rule's
   support for every class. So a rule is extended only while its support for
   some class reaches ``min_support``, and only if all of its shorter sub-rules
   were extended too. This keeps the same candidates as trying every
   combination, at a fraction of the cost. Up to three conditions the search
   stays exhaustive, which reproduces the published benchmark results exactly.

   .. code-block:: python

      model = FuzzyRulesClassifier(nAnts=5, random_state=0)

2. **If fitting is too slow, shrink the search.** Raising ``min_support``, for
   example to ``0.1``, prunes the most, because it removes rare combinations
   early. Lowering ``max_features`` reduces the number of feature
   combinations, and lowering ``n_linguistic_variables`` reduces the number of
   term combinations per feature set.
3. **Compare against the default by cross-validation.** Longer rules are more
   specific, not automatically more accurate.

   As a guide, eight KEEL datasets were run with the benchmark protocol
   (5-fold cross-validation, 2 to 10 classes): pima, wdbc, vehicle, german,
   car, contraceptive, yeast and segment.

   .. list-table::
      :header-rows: 1

      * - ``nAnts``
        - Mean accuracy, additive
        - Mean accuracy, sufficient
      * - 3 (default)
        - 0.745
        - 0.743
      * - 4
        - 0.750
        - 0.741
      * - 5
        - 0.752
        - 0.745

   Longer rules helped car and german by about two points and cost yeast one
   to two points. Fit time grew by up to about twice (yeast), and much less
   on the other datasets. Mining itself stays cheap: on 2,000 samples with 8
   features and 5 terms, mining rules of up to five conditions took 0.07 s
   with pruning, against 3.2 s for exhaustive enumeration. Most of the extra
   time goes to selecting among the additional rules.

4. **Check what the model used.** ``n_conditions_`` holds the resolved limit,
   and ``print_rules()`` shows how many conditions the selected rules actually
   have. The genetic selection often keeps short rules even when long ones are
   allowed.

Inspecting the rules
====================

``selected_features_`` holds the indices of the features the rules use, and
``linguistic_variables_`` their partitions. ``rule_base_`` is an Ex-Fuzzy
:class:`~ex_fuzzy.rules.MasterRuleBase`, with one rule base per class and the
certainty factors as rule weights, so the usual rule inspection and
persistence tools apply. ``internal_classifier()`` wraps it as a
:class:`~ex_fuzzy.BaseFuzzyRulesClassifier`. That model expects only the
selected features and uses winning-rule inference without the majority-class
fallback.

Compatibility
=============

The constructor keeps the arguments of the earlier two-stage genetic
classifier of the same name, and ``fit(X, y, n_gen=..., pop_size=...)``
still works. ``n_class`` and ``runner`` are accepted but ignored.
``expansion_factor`` now multiplies the prescreened pool. Only Type-1 fuzzy
sets are supported.
