================
Mixture of rules
================

``BaseFuzzyRulesClassifier`` decides by default with **one rule per sample**: the rule with the largest
association degree (firing strength, times its dominance score or weight depending on ``ds_mode``) gives
its class. That rule cannot add up evidence from other rules that also fire. ``inference='mixture'``
makes every rule speak:

.. math::

   \mathrm{score}_c(x) = b_c + \sum_r W_{r,c}\,\mu_r(x), \qquad \hat y(x) = \arg\max_c \mathrm{score}_c(x)

where :math:`\mu_r(x)` is the firing strength of rule :math:`r` (the t-norm of its antecedent
memberships), :math:`W` holds one weight per rule and class and :math:`b` one bias per class. The weights
are **learned**, instead of using the rules' dominance scores as the decision.

Design
======

* **Two stages.** The genetic search finds and prunes the rules exactly as with winning-rule inference
  (same objective, same seeded result). Then :math:`W` and :math:`b` are fitted on the training samples by
  multinomial logistic regression on the ``samples x rules`` matrix of firing strengths, with an L2 penalty
  ``||W||^2 / (2 C)`` (scikit-learn's convention; the bias is not penalised), solved with L-BFGS-B.
* **Weights.** Unconstrained (default) or non-negative (``mixture_non_negative=True``). Adding a constant to
  all the class weights of a rule changes no prediction, so both describe the same family of models and only
  the penalty differs. Unconstrained weights of a rule add up to zero over the classes, so a negative weight
  reads "less evidence for this class than for the average class", not absolute evidence against it.
  Non-negative weights read "this rule votes for class *c* with strength *w*", and every rule then has weight
  0 for at least one class.
* **Search under the mixture** (``mixture_search=True``, optional). Each candidate rule base of the genetic
  search is pruned as a fitted model would be, gets its own mixture fitted on the training samples, and is
  scored by that mixture's macro F1 with the usual coverage and compactness penalties. It runs one logistic
  regression per new candidate, outside the built-in evaluators and their caches, so it is much slower
  (on 30,000 samples, about 0.1-0.3 s per candidate with single-threaded BLAS; set ``OMP_NUM_THREADS=1``,
  since multi-threaded BLAS is slower on these small products).
* **Consequents.** A rule's consequent only matters for the search and for pruning; in the mixture its
  weights say which classes it supports. Rules with the same antecedents fire identically, so only the first
  gets weights (``mixture_.duplicate_of_``).
* **No rule fires.** The scores are then the biases. With ``allow_unknown=True`` the sample is predicted as
  unknown, as with winning-rule inference; otherwise as the class of the largest bias.

Usage
=====

.. code-block:: python

   from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier

   model = BaseFuzzyRulesClassifier(nRules=30, nAnts=2, inference='mixture', mixture_non_negative=True)
   model.fit(X_train, y_train)
   model.predict(X_test)                    # argmax of the scores
   model.mixture_scores(X_test)             # (samples, classes): b + firing @ W
   model.mixture_contributions(X_test)      # (samples, rules, classes): W[r, c] * firing_r
   model.print_mixture()                    # every rule with its weights, its strongest class, the biases
   model.mixture_.coef_, model.mixture_.intercept_

   # Weights for the rules of an existing winning-rule model (the rules stay as they are):
   winning.set_params(inference='mixture')
   winning.fit_mixture(X_train, y_train)

``print_mixture`` writes, for iris with non-negative weights (10 rules searched, 3 kept)::

   Bias: setosa +1.13 | versicolor -0.92 | virginica -0.21
   Rule 0: IF sepal_length IS Medium; strongest class: versicolor; weights: setosa +0.00 | versicolor +1.35 | virginica +1.17
   Rule 1: IF petal_width IS Medium; strongest class: versicolor; weights: setosa +0.00 | versicolor +4.42 | virginica +0.00
   Rule 2: IF sepal_width IS Medium AND petal_length IS High; strongest class: virginica; weights: setosa +0.00 | versicolor +0.00 | virginica +4.54

No rule here was found for setosa: it is the class of the largest bias, predicted where the other rules
fire little.

``predict_proba`` returns ``softmax(scores)``: the class probabilities of a fitted logistic model on data
like its training data (with any class-aware sampling of that data), **not** calibrated probabilities.
A firing strength stays a degree of match and a weight a fitted coefficient; neither is a probability,
a causal effect or a risk. ``explainable_predict`` refuses the mixture, because no single rule decides.

Pickling keeps the weights. The text format of :doc:`persistence` stores the rules only. The weights are
checked against the rules they were fitted for: after the rules change (``select_pareto_solution``,
removing a rule) ``predict`` asks for ``fit_mixture`` again. A classifier pickled by a version without the
option keeps winning-rule inference; set ``model.inference = 'mixture'`` and call ``fit_mixture`` to give it
weights (``set_params`` needs every current constructor argument on the object).
