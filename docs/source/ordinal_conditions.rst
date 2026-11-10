===================================
Ordinal "at least / at most" rules
===================================

A condition of a rule is by default one linguistic term: ``IF no2 IS Fair``. When the terms of a variable are
ordered (Low, Medium, High, ...) a rule often means a range: "Fair **or worse**", "Medium **or better**".
``ordinal_conditions=True`` lets the genetic search write such conditions:

.. code-block:: text

   IF no2_last IS at least Fair AND humidity IS at most Medium WITH DS 0.41, ACC 0.83

* ``x IS at least t`` is the union of ``t`` and every term above it (up to the highest term).
* ``x IS at most t`` is the union of every term below ``t`` (from the lowest term) and ``t``.

The terms of a variable are ordered as they are listed in its partition, from low to high. The partitions the
library builds (``utils.construct_partitions``) are listed in that order; a partition you give is checked and
a ``UserWarning`` says when its terms are not listed from low to high.

Two further options select where the conditions apply and add **named compound terms** (see below):
``ordinal_conditions`` also takes a collection of features, so that a cyclic variable such as a clock hour does
not get ``at most Medium``, and ``compound_terms`` adds named unions of terms that need not be contiguous.

The option is **off by default**. Off, nothing changes: the gene encoding, the rules, the printed text, the
saved files and every seeded fit are those of 3.4.0 (a test replays nine fits recorded on that release).

Usage
=====

.. code-block:: python

   from sklearn.datasets import load_iris
   from ex_fuzzy import BaseFuzzyRulesClassifier

   X, y = load_iris(return_X_y=True)
   model = BaseFuzzyRulesClassifier(nRules=15, nAnts=2, n_linguistic_variables=4, ordinal_conditions=True,
                                    n_gen=30, pop_size=50, random_state=0, patience=None)
   model.fit(X, y)
   model.print_rules()
   # Rules for consequent: 2
   # IF 0 IS at least Medium AND 3 IS at least High WITH DS 0.19, ACC 0.96

   # The operator of every condition, without parsing the text:
   for rule in model.rule_base.get_rules():
       for condition in rule.conditions(model.rule_base.antecedents):
           print(condition.variable, condition.operator, condition.term, condition.terms)
   # ...
   # 3 at_most Medium ['Low', 'Medium']
   # 0 at_least Medium ['Medium', 'High', 'Very High']
   # 3 at_least High ['High', 'Very High']

The option exists on the three estimators the search runs through: ``BaseFuzzyRulesClassifier`` (winning-rule
and mixture inference, also with ``mixture_search``), ``BaseFuzzyRulesRegressor`` (crisp and fuzzy
consequents) and the problems behind them, ``FitRuleBase`` and ``FitRuleBaseRegression``. It works with
searched partitions and with ``linguistic_variables`` you give (the given variables are not modified: the
search uses copies), and with Type-1 and interval Type-2 sets. It is refused for general Type-2 sets and
together with ``candidate_rules`` (which only chooses among given rules). Categorical variables keep single
terms, since their categories are not ordered; so do variables with fewer than 3 terms.

Per-variable selection
======================

``ordinal_conditions`` accepts ``True`` (every variable that can have them), ``False`` (the default) or a
**collection of features**, by name or by index (a list, tuple, set or array; names are the variable names, or
the column names of a DataFrame):

.. code-block:: python

   BaseFuzzyRulesClassifier(linguistic_variables=variables, ordinal_conditions=['temp', 'level'])
   BaseFuzzyRulesClassifier(n_linguistic_variables=4, ordinal_conditions=[1, 2])   # by index

The other variables keep single terms (and any compound terms). A feature named explicitly must be able to
have the conditions (3 or more terms, not categorical), otherwise a ``ValueError`` names it; an unknown name
or an index out of range is an error too. A bare string, ``{'a': 1}`` or a number is rejected: pass a list.
With ``True`` the variables that cannot have them are skipped quietly. An empty collection, ``False`` and
``None`` are off, and then the problem is built exactly as in 3.4.0.

Named compound terms
====================

Some variables are not ordinal in the way ``at least`` assumes. The terms of a clock hour are cyclic, so a
useful condition is ``local_hour IS Night`` = the last term **or** the first one. ``compound_terms`` adds such
conditions explicitly:

.. code-block:: python

   model = BaseFuzzyRulesClassifier(
       linguistic_variables=variables,
       ordinal_conditions=['temp'],                       # not the clock
       compound_terms={'local_hour': [('Night', ['Very High', 'Low']),   # wraps around midnight
                                      ('Daytime', ['Medium', 'High'])]},
       ...)
   # IF local_hour IS Night AND temp IS at least Medium WITH DS ..., ACC ...

* **Shape.** ``compound_terms`` is ``None`` or a mapping ``{feature: [(name, [term, ...]), ...]}``. A feature is
  a name or an index (as in ``ordinal_conditions``); each term is a term name or a term index. Every entry adds
  one condition ``x IS <name>`` to that variable, whose membership is the bounded sum of the listed terms. The
  terms need not be contiguous, so a union that wraps around is allowed. The set is kept sorted low to high
  (``[last, first]`` and ``[first, last]`` are the same condition).
* **Validation** (``ValueError`` with the variable and the compound named): at least 2 distinct terms (a term
  listed twice is an error); not every term of the variable (it would be a don't care); an unknown term name or
  an index out of range; no two compounds of a variable with the same set; no compound with the set of an
  ordinal condition the variable has (for example ``[T1, T2, T3]`` with ``at least T1``); the **name must be
  unique within its variable**: it cannot be the name of a term, of another compound, or of an ordinal
  condition, and it cannot be empty, contain ``;``, line breaks, parentheses or the words ``AND``, ``IS``,
  ``WITH``, ``THEN``, nor start with ``at least`` / ``at most`` (the rule text must be readable back). Two
  keys for the same feature are an error too.
* **Together or alone.** Compound terms work with or without ``ordinal_conditions`` on the same variable, with
  searched partitions (terms by name ``Low``... or index) or ``linguistic_variables``, with Type-1 and interval
  Type-2 sets, and also on categorical variables (any set of categories is a valid union). They are refused for
  general Type-2 sets and with ``candidate_rules``. A variable needs 3 or more terms (2 of 2 terms is a don't
  care). With ordinal conditions on, a contiguous run that touches an end repeats one of them (with 3 terms
  ``[T0, T1]`` is ``at most T1``) and is refused; ``[T0, T2]`` is not and is accepted.
* **Membership.** The bounded sum of the listed terms, as for ``at least`` / ``at most``. A union that wraps
  around on a linear axis is **two shoulders, one at each end**: for ``[last, first]`` it is 1 at both ends of
  the domain, falls over the ramp of the first term, is 0 in the middle and rises over the ramp of the last
  term. On a strong partition it equals one minus the memberships of the other terms. Nothing here knows the
  axis is cyclic: a value just before and just after midnight are the two ends and both are covered.
* **Codes.** After the single terms and the ordinal conditions: with ``n`` terms, the compounds are
  ``n_conditions - len(compound) ... n_conditions - 1``, in the order given (``condition_code('any_of', name)``).
  They go through the same machinery (rows of ``compute_memberships``, the gather table, the remap, canonical
  codes for duplicates, and the array, ``batch_size``, population and PyTorch objectives; the latter declines
  with searched partitions).

A variable can be built with them directly: ``fuzzyVariable(name, sets, ordinal_conditions=False,
compound_terms=[(name, [terms])...])`` or ``fuzzy_sets.enable_conditions(variables, ordinal_conditions,
compound_terms)``.

A variable can also offer the conditions on its own, for any code that reads rules:
``fuzzyVariable(name, sets, ordinal_conditions=True)`` or ``fuzzy_sets.enable_ordinal_conditions(variables)``.

Which conditions exist
======================

For a variable with ``n`` ordered terms ``t_0 ... t_(n-1)``: the ``n`` single terms, ``at least t_j`` for
``j = 1 ... n - 2`` and ``at most t_j`` for ``j = 1 ... n - 2``, that is ``3n - 4`` conditions (8 for 4 terms,
5 for 3). ``at least t_0`` is always true (a don't care) and ``at least t_(n-1)`` is ``t_(n-1)``, and likewise
``at most t_(n-1)`` and ``at most t_0``, so they are not offered. General "between" intervals are not offered.
No two conditions cover the same set of terms, so a set has exactly one code.

Membership
==========

The membership of a union is the **bounded sum**, the Lukasiewicz t-conorm:

.. math::

   \mu_{\text{at least } t_j}(x) = \min\Bigl(1, \sum_{k \ge j} \mu_{t_k}(x)\Bigr), \qquad
   \mu_{\text{at most } t_j}(x) = \min\Bigl(1, \sum_{k \le j} \mu_{t_k}(x)\Bigr)

* **Strong partitions.** On a strong (Ruspini) partition, where the memberships of a point add up to 1, the
  sum never exceeds 1 and ``at least t_j`` is exactly ``1 - (memberships of t_0 ... t_(j-1))``: again a
  shouldered trapezoid, with the ramp of the term below ``t_j``. The partitions built by
  ``utils.construct_partitions`` (trapezoidal) and ``construct_breakpoint_partition`` are strong. The terms of a
  partition found by the search (``linguistic_variables=None``) never add up to more than 1 but can add up to
  less (a point between two terms may belong to neither fully); the bounded sum is then just the sum, and
  ``at least t_j + at most t_(j-1)`` is below 1 there.
* **Partitions that are not strong.** Gaussian terms overlap more than a partition of unity: their sum
  exceeds 1 and the bounded sum saturates at 1 there. Nothing else changes.
* **Interval Type-2.** The lower and the upper bound are summed apart (and clipped to 1 each), which is the
  interval-arithmetic image of the bounded sum.
* **t-norm.** A condition is one antecedent with a membership in [0, 1]; the rule still combines its antecedents
  with its t-norm (the product by default), so the union takes part exactly as a single term would. The bounded
  sum is the dual of the Lukasiewicz t-norm, not of the product, so a union is not the De Morgan dual of a
  product intersection of complements; it is chosen because it keeps the trapezoids and
  ``at least t_j + at most t_(j-1) = 1`` on strong partitions.
* A firing strength is still a degree of match, not a probability, contribution or risk.

How the search encodes it
=========================

The term gene is widened, no gene is added. Without the option a term gene of slot ``i`` holds ``-1`` (don't
care) or a value in ``0 ... max_terms - 1``, which the decoder maps onto the terms of whichever variable the
slot's variable gene selected (``_fitness.remap_term``). With the option the gene holds a **condition code**
in ``0 ... max_codes - 1``, and the same proportional map is applied to the code count of the variable
(``FitRuleBase.n_term_codes``; the number of terms when the option is off):

======================  ===============================
code                    condition
======================  ===============================
``0 ... n-1``           ``IS t_code`` (single terms)
``n ... 2n-3``          ``IS at least t_j``, ``j = code - n + 1``
``2n-2 ... 3n-5``       ``IS at most t_j``, ``j = code - (2n-2) + 1``
======================  ===============================

``RuleSimple.antecedents`` stores the same code (``-1`` for a don't care), so a rule is identified, compared,
hashed and de-duplicated by its codes as before; because a set of terms has one code, two rules whose
conditions are the same sets are duplicates in the decoders (object, array, population and PyTorch), in
``RuleBase`` and in the mixture. Sampling, crossover and mutation work on the gene bounds, which are the
wider ones, so they need no change. The fitness routes (array, streamed with ``batch_size``, population and the
PyTorch objective on fixed partitions) take the extra conditions as extra columns of the same membership
table and agree with the object decoder bit for bit. With ``linguistic_variables=None`` the PyTorch objective,
which builds trapezoids itself, declines and the CPU scores the fit.

``fuzzyVariable.condition(code)`` gives ``(operator, term_index)``, ``condition_code(operator, term_index)``
the inverse, ``condition_terms(code)`` the terms covered and ``condition_names()`` the printed names.
``rules.rule_conditions(rule, antecedents)`` (also ``rule.conditions(antecedents)``) returns ``Condition``
tuples ``(feature, variable, operator, term, term_index, terms, code, name)`` with ``operator`` one of ``'is'``,
``'at_least'``, ``'at_most'`` and ``'any_of'``. For the first three ``term`` / ``term_index`` are the term the
operator refers to, ``terms`` the terms covered and ``name`` is ``None``; for ``'any_of'`` ``term`` and
``term_index`` are ``None``, ``terms`` lists the constituent term names (low to high) and ``name`` is the given
name of the compound term. ``fuzzyVariable.condition(code)`` returns ``('any_of', position in
variable.compound)`` for it.

Search space and budget
=======================

A variable of ``n`` terms offers ``3n - 4`` conditions instead of ``n``: 5 instead of 3, 8 instead of 4. The
gene length is unchanged but the space is larger, and uniform sampling draws a single term in only ``n / (3n - 4)``
of the cases (60 % for 3 terms, 50 % for 4). A search with the same ``n_gen`` and ``pop_size`` therefore has
less of its budget on single terms; if the comparison with the default must be fair, give the option a larger
budget or compare the best of several seeds (see ``benchmarks/benchmark_ordinal_conditions.py``). Each
condition still counts as one in the compactness penalty and in ``nAnts``.

Printing, saving and loading
============================

``print_rules`` writes ``IS at least Fair`` / ``IS at most Fair`` and ``IS <name>`` for a compound term (also ``print_mixture``, the regression
rules and ``vis_rules.rules_to_latex``, which writes ``$\geq$ Fair`` / ``$\leq$ Fair``). ``persistence.load_fuzzy_rules``
reads them back: a variable without ordinal conditions in the list you pass is replaced, in the returned rule
base, by a copy that has them; the name of a compound term needs the variables to carry it, so load the
variables first. ``save_fuzzy_variables`` writes the line ``Ordinal conditions`` after the header of a
variable that has them and one line ``Compound;<name>;<term indexes>`` per compound term (variables without
either are written exactly as before), and ``load_fuzzy_variables`` reads them. Pickles keep the option; a classifier, regressor or variable pickled before
it existed loads with it off. ``pattern_stability`` rule keys name the condition (``('x', 'at least Fair')``)
and ``consensus_rule_base`` rebuilds them.

Not covered
===========

``FuzzyRulesClassifier`` (fuzzy association rules), ``RuleMineClassifier``, ``RuleFineTuneClassifier``, the
FERL learners and the null distribution of the bootstrapped rule p-values (random rules over single terms)
do not take the option.
