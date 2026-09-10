====================
Training Performance
====================

Genetic training in Ex-Fuzzy runs through several optimizations that avoid work
without changing what the search finds. They are automatic: there is no flag to
turn them on, and no public option changes because of them.

This page explains what those optimizations are, what switches each one on, and
what to do when training is slower than you expect or when you suspect a fast
path of misbehaving. You do not need any of this to use the library — it is here
for when you want to know why a fit took the time it did.

The guarantee
=============

Every optimization described here is **bit-for-bit exact**. A fit produces the
identical fitness values, the identical selected chromosome, the identical rules
and the identical predictions whether or not any fast path is active.

This is not merely a design goal; it is an acceptance criterion. The genetic
search is chaotic, so a difference in the last bits of one candidate's score can
change which rules survive and produce a visibly different model. The
optimizations are therefore tested against the original implementation, which is
kept as the reference, and the benchmarks refuse to report a timing unless every
variant produced the identical fitted model.

The practical consequence: **if turning an optimization off changes your
results, that is a bug.** :ref:`Reporting one <performance-suspected-bug>` is
described below.

What happens during a fit
=========================

A candidate rule base is scored in one of two ways.

**The object path** is the original implementation. It decodes a chromosome into
``RuleSimple`` objects and a ``MasterRuleBase``, then scores it. This path still
exists and still runs: it is the reference the fast paths are tested against, it
handles every case the fast paths decline, and it builds the final model,
checkpoints and anything a custom loss receives.

**The array path** computes the same objective directly from arrays, without
building any rule object for a candidate that is only going to be scored and
discarded. Rule-base construction alone accounted for roughly a third of fit
time before this existed.

On top of that, three further layers avoid repeating work:

- **Precomputed problem data.** Normalization bounds, the integer label layout
  and, for fixed partitions, a packed table of membership values are built once
  per fit and reused by every candidate.
- **Fit-local caches.** A chromosome that the search generates twice is scored
  once. With fixed partitions, firing strengths for repeated rules are reused
  across candidates as well. Both caches are bounded in size, live only for the
  duration of one ``fit()`` call, and are discarded afterwards — including when
  the fit raises.
- **Population batching.** Instead of scoring candidates one at a time, an
  entire generation is decoded, fired, scored and pruned in single array
  operations. This removes per-candidate interpreter overhead, which dominates
  on small datasets.

Caching does not change how many evaluations the search performs. A repeated
chromosome still counts as an evaluation; it is simply not recomputed.

What enables each fast path
===========================

Ex-Fuzzy falls back to a slower but equivalent path whenever it cannot prove a
faster one applies. This table is the quickest way to find out why a fit is not
as fast as another one.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Optimization
     - Active when
   * - Array evaluator
     - The built-in objective (no ``custom_loss``), Type-1 or Type-2 sets, and
       ordinary — not temporal — partitions.
   * - Fit-local caches
     - The built-in objective, the ``pymoo`` backend, no thread runner and no
       checkpointing. The firing cache additionally needs fixed partitions.
   * - Population batching
     - Everything the caches need, plus Type-1 sets and fixed partitions.

Two entries in that table are worth spelling out, because they surprise people:

**Fixed partitions.** Passing ``linguistic_variables=`` fixes the fuzzy sets, so
membership values can be computed once for the whole fit. Letting the optimizer
tune the partitions instead means the same antecedent index denotes a different
fuzzy set in every candidate, so memberships must be recomputed and firing
strengths cannot be reused. Optimizing partitions is a legitimate choice — it
simply costs more per candidate.

.. code-block:: python

   from ex_fuzzy import BaseFuzzyRulesClassifier, utils, fuzzy_sets as fs

   partitions = utils.construct_partitions(X_train, fs.FUZZY_SETS.t1)
   clf = BaseFuzzyRulesClassifier(nRules=20, nAnts=4,
                                  linguistic_variables=partitions)

**Threads and checkpoints turn the caches off.** A thread runner or a
checkpointing fit does not have the private, stable fit context the caches
require, so both are disabled. Because the caches are substantial, a threaded
fit is not reliably faster than a serial one — measure before assuming it is.

A custom loss disables the caches and the array evaluator together, since
Ex-Fuzzy cannot know that your objective depends only on the rule base. Your
loss still receives an ordinary ``MasterRuleBase``, exactly as before.

How the batching route is chosen
================================

Batching a whole generation is faster on small datasets and *slower* on large
ones, because the intermediate arrays grow with the number of samples. The
crossover point depends on your data shape and on the machine's cache and memory
bandwidth, so Ex-Fuzzy measures it rather than assuming it.

Early in each eligible fit, a few generations run alternately on each route and
their cost per candidate is recorded. The cheaper route is then used for the
rest of the fit. Since both routes compute the same objective, this changes only
speed. Probing duplicates no work — every probing generation is a generation the
fit had to run anyway — and costs a few percent of total fit time, less the
longer the fit.

If you would rather not pay even that, you can measure the choice once, offline,
and store the result:

.. code-block:: bash

   python benchmarks/calibrate_population_dispatch.py

This sweeps a grid of data shapes, rule counts and population sizes, verifies
that both routes produce identical fits at every point, and writes a profile to
``~/.cache/ex-fuzzy/dispatch_profile.json``. Later fits look the answer up
instead of measuring it. Set ``EX_FUZZY_DISPATCH_PROFILE`` to store it
elsewhere, and pass ``--quick`` for a smaller sweep or ``--dry-run`` to see the
measurements without writing anything.

The profile is optional and safe to delete: without one, fits simply probe. It
records the machine, Python and NumPy version it was measured on and is ignored
if any of those differ, so a profile copied to another machine cannot mislead a
fit. It affects only which route runs, never a result.

When training is slower than expected
=====================================

Work down this list:

1. **Check the table above.** A custom loss, a thread runner, checkpointing,
   optimized partitions, Type-2 sets or temporal partitions each disable one or
   more fast paths. This explains most large differences between two fits.
2. **Scale the search, not the data.** ``n_gen`` and ``pop_size`` multiply
   directly into the number of evaluations. Start small and increase.
3. **Reconsider threads.** They disable both caches, so a serial fit is often
   faster. Compare with the same seed before committing.
4. **Expect large datasets to cost more per candidate, not less.** Beyond a few
   thousand samples the per-candidate work is dominated by the data itself, and
   the optimizations that remove fixed overhead have proportionally less to give.

For a rough sense of scale: on synthetic 1,000-sample data, the optimizations
described here take a complete fit from about 4.0 s to about 0.5 s with
optimized partitions, and from about 4.5 s to about 0.5 s with fixed partitions.
These are local measurements on one machine, not a promise for your workload.

.. _performance-suspected-bug:

When you suspect a fast path is wrong
=====================================

Because every optimization is exact, you can test the hypothesis directly:
disable them and see whether your results change.

.. code-block:: python

   import ex_fuzzy.evolutionary_fit as evf

   evf.FitRuleBase.array_evaluation = False   # force the original object path

This single switch turns off both the array evaluator and population batching,
sending every candidate through the original rule-object implementation. Restore
it by setting it back to ``True``.

To compare properly, fit twice with the same ``random_state`` and check that the
results match:

.. code-block:: python

   import numpy as np
   import ex_fuzzy.evolutionary_fit as evf

   def fitted():
       clf = evf.BaseFuzzyRulesClassifier(
           nRules=20, nAnts=4, linguistic_variables=partitions)
       clf.fit(X_train, y_train, n_gen=30, pop_size=40, random_state=0)
       return clf.performance, clf.predict(X_test)

   fast_score, fast_pred = fitted()
   evf.FitRuleBase.array_evaluation = False
   slow_score, slow_pred = fitted()
   evf.FitRuleBase.array_evaluation = True

   assert fast_score == slow_score
   assert np.array_equal(fast_pred, slow_pred)

Both fits must agree exactly, not approximately. If they do not, please
`report it <https://github.com/fuminides/ex-fuzzy/issues>`_ with the two scores,
your data shape, the fuzzy set type and whether you passed
``linguistic_variables``. That is a real defect and the assertion above is
enough to reproduce it.

If the results *do* agree and the fast path is simply slower for your workload,
leaving ``array_evaluation = False`` is a valid workaround, though the route
probe already avoids the batched path when it does not pay.

Note that this is a class-level attribute, not a constructor argument: it affects
every classifier created afterwards in the same process. It exists for
diagnostics and parity testing rather than as a tuning knob.

Further reading
===============

The development record behind these optimizations lives in the repository rather
than in this guide. ``LLM_memory/SPEED_UP.md`` documents each change, the measurements
behind it and the reasoning; ``LLM_memory/SPEED_UP_REVIEW.md`` catalogues every proposal
that was considered, including the ones that were measured and rejected — which
is often the more useful half if you are thinking of contributing an
optimization of your own.
