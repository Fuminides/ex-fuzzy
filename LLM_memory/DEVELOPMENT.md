# Development context

## Repository and conventions

Ex-Fuzzy is an AGPL v3 Python library for interpretable fuzzy rule learning.
The Python package is `ex_fuzzy/ex_fuzzy/`; tests are in `tests/`, opt-in
benchmarks in `benchmarks/`, examples in `Demos/`, and Sphinx sources in
`docs/source/`. Check packaging metadata for current versions and dependencies.

Use PascalCase classes, snake_case functions, UPPER_CASE constants, leading
underscores for private helpers, type hints, and Google-style docstrings.
Preserve relative imports and existing direct-execution fallbacks where needed.
Load optional dependencies lazily and provide actionable installation errors.
Classifiers follow the scikit-learn `fit`/`predict` API.
Never use interactive Git commands.

## Code map

All module paths below are relative to `ex_fuzzy/ex_fuzzy/`.

| Area | Modules |
| --- | --- |
| Fuzzy sets and linguistic variables | `fuzzy_sets.py`, `utils.py`, `temporal.py`, `centroid.py` |
| Rule representation, inference, firing kernels | `rules.py` |
| Genetic classification | `evolutionary_fit.py`, `evolutionary_backends.py`, `evolutionary_search.py` |
| Exact built-in objective | `_fitness.py`, `_array_fitness.py` |
| Population route selection | `_dispatch_profile.py` |
| Regression | `evolutionary_fit_regression.py` |
| High-level classifiers and mining | `classifiers.py`, `rule_mining.py` |
| Evaluation and visualization | `eval_tools.py`, `eval_rules.py`, `vis_rules.py` |
| Statistical analysis | `bootstrapping_test.py`, `permutation_test.py`, `pattern_stability.py` |
| Persistence | `persistence.py` |
| Conformal prediction | `conformal.py` |
| Other learners | `ferl.py` (compact and medium FERL), `ferl_deep.py` (`DeepFERL`), `_evidence.py` (shared Dempster--Shafer combination), `ferl_partitions.py`, `tree_learning_new/`, `cognitive_maps.py` |

PyMoo is the default backend and supports checkpoints. Consult
[the EvoX documentation](../docs/source/evox_backend.rst) and implementation for
current device support; old JAX descriptions and blanket GPU speed claims are stale.
Fuzzy sets include Type-1, interval Type-2, and general Type-2; fast paths support
only subsets of those cases. Conformal prediction requires a held-out calibration
set separate from training and final evaluation data.

## Performance invariants

Before changing `FitRuleBase`, `_fitness.py`, `_array_fitness.py`, or firing
kernels in `rules.py`, read [SPEED_UP.md](SPEED_UP.md) and
[SPEED_UP_REVIEW.md](SPEED_UP_REVIEW.md).

- Preserve objective values bit for bit, including reduction and pruning order.
  Numerical closeness is insufficient: small changes can alter the entire search.
- The object constructor and `_fitness.score_rulebase` remain the oracle and
  fallback, and build final models, checkpoints, and objects for custom losses.
  `FitRuleBase._array_score` returns `None` for unsupported cases.
- NumPy reduction layout and pairwise summation affect exactness. Compare new
  reductions with the reference across relevant types and array layouts.
- Fit-local fitness/firing caches and packed memberships must remain bounded and
  scoped to a fit, with cleanup on success and failure. Do not make them global.
- Respect fast-path eligibility and preserve logical evaluation counts even on
  cache hits. Keep custom-loss, checkpoint, and worker fallbacks intact.
- Preserve reference duplicate-rule behavior, including known hash/equality
  quirks, unless a separately scoped semantic change explicitly addresses it.
- Do not present kernel timings as complete-fit gains or benchmark prototypes
  as shipped capabilities.

## Validation

Install from source with `pip install -e .`; use `pip install -e ".[evox]"`
when the optional backend is needed. Run relevant tests with `pytest tests/` or
selected test files. Fixtures live in `tests/conftest.py`.

For evaluator changes, relevant suites include `test_array_evaluation.py`,
`test_genetic_fitness_semantics.py`, `test_fast_fitness.py`, cache tests,
`test_population_evaluation.py`, and `test_route_dispatch.py`.
Measure complete seeded fits with
`python benchmarks/benchmark_evaluator_variants.py`; it refuses timings unless
variants produce identical fitted models. Avoid competing workloads during timing.
See the speedup records for workload-specific reproduction commands and limitations.

The README accuracy figure comes from `benchmarks/benchmark_keel.py`, one
result per (dataset, method) pair, spread over CERES by
`benchmarks/cluster/submit_keel.sh` and published by
`benchmarks/aggregate_keel.py`, which refuses to publish an incomplete grid.
Read [the KEEL methodology](../docs/performance/KEEL.md) before changing the
protocol or quoting its numbers. It measures defaults plus a stated GA search
budget, not tuned models.

`DeepFERL` ports `LearnedFuzzyTree` from `../fuzzy_greedy_tree` (FERL-deep in
that paper). `tests/test_ferl_deep.py` pins its trees and predictions against
a golden fixture, and runs a live bit-for-bit comparison when that checkout exists.
`tests/test_ferl_evidence.py` pins `FERL.predict_ds` to outputs recorded before
the combination rules moved into `_evidence.py`. Keep both pins passing.
