# Ex-Fuzzy speedup validation

The figure compares **Ex-Fuzzy 2.0** (the preserved full object reference
objective) with **Ex-Fuzzy 3.0** (the current default evaluator) in the same
checkout. These are evaluator labels, not measurements of two release wheels.
The reference uses `FitRuleBase._evaluate_slow`, disables array evaluation and
restores the per-rule firing loop. Shared correctness fixes, optimizer,
partition normalization and final model construction remain the same. This
isolates evaluator improvements without confounding changed release semantics.

## Reproduce

From the repository root, with the package installed from source:

```bash
python -m pip install -e .
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python benchmarks/benchmark_speedup.py
```

Run without competing compute workloads. The script creates `speedup.json`
and `speedup.svg` here, and refuses to publish either if parity fails.
To redraw from the recorded raw measurements:

```bash
python benchmarks/benchmark_speedup.py --plot-only
```

## Measurement design

- Synthetic three-class classification, 10 features (9 informative), data seed
  42; 100, 1,000 and 5,000 training samples.
- T1 and interval T2; fixed and genetically optimized membership partitions.
- 20 rules, 4 antecedent slots, population 40, 5 generations, no early stopping;
  fit seeds 7, 19 and 41, identical between evaluators.
- Each fit runs in a fresh subprocess; the six runs per workload are shuffled
  deterministically. Numerical library thread counts are fixed to one by the
  reproduction command. No GPU or compiled prototype is used.
- The timer covers `fit`, including route probing, cache setup and finalization.
  Imports, synthetic data generation, fixed partition construction and prediction
  checks are outside the timer. First-use costs inside `fit` remain included.
- Bars are median seconds across the three seeds; whiskers are the observed
  minimum and maximum, **not confidence intervals**. Speedups are ratios of
  medians. Raw times, environment and workload settings are retained in JSON.
  These short fits expose setup overhead; they do not establish performance for
  longer searches, other datasets, hardware, custom losses or other backends.

## Exactness gates

Every timed reference/current pair must match the byte-level SHA-256 trace of
all evaluated chromosome and objective arrays, including rejected offspring,
not just the final winner. Hashing includes shape and dtype and is included in
both timers. Final populations, evaluation counts, selected chromosome,
performance, rule matrices, consequents, scores, training predictions and 101
additional deterministic probe predictions must also match exactly. These
probe points are synthetic checks of model equivalence, not an accuracy test.
The JSON retains the matching trace and evaluation count for each seed.

`tests/test_speedup_validation.py` repeats complete-search comparisons across
T1/T2, both partition modes and all three dominance-score modes, with pruning
and unknown predictions enabled. It also checks that the benchmark rejects
changed results. Existing candidate, firing, cache and dispatch suites cover
penalties, degenerate candidates, rounding-sensitive reductions, eligibility
fallbacks and cache/pool cleanup. Tests deliberately avoid wall-clock pass/fail
thresholds, which would be unreliable across CI machines.

```bash
python -m pytest -q tests/test_speedup_validation.py \
  tests/test_array_evaluation.py tests/test_genetic_fitness_semantics.py \
  tests/test_fast_fitness.py tests/test_fitness_cache.py tests/test_firing_cache.py \
  tests/test_population_evaluation.py tests/test_route_dispatch.py \
  tests/test_gather_firing.py tests/test_dominance_grouped.py \
  tests/test_runner_lifecycle.py tests/test_finalization_work.py
```

## Recorded validation

The recorded measurements contain 72 complete fits (36 matched pairs) and
7,200 evaluated candidate positions checked pairwise. All pairs passed. Median
complete-fit gains range from **10.7× to 31.2×** on an AMD Ryzen 5 5600X, using
NumPy 2.3.5, PyMoo 0.6.1.6 and scikit-learn 1.9.0. No stored dispatch profile
was present for this run.

Targeted validation: **329 tests passed, 1 skipped**, combining the new
end-to-end tests with the suites listed above. The skip is the existing
scikit-learn estimator-introspection compatibility check. CPU objective
comparisons in `test_genetic_fitness_semantics.py` now require exact equality;
the separate EvoX float32 custom-loss bridge retains its tolerance-based test
and is outside this CPU speedup claim.
