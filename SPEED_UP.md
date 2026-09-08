# Genetic training speedup plan

**Status: proposals for user review; no new implementation is authorized by this document.**

Scope: `BaseFuzzyRulesClassifier`, primarily its built-in classification objective
and `FitRuleBase`. This is a comprehensive catalogue of practical optimization
families, including speculative alternatives; it is not a claim that every
conceivable optimization has been enumerated. Select individual IDs below before
implementation. Recommendations are proposals, not selected work.

Preserve existing constructor, `fit()`, prediction, explanation, checkpoint and
custom-loss interfaces wherever possible. New configuration flags or changes to
search behavior need an explicit decision. Do not silently replace the algorithm
with an approximation to obtain a faster benchmark.

### Measured baseline

Local x86-64 environment: Python 3.12.3, NumPy 2.4.2, pymoo 0.6.1.6.
Synthetic classification: 1,000 samples, 10 features, 3 classes, 20 rules,
4 antecedent slots, population 40, 5 generations, seed 7. Median of three
complete fits, early stopping disabled, no concurrent test run.

| Partition mode | Corrected full reference | Current optimized evaluator | Fit speedup |
| --- | ---: | ---: | ---: |
| Membership functions optimized | 3.96 s | 0.52 s | 7.6× |
| Fixed membership functions | 4.48 s | 0.47 s | 9.4× |

The benchmark checked exact equality of evaluated fitness values, selected
chromosomes, final rule scores, rule matrices and predictions. These are local
measurements, not promised speedups for other workloads. Compare future work
against the **current optimized baseline**, not only the much slower reference.

A subsequent profile of 100 chromosomes on the same data shape found:

| Component | Optimized partitions | Fixed partitions |
| --- | ---: | ---: |
| Rule-base construction | about 34% | about 30% |
| Rule firing-strength calculation | about 27% | about 35% |
| Fuzzy-set membership calculation | about 15% | precomputed |
| Dominance scoring | about 15% | about 22% |

These are approximate cumulative profile fractions, with instrumentation
overhead. Nested profile entries must not be added as independent costs.
Eliminating a component taking one third of total time can at most give about
1.5× overall speedup by itself; kernel speedups do not multiply automatically.

## 1. Decision catalogue

For each selected ID, record **execute / investigate / defer / reject**.
All entries below are currently **pending**.

Effort is relative: S = localized work, M = several coordinated changes,
L = substantial evaluator or backend work. Benefits are hypotheses unless
explicitly measured above. “Exact” means equivalence is the required acceptance
criterion, not an assertion that an implementation is already proven exact.

### A. Reduce CPU work while preserving the objective and search

| ID | Proposal | Expected opportunity | Effort / main risk |
| --- | --- | --- | --- |
| A01 | Direct indexed membership gathering for rule firing | High: targets a measured hot path | M; preserve reduction order, don't-cares, modifiers and T2 axes |
| A02 | Reuse bounded scratch buffers and improve array layout | Medium: fewer allocations and copies | S–M; thread isolation, aliasing and floating-point reduction changes |
| A03 | Precompute immutable problem metadata | Small–medium: less repeated bounds/layout/label work | S; invalidate on refit or configuration changes |
| A04 | Array-based chromosome decoder | High: avoid building Python rules/sets for every candidate | L; normalization, duplicate handling and ordering parity |
| A05 | Direct vectorized membership evaluation from decoded parameters | Medium–high when optimizing fuzzy sets | M–L; endpoint semantics and custom-set fallbacks |
| A06 | Fuse or vectorize dominance/support/confidence reductions | Medium: measured scoring cost remains | M; reference reduction order and T2 pooling |
| A07 | Pre-encode integer labels and reuse class masks/count layout | Small–medium, especially small datasets | S; unknown/non-contiguous labels and memory for masks |
| A08 | Compute pruning and rule-complexity penalties entirely on arrays | Medium after A04; fewer object traversals | M; pre-pruning winners, empty classes, strict thresholds |
| A09 | Exact shortcuts for degenerate candidates | Workload-dependent: empty/disabled/zero-firing candidates | S–M; prove the returned objective including penalties and unknowns |
| A10 | Avoid redundant finalization/checkpoint work | Usually small, larger with frequent checkpoints | M; preserve all reported metrics and callback behavior |
| A11 | Improve Python duplicate-rule detection without changing results | Medium if many repeated rules | M; match the constructor's retention and ordering exactly |

### B. Reuse work between candidates, with explicit dependency keys

| ID | Proposal | Expected opportunity | Effort / main risk |
| --- | --- | --- | --- |
| B01 | Measure genotype and decoded-rule duplication rates | Determines whether caching is worthwhile | S; measurement must not change sampling |
| B02 | Bounded memoization of exact chromosome fitness | Potentially high with repeated deterministic candidates | M; include full fit context, preserve logical evaluation counts |
| B03 | Within-generation duplicate evaluation sharing | Avoid repeated work while retaining duplicate population members | M; scatter results back in original order, preserve RNG/callbacks |
| B04 | Cache firing strengths of identical rules for fixed partitions | Potentially high when rule fragments recur | M; memory grows with samples × cached rules |
| B05 | Cache identical optimized partitions per feature | Conditional: mutations may leave some features unchanged | M–L; feature-wide normalization dependencies, cache hit rate |
| B06 | Incremental evaluation from parent to offspring | Potentially high for sparse mutations | L; crossover provenance and rule competition invalidate many quantities |
| B07 | Cache decoded phenotype rather than raw genotype | More cache hits from semantically redundant encodings | L; prove equality including rule order, weights and float parameters |

Do not confuse B03 with enabling pymoo's duplicate elimination. The current GA
sets `eliminate_duplicates=False`. Sharing an evaluation preserves population
members; eliminating and replacing them changes the search (see E03).

### C. Batching, parallelism and memory

| ID | Proposal | Expected opportunity | Effort / main risk |
| --- | --- | --- | --- |
| C01 | Chunked population evaluation with fixed partitions | High for sufficient population/sample sizes | L; padded rule masks, stable ordering and peak memory |
| C02 | Chunked population evaluation with optimized partitions | High on large workloads | L; separate memberships for every chromosome |
| C03 | Sample-wise streaming for large datasets | Primarily memory/scalability; may be slower | L; multiple passes and changed summation order |
| C04 | Benchmark and tune the existing thread runner | Conditional after reducing Python work | S–M; GIL, scheduling cost, nested numeric threads |
| C05 | Persistent process workers with shared read-only data | Conditional for expensive Python-heavy evaluations | M–L; serialization, startup, Windows spawn and custom losses |
| C06 | Compiled parallel loops over independent candidates/rules | Potentially high after native kernels exist | M–L; OpenMP/toolchain availability and thread oversubscription |
| C07 | Control concurrency, pool lifetime and nested thread counts | Reliability and workload-dependent speed | M; avoid global settings that surprise embedding applications |
| C08 | Parallelize independent fits, folds, seeds or tuning trials | High total experiment throughput | M; does not accelerate a single fit; avoid nested pools |
| C09 | Multi-node evaluation or multi-GPU sharding | Relevant only at substantially larger scale | L; scheduling, network transfer and deterministic result ordering |
| C10 | Memory-mapped/chunked data, compact index dtypes and layout tuning | Large-data capacity, possible bandwidth gains | M; bounds/overflow checks and storage latency |

Pymoo provides population-level problem evaluation, allowing C01/C02 to remain
an internal implementation choice. Its starmap interface supports thread or
process execution for elementwise evaluation. Validate against the installed
version rather than upgrading it as an implicit part of this work.
[Population evaluation](https://pymoo.org/parallelization/vectorized.html),
[starmap interface](https://pymoo.org/parallelization/starmap.html).

### D. Compiled and accelerator implementations

These are alternative implementation routes, not a recommendation to maintain
every backend simultaneously. Select a route after profiling the array design.

| ID | Proposal | Expected opportunity | Effort / main risk |
| --- | --- | --- | --- |
| D01 | Optional Cython kernels for gathering and scoring | High if loops/allocations remain dominant | M–L; wheel builds and exact arithmetic; reuse ex-fuzzy packaging |
| D02 | Optional Numba/JIT prototype instead of Cython | Alternative way to test compiled loops | M; dependency/version support, compilation latency and fallback |
| D03 | C++/SIMD implementation instead of Cython | Potential for specialized kernels | L; extra maintenance/build complexity must beat simpler options |
| D04 | Fully batched Torch classification fitness on GPU | High only at sufficient workload size | L; exact decoding/pruning, float64 throughput and transfers |
| D05 | Fuse GPU operations / compile tensor graphs | Conditional after D04 | L; dynamic shapes, compile warm-up and numerical differences |
| D06 | Custom GPU kernels for gather/reduction/voting | Only if profiling D04/D05 justifies it | L; hardware dependence and substantial correctness burden |
| D07 | Alternative accelerator array backend, such as JAX/CuPy | Exploratory alternative to D04, not an additional default | L; ecosystem/build burden and whole-objective parity |
| D08 | Upgrade or replace optimizer/backend dependencies | Only if measured optimizer overhead or a required feature justifies it | M–L; changed RNG/operators/defaults may change search results |

For D01/C06, Cython offers GIL-free parallel loops using OpenMP. That is a
capability to test, not evidence of an ex-fuzzy speedup. Keep ordinary installs
compiler-free and provide tested fallback behavior.
[Cython parallelism documentation](https://cython.readthedocs.io/en/latest/src/userguide/parallelism.html).



## 2. Implementation sketches for the leading candidates

### A01–A03: firing strengths and immutable metadata

Start with T1, then T2. Build an ordered antecedent index representation and
gather from membership arrays rather than repeatedly constructing temporary
sample-by-feature arrays in Python. Preserve feature order and the reference
t-norm. Do not treat repeated selected features as multiple independent
antecedents: use the reference decoder's resolution first.

Test full ordered reductions against sequential in-place multiplication. The
latter saves memory but may change rounding; adopt it only if the required
parity is established, or keep it under a separately approved numerical mode.
Sparse zero-membership skipping can be investigated, but typical fuzzy supports
may be too dense for sparse containers to help.

Precompute gene slices, bounds, rule-slot indexes and label mappings once per
problem. Scratch space must be local to an evaluation or owned by one worker.
Benchmark both small and large rule/feature counts to detect extra gather-copy
costs. Initial files: `rules.py`, `_fitness.py`, `evolutionary_fit.py`.

### A04–A08: array decoder and numeric scoring

Represent a candidate with arrays for effective antecedents, consequents,
weights, validity and stable rule order. Match reference duplicate removal
before calculating fitness. Keep partition normalization in one shared,
tested implementation rather than maintaining two subtly different formulas.

Construct normal `MasterRuleBase` objects for final output and whenever existing
callbacks/custom losses need them. This preserves explanations, serialization
and the public API. Retain the full decoder as an oracle and fallback.

For optimized partitions, evaluate each candidate's own decoded membership
parameters. Avoid a population-global membership table. For custom fuzzy-set
classes, prefer fallback over replacing their membership semantics.

### B01–B07: cache only after measuring reuse

First measure raw-genotype, decoded-partition and rule repetition. Report hit
rate, hashing cost, peak bytes and time saved. Use a bounded fit-local cache;
discard it between fits. Account for datasets and labels without rehashing an
entire large dataset on every candidate lookup.

Cache hits must preserve population order, duplicate entries, logical evaluation
counts and stopping/callback behavior. Never reuse a parent's fitness merely
because only a few genes changed. Even one changed rule can alter every rule's
winning samples and which rules are pruned.

### C01–C03: bounded batches and exact streaming

Introduce a private population evaluator and adapt pymoo internally. Return
fitness in input order, including inactive or duplicate entries. Preserve
existing elementwise and custom-loss fallbacks and avoid double scheduling
when `runner` is already configured.

For population chunk size B, samples N, rules R, antecedents A, features D and
linguistic sets L, naive float64 intermediates can cost:

- Rule firing: roughly `8 * B * N * R` bytes.
- Gathered antecedents: roughly `8 * B * N * R * A` bytes.
- Memberships: roughly `8 * N * D * L` for fixed partitions, or
  `8 * B * N * D * L` for optimized partitions.

These are components, not a complete peak-memory bound; scores, masks, outputs,
temporary copies and worker duplication also count. Choose conservative chunks
and avoid constructing the full gathered tensor where possible.

Sample streaming requires global dominance statistics, then pre-pruning winners
and correctness counts, then post-pruning predictions/MCC. Without stored firing
strengths this generally means multiple passes, potentially three. It is not
correct to prune independently inside each sample batch. Validate reduction
order; exact streaming may need a reference-compatible reduction strategy.

### C04–C09 and D01–D07: concurrency and native code

Benchmark single-worker performance first. Compare threads, processes and
compiled parallel loops independently. Select one principal parallel layer to
avoid multiplying outer-fit workers, GA workers, OpenMP and numeric-library
threads. Reuse worker pools during a fit and close them on success or failure.

For GPU fitness, keep membership evaluation, dominance, pre-pruning winner
counts, pruning masks, final predictions and MCC on the device. Avoid Python
loops with per-scalar transfers, and bound device memory. Fixed partitions are
the simpler first target; optimized partitions need a chromosome dimension.
Measure cold compilation/startup, transfers and complete fit time separately.
Do not advertise GPU fitness based only on GPU crossover/mutation operations.

## 5. Proposed execution order and decision points

Each phase remains pending user selection. Completion of an earlier phase does
not authorize later phases or imply their benefits will compound.

1. **Measure and validate:** expand workload coverage, retain the reference
   oracle, and collect B01 duplication statistics.
2. **First CPU experiment:** A01, A02 and A03. Reprofile after each meaningful
   change. Stop adding complexity if the hot path has moved elsewhere.
3. **Remove Python construction costs:** A04, then A05/A08; consider A06/A07
   where the updated profile still supports them.
4. **Select a reuse strategy:** B02/B03 first if duplication is high; B04/B05
   only if their measured reuse offsets their memory and lookup costs.
5. **Select a batching/native strategy:** C01 before C02. Prototype D01 if
   array operations still spend substantial time in loops or allocations.
   D02/D03 are alternatives, not automatic additions.
6. **Select one concurrency route:** C04, C05 or C06 based on the new workload.
   Consider D04 only when CPU batching and dataset sizes justify GPU work.
7. **Separate research decisions:** E01–E13 and F01–F05 require individual
   selection and their own outcome criteria.

Recommended starting selection: **A01, A02, A03 and B01**, with measurement and
parity checks. A04 is the next substantial opportunity if profiling still shows
construction dominating. No additional speedup factor is promised in advance.

## 6. Validation and acceptance criteria

### Equivalence-preserving changes

- Compare against both the full reference and the current optimized evaluator.
- Check membership arrays, rule firing, dominance, retained rule identities and
  order, final predictions, penalties and fitness—not just training accuracy.
- Exercise T1/T2, fixed/optimized/unequal partitions, categorical features,
  supported modifiers/t-norms, weighting modes, unknown labels, ties, disabled
  and duplicate rules, and thresholds at/exactly adjacent to computed scores.
- Include zero-firing/empty cases, different input layouts, multiple dataset
  sizes, changed partitions and repeated fits on different same-shaped data.
- Check custom-loss fallback and invocation behavior, checkpoints, early
  stopping, worker ordering and exceptions.
- Compare seeded generation fitness arrays where practical, selected
  chromosomes and final fitted models. A tiny fitness difference can change
  evolution; tolerances require an explicit numerical-policy decision.
- Test supported dependency versions and optional-extension absence. Native
  code requires compiled CI coverage, not only skipped optional tests.

### Performance measurements

Use `benchmarks/benchmark_classifier_fitness.py` as the starting point. Expand
to Iris/Wine-sized data, synthetic 1K/10K samples, then larger data within the
machine's capacity; vary features, rules, antecedents, class balance, population
size and fuzzy type. Include real intended workloads when available.

Record candidate throughput, complete fit time, peak resident/device memory,
setup/compilation cost, cache hit rates and CPU/GPU utilization. Record package
versions, CPU/GPU model, thread settings, seeds and generation budgets. Run
without competing benchmarks/tests and report medians plus variability.

Keep search budgets fixed when measuring execution speed. Measure cold and warm
native/JIT runs separately. Gains must exceed run-to-run noise on the workloads
the option targets, with disclosed regressions on other workloads. Do not use
fragile wall-clock assertions in ordinary unit tests.

For search-changing E options, compare quality versus wall-clock time across
multiple seeds using held-out data, MCC/accuracy, rule count, interpretability
and relevant uncertainty metrics. Report altered budgets and any quality loss.

## 7. Review record

Fill or extend this table before implementation. “Investigate” means a bounded
prototype/measurement with a stated scope, not automatic production adoption.

| Selected IDs | Decision | Target workload / constraints | Acceptance requirement |
| --- | --- | --- | --- |
| — | Pending user review | — | — |

Current implementation and test entry points:

- `ex_fuzzy/ex_fuzzy/evolutionary_fit.py`
- `ex_fuzzy/ex_fuzzy/_fitness.py`
- `ex_fuzzy/ex_fuzzy/rules.py`
- `ex_fuzzy/ex_fuzzy/evolutionary_backends.py`
- `ex_fuzzy/ex_fuzzy/evolutionary_search.py`
- `tests/test_fast_fitness.py`
- `tests/test_genetic_fitness_semantics.py`
- `benchmarks/benchmark_classifier_fitness.py`

When we finish implementing everything, it would be nice to have a visual report wih the speedup that everything brought.
