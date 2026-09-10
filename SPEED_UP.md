# Genetic training speedup plan

**Status (2026-09-10): the earlier exact evaluator optimizations are complete.
The follow-up adds bounded population batching for small fixed-partition T1
fits and resolves enum serialization. An exact compiled-reduction prototype
remains benchmark-only. See the follow-up measurements below; further compiled,
T2 batching, optimized-partition batching and process-pool work are separate
increments, not shipped capabilities.**

### Current implementation status

The entries below describe bounded increments, not completion of their entire
proposal IDs. The current user authorization covers continued exact speedup
implementation guided by this plan. Search-changing behavior and new public
configuration options still require an explicit decision.

| ID | Current status | Implemented or authorized scope |
| --- | --- | --- |
| A01 | Implemented | Direct indexed lookup, plus one shared gathered kernel now used for T1 and T2 inside the array evaluator. |
| A02 | Implemented | Evaluation-local scratch reuse, plus fit-local packed membership tables that remove the per-candidate table build. |
| A03 | Implemented | Fit-local normalization bounds, label layout and packed membership table. |
| A04 | Implemented | Object-free decode/score path for built-in T1/T2 objectives; the object decoder stays the oracle and the fallback. |
| A05 | Implemented (packing route) | Memberships are evaluated straight into the gather table. Vectorising the trapezoid formula itself was measured and **not** adopted. |
| A06 | Implemented | Grouped, layout-preserving dominance reductions over contiguous consequent runs. |
| A07 | Implemented | Evaluation-local class masks and a fit-local integer label layout replacing the per-candidate `np.unique` in the MCC. |
| A08 | Implemented | Pruning masks and both complexity penalties computed on arrays. |
| A09 | Implemented | Empty-phenotype and fully-pruned candidates return the reference's `0.0` without scoring. |
| A10 | Implemented | Final fit computes the global classification metrics once, after pruning. |
| A11 | Partial: implemented | Exact `dict.setdefault` lookup; the pre-existing hash/equality inconsistency is preserved deliberately. |
| B01 | Done: diagnostic | Seeded reuse measurements and bounded offline cache simulation. |
| B02 | Implemented: scoped | Exact-genotype memoization for serial, built-in PyMoo fits only. |
| B04 | Implemented: scoped | Fit-local firing reuse, fixed partitions only, 8 MiB of retained columns. |
| B05 | Measured: rejected | Reusing firing across candidates with optimized partitions changed the objective of about half the candidates. |
| C01/C02 | C01 implemented in narrow scope; C02 deferred | Serial built-in T1, fixed partitions, at most 512 samples and a 32 MiB gather estimate; unsupported contexts retain scalar evaluation. |
| C05 | Serialization prerequisite fixed | Stable enum definition; fresh-process and spawned-worker parity tests. Persistent/shared-memory workers are not implemented. |
| C07 | Partial: implemented | Fit-scoped owned pools; success/error cleanup and external ownership preserved. |
| C10 | Measured | Peak RSS of the new fit-local caches reported below. |
| D01/D02/D03 | Exact prototype measured; not adopted | Explicit pairwise reductions pass tested parity; optional Numba dispatch stays in benchmarks. |
| Other IDs | Reviewed, pending or deferred | See [all 36 decisions and evidence](SPEED_UP_REVIEW.md). |

Scope: `BaseFuzzyRulesClassifier`, primarily its built-in classification objective
and `FitRuleBase`. This is a comprehensive catalogue of practical optimization
families, including speculative alternatives; it is not a claim that every
conceivable optimization has been enumerated. Record selected IDs and their
bounded scope before implementation; unselected recommendations remain proposals.

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
See the review record for shipped changes and [the catalogue review](SPEED_UP_REVIEW.md)
for every proposal's evidence and remaining work.

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
members; eliminating and replacing them changes the search.

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

The phases below are a sequencing guide for the currently authorized exact
speedup work. Completion of an earlier phase does not imply that benefits will
compound, and search-changing behavior or new options still require a decision.

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
7. **Separate research decisions:** search-changing alternatives require individual
   selection and their own outcome criteria. Earlier references to E/F proposal
   IDs were dangling references: this document defines only A01–D08 (36 items).

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
| A01 (direct lookup subset), A02 (local buffer reuse) | Execute: first increment implemented | T1/T2 built-in product/minimum reductions; preserve ordered float64 reduction, modifiers and custom t-norm behavior | Exact firing arrays and seeded search/model parity; compare against the previous optimized evaluator |
| A03 (normalization bounds) | Execute: second increment implemented | Compute empirical bounds once per problem; skip for fixed partitions | Exact decoded-model and seeded fitness parity; refit isolation |
| B01 | Execute: diagnostic implemented | Actual seeded populations, 256-entry simulated LRU; raw genotypes, decoded partitions and firing-equivalent rules | Instrumentation must preserve fitted result; report key cost and memory estimates |
| B02 | Execute: serial built-in PyMoo fits implemented | 256 entries, 1 MiB key-payload limit, one fit context; no custom losses, checkpoint mode, workers or custom backends | Preserve fitness sequence, final population, logical evaluation count and stopping; discard on return/exception |
| A01 (batched indexed gathering) | Execute for T2; defer T1 after experiment | Built-in product, unmodified rules, supplied memberships; bounded sample/rule blocks | Exact original firing arrays and seeded search/model results; complete-fit improvement |
| A07 (class masks) | Execute: bounded reuse implemented | Reuse masks across the two dominance passes of one candidate, up to 1 MiB retained payload | Exact original reductions, bounded retention and evaluation isolation; broader label encoding remains pending |
| A11 (dictionary lookup) | Execute: exact lookup optimization implemented | Use `dict.setdefault` while preserving existing `RuleSimple` equality/hash behavior and first retention/order | Match the original dictionary algorithm, including collision and hash-inconsistency cases |
| A03 (other metadata), B03–B07 | Pending | Reprofile before adding metadata or membership caches | Measure benefit and preserve search semantics |

### First implementation increment (2026-09-08)

`rules.py` now indexes list/tuple/array membership containers directly and reuses
one scratch array within each T1/T2 rule-base firing call for `np.prod` and
`np.min`. Every feature column is overwritten before reduction. Feature order,
don't-care slots, modifier arithmetic and reduction layout are preserved.
Custom t-norms keep independent per-rule allocations, and GT2 keeps its existing
allocation path. No persistent cache, new option or search change is introduced.

`tests/test_firing_buffers.py` compares exact results with an independent copy
of the original firing algorithm, including T1/T2, unequal partitions, modifiers,
disabled/duplicate rules, C/Fortran/strided arrays, changed rules and memberships,
and custom t-norm input ownership. The focused firing/fitness/semantics/rules
suite passes: **109 passed, 11 skipped**.

The benchmark now explicitly imports this checkout and reports its source path
and individual timing runs. Previously, executing the script directly could
benchmark a different installed release.

Complete-fit comparison against the optimized evaluator at `e67fc16`, replacing
only its firing method with the new method: Python 3.12.3, NumPy 2.3.5,
pymoo 0.6.1.6, Ryzen 5 5600X. Synthetic T1 data, 10 features, 3 classes,
20 rules, 4 antecedents, population 40, 5 generations, seed 7, no early stopping,
single evaluator worker and no competing test/benchmark run. Medians of three
runs; brackets show min–max seconds. These small gains need confirmation on
other machines; the optimized-partition 10K result is within timing noise.

| Samples | Partitions | Previous optimized fit (s) | New fit (s) | Speedup |
| --- | --- | --- | --- | --- |
| 150 | Optimized | 0.323 [0.323–0.327] | 0.317 [0.316–0.319] | 1.019× |
| 150 | Fixed | 0.245 [0.244–0.248] | 0.239 [0.238–0.242] | 1.024× |
| 1,000 | Optimized | 0.468 [0.465–0.470] | 0.458 [0.456–0.460] | 1.020× |
| 1,000 | Fixed | 0.392 [0.390–0.392] | 0.383 [0.383–0.384] | 1.022× |
| 10,000 | Optimized | 2.368 [2.368–2.397] | 2.373 [2.340–2.376] | 0.998× |
| 10,000 | Fixed | 2.367 [2.365–2.370] | 2.265 [2.257–2.331] | 1.045× |

All six comparisons checked exact equality of the evaluated fitness sequence,
selected chromosomes, final rule matrices/order/scores and predictions.
Peak RSS and hardware utilization were not measured in this increment.

### Second implementation increment (2026-09-08)

Profiling 100 optimized-partition T1 candidates at 10K samples after the first
increment put firing at about 48% and construction at about 13% of evaluation
time. Inspection also found an unnecessary full data scan in every decoder call,
including fixed-partition candidates which never use its results.

A03 now computes empirical normalization bounds once when creating an
optimized-partition `FitRuleBase`. Bounds are read-only and local to that
problem. Fixed-partition decoding skips this work entirely. The nan-aware
numeric bounds and categorical unique-count behavior remain the same, including
the existing distinction between empirical normalization and a supplied sampling
domain. Refit creates a new problem; optimized partitions get new bounds and
fixed partitions need none. The classifier's existing behavior of retaining
learned partitions on subsequent fits is preserved. As with existing
precomputed memberships, a problem's training data must stay stable during its
optimization. The helper can also initialize bounds for older serialized problems.

Comparison below retains the first increment's firing code in both variants and
swaps only the previous/current decoder method. Both variants include the new
one-time initialization scan, making the old-decoder timings slightly conservative.
Workload, versions and hardware match the first increment. Three paired runs
alternate previous/new implementations; every pair checks the complete fitness
sequence, selected chromosome, final rule matrices/order/scores and predictions.

| Samples | Type | Partitions | Previous decoder fit (s) | New fit (s) | Speedup |
| --- | --- | --- | --- | --- | --- |
| 150 | T1 | Optimized | 0.312 [0.312–0.314] | 0.292 [0.291–0.294] | 1.071× |
| 150 | T1 | Fixed | 0.239 [0.238–0.239] | 0.218 [0.217–0.218] | 1.097× |
| 1,000 | T1 | Optimized | 0.454 [0.454–0.456] | 0.425 [0.424–0.426] | 1.070× |
| 1,000 | T1 | Fixed | 0.384 [0.384–0.385] | 0.357 [0.356–0.358] | 1.075× |
| 10,000 | T1 | Optimized | 2.265 [2.196–2.336] | 2.253 [2.223–2.291] | 1.005× |
| 10,000 | T1 | Fixed | 2.286 [1.905–2.349] | 2.253 [2.251–2.266] | 1.015× |
| 1,000 | T2 | Optimized | 1.319 [1.318–1.319] | 1.290 [1.289–1.292] | 1.022× |
| 1,000 | T2 | Fixed | 1.260 [1.260–1.263] | 1.235 [1.227–1.238] | 1.021× |

The 10K runs have too much variability to establish a gain. No peak-memory or
utilization measurements were added. `tests/test_decoder_metadata.py` covers
empirical versus supplied domains, NaNs, categorical counts, constant columns,
absence of candidate-time scans, and repeated fits on different same-shaped data.
The combined metadata, firing, fast-fitness, genetic-semantics, training and
backend test suites pass: **147 passed**.

### Third implementation increment (2026-09-08): B01 and B02

`benchmarks/benchmark_candidate_reuse.py` now observes actual seeded searches
and simulates entry-bounded LRU caches offline. It measures exact-genotype,
decoded feature-partition and rule-firing repetition, key/lookup cost, extra
decoding time, retained key payload and hypothetical firing-cache value bytes.
It disables production memoization while measuring candidate evaluation costs
and checks the observed fit against an unobserved fit. Its generated T1/T2
workloads use built-in sets, product reductions and no modifiers; its decoded
keys are diagnostic representations, not production cache keys. Memory figures
exclude Python object overhead and are not peak RSS measurements.

For 1K samples, 10 features, 20 rules, population 40, 30 generations and seed 7,
the 256-entry simulation found:

| Partition mode | Exact genotype hits | Decoded feature-partition hits | Rule-firing hits |
| --- | --- | --- | --- |
| Optimized | 587 / 1,200 (48.9%) | 10,532 / 12,000 (87.8%) | 14,948 / 19,118 (78.2%) |
| Fixed | 777 / 1,200 (64.8%) | 11,990 / 12,000 (99.9%) | 17,490 / 19,594 (89.3%) |

Raw key payload peaked at about 598 KB / 372 KB respectively. A full 256-entry
T1 firing cache at this sample size would need 2,048,000 bytes for values alone.
Decoded diagnostic keys were much more expensive than raw keys, so this
increment selects exact-genotype fitness caching instead of partition/rule caches.
In a diagnostic rerun, raw key/lookup work cost 0.0049 s / 0.0042 s for the
1,200 candidates, versus 0.224 s / 1.732 s for decoded keys and an additional
0.642 s / 0.387 s for offline decoding. Estimated avoided raw-genotype
evaluation work was 1.092 s / 1.182 s before lookup overhead; actual fit gains
are measured separately below. Both observed fits matched the unobserved fits.

B02 is enabled only during a normal, serial, built-in PyMoo classifier fit.
It caches scalar fitness for exact one-dimensional numeric genotype bytes plus
dtype, with an LRU limit of 256 entries and a separate 1 MiB key-payload limit.
Oversized/non-numeric keys bypass caching. The cache belongs to the newly created
problem and is removed in a `finally` block before model finalization, including
when optimization fails. There is no persistent cache or constructor option.
Training data and objective configuration are fixed within this scope; no large
dataset hash is computed for each candidate. Custom losses, checkpoint mode,
workers, other/custom backends and direct standalone problem evaluation retain
their previous behavior. T1/T2 built-in objectives alone read the cache.

Population entries, fitness assignment, optimizer evaluation counts, RNG calls
and early stopping all remain in the original optimizer loop; only repeated
objective computation is skipped. `benchmark_classifier_fitness.py` now includes
an uncached optimized fit alongside the full reference and cached fit, and
reports `cache_fit_speedup` separately from the full-reference speedup.

Complete-fit timings with the previous two increments enabled in both variants,
same hardware/versions as above, population 40, seed 7 and early stopping disabled.
Medians of three paired uncached/cached runs; brackets show min–max seconds.

| Samples | Generations | Type | Partitions | Uncached fit (s) | Cached fit (s) | Cache speedup |
| --- | --- | --- | --- | --- | --- | --- |
| 150 | 5 | T1 | Optimized | 0.289 [0.289–0.291] | 0.202 [0.201–0.202] | 1.432× |
| 150 | 5 | T1 | Fixed | 0.216 [0.216–0.217] | 0.138 [0.137–0.138] | 1.568× |
| 1,000 | 30 | T1 | Optimized | 2.339 [2.336–2.348] | 1.255 [1.249–1.257] | 1.864× |
| 1,000 | 30 | T1 | Fixed | 1.937 [1.935–1.953] | 0.764 [0.764–0.764] | 2.534× |
| 10,000 | 30 | T1 | Optimized | 12.091 [11.744–12.432] | 7.006 [6.787–7.053] | 1.726× |
| 10,000 | 30 | T1 | Fixed | 12.273 [10.000–12.350] | 4.883 [4.883–4.909] | 2.514× |
| 1,000 | 30 | T2 | Optimized | 7.544 [7.513–7.547] | 4.143 [4.143–4.151] | 1.821× |
| 1,000 | 30 | T2 | Fixed | 6.904 [6.882–6.908] | 2.929 [2.920–2.934] | 2.357× |

All paired runs produced identical complete fitness sequences, selected
chromosomes, final rule scores and predictions. The longer runs reuse more
genotypes; these factors are workload-specific, not a promise for every search.
Larger key sets may also hit the byte limit before reaching 256 entries.
The combined cache, diagnostic, metadata, firing, fitness, training and backend
suite passes: **167 passed**. Cache tests cover T1/T2, all three weighting modes,
unknown predictions, fixed/optimized partitions, exact population and evaluation
count parity, early stopping, LRU/byte limits, exception cleanup and custom-loss,
checkpoint and worker fallbacks.

### Fourth implementation increment (2026-09-08): batched T2 firing (A01)

A profile of the cached 1K fixed-partition fit attributed about 30% of total fit
time to firing and 27% to dominance scoring. The gathering experiment packs
supplied membership values by feature/term, indexes all ordered antecedents in
small rule/sample blocks, and reduces over the original full feature axis.
Don't-cares remain explicit ones and completely disabled rules remain zero.
The float64 feature/interval layout matches the original T2 product reduction.
Table and gathered scratch target about 1 MiB (at least one sample), excluding
the output, index arrays and NumPy's internal temporaries. Scratch is call-local.

The new `rules._gather_rule_firing` path is enabled only for ordinary
`RuleBaseT2` bases with product t-norm, no modifiers, and supplied numeric
membership arrays of the expected shape. T1, GT2, custom bases/reductions,
modifiers, instance overrides, unsupported membership containers and empty
inputs keep the previous implementation. No persistent membership cache is
introduced, and prediction without supplied memberships keeps its prior path.

The T1 prototype showed inconsistent gains: the 1K optimized-partition fit
regressed from 0.291 s to 0.304 s (about 4%), while other T1 cases were close to
noise or improved only 2–3%. It was not enabled in production. T2 was retained
after the following paired complete-fit measurements, with all prior increments
enabled, 20 rules, 10 features, 4 antecedents, population 40, 5 generations,
seed 7 and early stopping disabled. Hardware and versions match prior sections.
Medians of three pairs; brackets show min–max seconds.

| Samples | T2 partitions | Previous firing fit (s) | Gathered firing fit (s) | Speedup |
| --- | --- | --- | --- | --- |
| 150 | Optimized | 0.370 [0.369–0.373] | 0.359 [0.359–0.360] | 1.030× |
| 150 | Fixed | 0.269 [0.268–0.270] | 0.258 [0.257–0.258] | 1.041× |
| 1,000 | Optimized | 0.778 [0.776–0.782] | 0.741 [0.738–0.742] | 1.049× |
| 1,000 | Fixed | 0.800 [0.800–0.801] | 0.754 [0.752–0.755] | 1.061× |
| 10,000 | Optimized | 6.150 [6.121–6.152] | 5.757 [5.743–5.767] | 1.068× |
| 10,000 | Fixed | 5.605 [5.094–5.624] | 5.275 [5.271–5.276] | 1.063× |

Paired runs checked complete fitness sequences, selected chromosomes, rule
scores and predictions. The 150/10K follow-up also checked final population
chromosomes and consequent order. The new kernel tests compare exact output
against the original per-rule path across 1/180/4097 samples, 1/7/33 features,
float32/float64 and strided memberships, uneven term counts, zero/disabled rules,
empty classes and rule/sample block boundaries. Fallback cases are tested too.
The broad relevant suite passes: **211 passed, 11 skipped**.

The existing benchmark now accepts `--fuzzy-type t2` and a benchmark-only
`--legacy-firing` switch to reproduce comparisons with the previous firing path:

```bash
python benchmarks/benchmark_classifier_fitness.py --fuzzy-type t2
python benchmarks/benchmark_classifier_fitness.py --fuzzy-type t2 --legacy-firing
```

### Fifth implementation increment (2026-09-09): A07 and A11

Work followed `Delegation_strategy/MULTIAGENT_ASTRA.md` and the configured
Terra/medium worker default: Terra handled scoring, Luna handled the status audit
and narrow duplicate-lookup change, and the root agent integrated, reviewed and
verified their changes. File ownership was separated and timing runs did not
overlap test execution.

A07 reuses `y == consequent` masks within one candidate's fitness evaluation,
including the second dominance pass after pruning. The cache retains at most
1 MiB of boolean-array payload; uncached masks are temporary. This is not a cap
on total RSS or dictionary overhead. Per-rule slicing, reduction order and T2
pooling remain unchanged. The cache ends with the evaluation, so it cannot reuse
labels from a different candidate evaluation or fit. Integer-label encoding and
MCC/count-layout reuse remain pending under A07.

A11 now uses `dict.setdefault(rule, index)` instead of a lookup followed by a
second lookup/insertion on a miss. Existing keys, hash/equality semantics,
first-rule retention and rule order are preserved. This removes duplicate hash
work for new entries without changing the pre-existing inconsistency between
`RuleSimple.__eq__` and its string-based hash. A semantic correction to that
inconsistency remains deferred because it could change the search result.

Separate and combined comparisons check exact fitness sequences, selected
chromosomes, final population chromosomes, rule scores, consequents, predictions
and logical evaluation counts. Initial single-process 10K timings showed large
order/allocation variability, so fresh-process measurements are used to assess
the new changes. Imports and process startup are excluded from fit timing.

Fresh-process T1 results (three repetitions per variant, randomized execution
order, each fit in a new process): 10 features, 3 classes, 20 rules, 4 antecedents,
population 40, 5 generations, seed 7, early stopping disabled. Python 3.12.3,
NumPy 2.3.5, pymoo 0.6.1.6 on the same Ryzen 5 5600X. All previous increments
stay enabled in both variants. Medians and min–max ranges are shown in seconds.

| Samples | Partitions | Before A07/A11 | Both changes | Speedup |
| --- | --- | --- | --- | --- |
| 1,000 | Optimized | 0.2925 [0.2923–0.2936] | 0.2839 [0.2829–0.2854] | 1.030× |
| 1,000 | Fixed | 0.2364 [0.2349–0.2373] | 0.2275 [0.2267–0.2301] | 1.040× |
| 10,000 | Optimized | 1.4860 [1.4839–1.4905] | 1.4782 [1.4726–1.4799] | 1.005× |
| 10,000 | Fixed | 1.3342 [1.3325–1.3459] | 1.3414 [1.3182–1.3445] | 0.995× |

The 1K improvement is primarily A11: isolated A11 medians were 0.2843 s /
0.2288 s for optimized/fixed partitions; isolated A07 medians were 0.2913 s /
0.2350 s. A07's isolated benefit is small and close to noise. The 10K results
are effectively neutral; no substantial large-data gain is claimed. Initial
single-process T2 runs were also nearly neutral (about 1% combined), and are
not used to claim a robust T2 speedup.

Benchmark-only switches reproduce the previous paths without changing public
classifier options:

```bash
python benchmarks/benchmark_classifier_fitness.py --samples 1000
python benchmarks/benchmark_classifier_fitness.py --samples 1000 --legacy-masks --legacy-duplicate-lookup
```

Use one switch at a time to isolate each optimization. New tests cover exact
T1/T2 dominance reductions on C/Fortran/strided arrays, absent classes and zero
firing, mask payload bounds and per-evaluation isolation. Duplicate-rule tests
compare object identity/order with the original dictionary algorithm, including
interleaved duplicates, modifier/score/weight differences and hash collisions.
Final combined verification: **230 passed, 11 skipped** across the dominance,
duplicate-rule, firing, cache, metadata, fitness, training and backend suites.

### Sixth implementation increment (2026-09-09): the array evaluator, A04–A10 and B04

This increment replaces the per-candidate rule-object pipeline for the built-in
T1/T2 classification objective. `FitRuleBase._array_score` decodes a chromosome
into integer antecedent/consequent/weight arrays and scores it without creating
any `RuleSimple`, `RuleBase` or `MasterRuleBase`. The object decoder plus
`_fitness.score_rulebase` remains both the oracle and the fallback: the array
path returns `None`, and the previous code runs, whenever a problem or candidate
leaves the supported case (custom losses, non-numeric labels, GT2, temporal
partitions, `time_moment`, `FitRuleBase` subclasses, out-of-range consequents,
membership containers the gather kernel cannot read). `_construct_ruleBase` is
still what builds the final fitted model, the checkpoint rule bases and anything
a custom loss or callback sees, so explanations, persistence and the public API
are unchanged.

Partition normalization was **not** duplicated. `_construct_ruleBase`'s fuzzy
variable decoding was extracted into `FitRuleBase._decode_antecedents`, and the
consequent gene offset into `FitRuleBase._consequent_pointer`; both decoders now
call the same helpers.

What ships:

- **A04/A08/A09** `_array_fitness.py`: array decoding with the reference's
  slot-overwrite order, per-class duplicate removal, array pruning masks, array
  complexity penalties and immediate `0.0` for empty or fully pruned candidates.
  Duplicate removal keeps `RuleSimple`'s observable dictionary behavior,
  including the hash/equality inconsistency that only appears in `ds_mode == 2`.
- **A01/A02** one gathered firing kernel, `rules._gather_firing_from_arrays`,
  now shared by T1 and T2 and by the object and array paths.
- **A05** `rules.pack_membership_table_from_variables` evaluates memberships
  straight into the gather table, removing both the intermediate per-feature
  arrays and the per-candidate table build. It calls the ordinary fuzzy-set
  `membership` methods, so categorical, gaussian and custom sets are covered.
- **A03** fixed partitions pack that table once per problem
  (`FitRuleBase._packed_memberships`), like the existing normalization bounds.
- **A06** `_fitness._dominance` reduces rules that share a consequent together.
  Every reduced axis is made contiguous first, so NumPy applies the same
  pairwise summation as the per-rule reference. `_dominance_reference` is kept
  as the parity oracle. The one reduction with no exact vectorised form is the
  T2 denominator, whose reference operand is a strided `(samples, 2)` view; it
  keeps its per-rule call.
- **A07** `_fitness._LabelDomain` rebuilds the MCC's sorted label set by
  counting instead of sorting `2 × samples` values on every candidate.
- **B04** `_fitness._FiringCache`, a fit-local LRU of firing columns keyed on a
  rule's effective antecedents, bounded to 8 MiB of retained columns. It is
  installed **only for fixed partitions**, in the same scope as the B02 fitness
  cache, and removed on optimizer return or exception.
- **A10** the final fit already computed the pre-pruning global metrics once;
  `tests/test_finalization_work.py` now covers it (its seeded-fit test had an
  undefined name and could not run).

#### Complete-fit results

Fresh processes, randomized execution order, medians. Ryzen 5 5600X,
Python 3.12.3, NumPy 2.3.5, pymoo 0.6.1.6. 10 features, 3 classes, 20 rules,
4 antecedents, population 40, 20 generations, seed 7, early stopping disabled.
`baseline` is the state before this increment, with all earlier increments
enabled. Every pair was checked for identical fitness, selected chromosome,
consequents, rule scores and predictions.

| Type | Samples | Partitions | Baseline (s) | Current (s) | Speedup |
| --- | ---: | --- | ---: | ---: | ---: |
| T1 | 150 | Optimized | 0.595 | 0.439 | 1.36× |
| T1 | 150 | Fixed | 0.320 | 0.168 | 1.91× |
| T1 | 1,000 | Optimized | 0.819 | 0.638 | 1.28× |
| T1 | 1,000 | Fixed | 0.471 | 0.255 | 1.85× |
| T1 | 10,000 | Optimized | 4.578 | 2.960 | 1.55× |
| T1 | 10,000 | Fixed | 3.235 | 1.432 | 2.26× |
| T2 | 150 | Optimized | 1.177 | 0.967 | 1.22× |
| T2 | 150 | Fixed | 0.622 | 0.343 | 1.81× |
| T2 | 1,000 | Optimized | 2.910 | 2.278 | 1.28× |
| T2 | 1,000 | Fixed | 1.807 | 0.951 | 1.90× |
| T2 | 10,000 | Optimized | 22.095 | 19.043 | 1.16× |
| T2 | 10,000 | Fixed | 14.039 | 7.472 | 1.88× |

Fixed partitions gain most because only they can reuse firing columns and a
packed membership table. Against the **full reference evaluator** at 1,000
samples and 5 generations the complete fit is now 16.0×/28.4× (T1
optimized/fixed) and 17.0×/24.5× (T2). Those reference multipliers are not the
useful comparison for future work; compare against the current evaluator.

Isolating the individual changes at 1,000 samples, 20 generations, T1:

| Variant | Optimized (s) | Fixed (s) |
| --- | ---: | ---: |
| Baseline | 0.819 | 0.474 |
| Grouped dominance only | 0.797 | 0.433 |
| Array evaluator only | 0.700 | 0.395 |
| Array + dominance, no firing cache | 0.679 | 0.351 |
| All of them | 0.678 | 0.281 |

#### Measured and rejected

- **A05, vectorised trapezoid formula.** Evaluating all terms of a variable in
  one broadcast expression was 0.85–0.87× (slower) for T1 and 1.09–1.12× for T2
  on the membership step alone, and did not beat writing the ordinary
  `membership` results straight into the packed table. It was removed rather
  than kept as an unused second implementation of the membership semantics.
- **B05, reusing firing across candidates with optimized partitions.** The same
  antecedent indexes denote different fuzzy sets on each candidate. With a
  256-entry firing cache, 104 of 200 candidates got a different objective. It is
  rejected under exact parity; `benchmarks/prototype_firing_cache.py` reproduces
  the divergence.
- **A06, naive axis vectorisation.** Reducing `firing` along axis 0 does not
  reproduce the per-column summation, and boolean-indexing `(rules, samples)`
  along axis 1 produces a Fortran-ordered array whose row sums differ. Both are
  covered by `tests/test_dominance_grouped.py`.

#### Memory (C10)

Peak RSS of a complete 10,000-sample, 10-generation fit, measured in fresh
processes with `benchmarks/prototype_remaining_routes.py --memory`. Both caches
are bounded to 8 MiB of payload each; the extra peak also covers dictionary and
array-object overhead and allocator behavior, and is not a promise for other
workloads.

| Type | Partitions | Caches on | Caches off | Extra peak |
| --- | --- | ---: | ---: | ---: |
| T1 | Fixed | 222.0 MB | 201.6 MB | +20.4 MB |
| T1 | Optimized | 210.7 MB | 211.1 MB | −0.4 MB |
| T2 | Fixed | 243.7 MB | 209.6 MB | +34.1 MB |
| T2 | Optimized | 229.3 MB | 229.1 MB | +0.2 MB |

Optimized partitions get neither cache, which is why they show no growth. The
budgets are module constants in `_fitness._FiringCache` and
`rules.pack_membership_table`; they are not public options.

#### Remaining headroom (2026-09-09 assessment)

The 2026-09-10 follow-up below supersedes the implementation status in this
historical assessment.

Uncached candidate evaluation time fits `a + b × samples` closely. The
sample-independent part `a` — interpreter dispatch and fixed-size decoding — is
what batching a whole population (C01/C02) could remove:

| Samples | T1 fixed share | T2 fixed share |
| ---: | ---: | ---: |
| 100 | 87% | 60% |
| 400 | 58% | 27% |
| 1,600 | 26% | 8% |
| 3,200 | 15% | 4% |

So population batching is worth a lot on small datasets and very little beyond a
few thousand samples. It also needs `elementwise=False`, padded rule masks and
per-candidate pruning, and it interacts with the thread runner and the custom
loss fallback, so it remains a separate decision rather than part of this work.

A compiled route (D01/D02/D03) hits a harder wall: a sequential Numba product
matches `np.prod` exactly in all 60 tested shapes, but a sequential sum differs
from `np.sum` in 42 of them, because NumPy uses pairwise summation on contiguous
reductions. A compiled kernel would have to reimplement pairwise summation
exactly, or the project would have to accept a numerical-policy change. Neither
is decided here.

#### Defects found (2026-09-09)

Both were pre-existing. The enum defect is fixed in the 2026-09-10 follow-up;
the direct `RuleBase` constructor defect remains outside this work:

- `rules.RuleBase.__init__` ends with `self.delete_duplicates()`, a method that
  does not exist anywhere in the package. It is unreachable in practice because
  `RuleBaseT1`, `RuleBaseT2` and `RuleBaseGT2` all override `__init__`, so only a
  direct `RuleBase(...)` instantiation would raise.
- `fuzzy_sets.FUZZY_SETS` defines `__eq__` without `__hash__`, so its members are
  unhashable, and `temporal` re-creates the enum so that `pickle` cannot resolve
  it by reference (`Can't pickle <enum 'FUZZY_SETS'>: attribute lookup
  FUZZY_SETS on temporal failed`). This is the concrete blocker behind C05:
  process workers cannot serialize a `FitRuleBase` until the enum has a single
  importable definition.

#### Reproduction

```bash
python benchmarks/benchmark_evaluator_variants.py --samples 1000 --generations 20
python benchmarks/benchmark_evaluator_variants.py --fuzzy-type t2 --samples 1000
python benchmarks/benchmark_classifier_fitness.py --samples 1000 --legacy-arrays
python benchmarks/prototype_firing_cache.py --fits --samples 1000
python benchmarks/prototype_remaining_routes.py --memory --overhead --numba
```

New tests: `tests/test_array_evaluation.py` (objective parity over ds_modes,
tolerances, penalties, unknown labels, degenerate and duplicate candidates,
categorical variables, phenotype identity and seeded fits),
`tests/test_dominance_grouped.py`, `tests/test_firing_cache.py`,
`tests/test_label_domain.py`, plus new kernel and packing cases in
`tests/test_gather_firing.py`. Full suite: **629 passed, 40 skipped**.

Current implementation and test entry points:

- `ex_fuzzy/ex_fuzzy/evolutionary_fit.py`
- `ex_fuzzy/ex_fuzzy/_fitness.py`
- `ex_fuzzy/ex_fuzzy/_array_fitness.py`
- `ex_fuzzy/ex_fuzzy/rules.py`
- `ex_fuzzy/ex_fuzzy/evolutionary_backends.py`
- `ex_fuzzy/ex_fuzzy/evolutionary_search.py`
- `tests/test_array_evaluation.py`
- `tests/test_dominance_grouped.py`
- `tests/test_firing_cache.py`
- `tests/test_gather_firing.py`
- `tests/test_label_domain.py`
- `tests/test_fast_fitness.py`
- `tests/test_genetic_fitness_semantics.py`
- `benchmarks/benchmark_classifier_fitness.py`
- `benchmarks/benchmark_evaluator_variants.py`
- `benchmarks/prototype_firing_cache.py`
- `benchmarks/prototype_remaining_routes.py`


## Exact-speedup follow-up — 2026-09-10

The selected follow-up is complete in its bounded scope: C01 small-data T1
batching is retained, C05's enum serialization prerequisite is fixed, and the
D01–D03 exact-reduction experiment remains benchmark-only. No numerical
tolerance, search change, public option or required dependency was introduced.

### C01: bounded population batching

`_population_fitness.py` batches fixed-partition T1 candidate decoding, firing,
dominance, pruning and MCC. It runs only for the built-in serial objective with
`ds_mode` 0/1, a supported integer-label layout, at most 512 samples, and a
32 MiB estimate for gathered values. This is a dispatch bound, not a promise
about total process memory. Unsupported cases retain the scalar/object paths;
custom losses, checkpoints and external runners retain their existing behavior.
Fitness/firing caches stay fit-local and bounded. All-cache-hit populations
return immediately. Duplicate population entries and logical evaluation counts
are preserved.

Fresh-process randomized-order complete fits, median of five repeats, Python
3.12.3, NumPy 2.3.5, pymoo 0.6.1.6 on the local x86-64 host. No concurrent test
or benchmark runs. Synthetic data seed 42, search seed 7, 10 features, 3 classes,
20 rules, 4 antecedents, population 40, 20 generations (800 logical evaluations),
early stopping disabled. Compare against the current optimized scalar evaluator,
including its caches. Thread environment was left unchanged.

| Samples | Scalar median (range), seconds | Batch median (range), seconds | Speedup | Peak RSS scalar/batch, MiB |
| ---: | --- | --- | ---: | ---: |
| 150 | 0.1700 (0.1689–0.1724) | 0.1077 (0.1061–0.1166) | 1.58× | 184.7/186.8 |
| 400 | 0.2044 (0.2036–0.2053) | 0.1465 (0.1461–0.1483) | 1.40× | 187.5/194.0 |
| 512 | 0.2199 (0.2180–0.2229) | 0.1817 (0.1809–0.1830) | 1.21× | 188.8/197.0 |
| 1,000 | 0.2543 (0.2534–0.2553) | 0.2548 (0.2533–0.2566) | 1.00× | 193.2/193.4 |

At 1,000 samples the batching dispatch declines, so that row measures fallback
overhead/noise. Peak RSS is the largest process peak among five repeats and
includes imports and allocator effects. Every repeat compared final population
chromosomes/fitnesses, selected chromosome, model performance, rule scores,
predictions and logical evaluation count exactly. The test suite also covers
penalties, unknown predictions, pruning, empty/duplicate phenotypes, cache hits,
label fallbacks and the sample/memory dispatch boundary. Additional randomized
review checked 180 contexts and padded/inactive rules at 512 samples.

### C05: serialization prerequisite

`fuzzy_sets.FUZZY_SETS` now contains the temporal members directly, with a hash
consistent with the existing value-based equality. `temporal.NEW_FUZZY_SETS`
is a compatibility alias; the temporal-only enum keeps its existing integer
values. Importing `temporal` no longer replaces the canonical enum.

Five serialization tests pass: fresh-process object/model/problem round trips
in both import modes, cross-import enum equality/hash, and real spawned-worker
candidate evaluation in both import modes. The spawn driver pins `PYTHONPATH`
to this checkout, avoiding an older installed package found during validation.
This fixes the serialization prerequisite; persistent/shared-memory worker
optimization and a process-pool speedup remain unimplemented.

### D01–D03: exact compiled prototype, not production adoption

`prototype_exact_compiled_reductions.py` uses optional Numba 0.63.1, explicitly
reproduces the tested pairwise sum tree and keeps layout/dtype fallbacks.
Initial validation: 56 direct sum cases, 72 T1/T2 firing/dominance cases and
216 fallback cases, with no mismatches. Compiling all tested signatures took
about 1.48 s in that probe. This supersedes the earlier conclusion that compiled
work must wait for a numerical-tolerance decision: explicit exact reductions
are feasible locally. It does not establish portable parity across every
supported NumPy version, CPU or array layout.

The evaluator benchmark now exposes opt-in `compiled-cold` and `compiled-warm`
variants. Both temporarily patch only the benchmark worker's array evaluator;
production never imports the prototype. Cold fit timing includes first-use JIT
compilation; warm timing follows compilation/parity checks. Package imports are
outside both fit timers. Fixed-partition firing caches remain enabled, so the
compiled firing adapter falls back to the cached production path there.

Same 1,000-sample/20-generation workload and environment as above, median of
three fresh-process randomized repeats. All repeats/variants had identical
final populations and fitnesses, selected chromosomes, consequents, rule scores,
predictions, model performance and logical evaluation counts.

| Type | Partitions | Current median (range), s | Compiled cold median (range), s | Compiled warm median (range), s | Warm speedup |
| --- | --- | --- | --- | --- | ---: |
| T1 | optimized | 0.649 (0.648–0.652) | 1.619 (1.614–1.634) | 0.559 (0.552–0.561) | 1.16× |
| T1 | fixed | 0.254 (0.253–0.256) | 1.243 (1.233–1.251) | 0.211 (0.211–0.212) | 1.20× |
| T2 | optimized | 2.301 (2.291–2.303) | 2.378 (2.364–2.389) | 1.231 (1.229–1.232) | 1.87× |
| T2 | fixed | 0.946 (0.942–0.947) | 1.887 (1.880–1.905) | 0.769 (0.768–0.771) | 1.23× |

**Decision:** retain the exact prototype and reproducible benchmarks, but do not
adopt it in production in this increment. Warm gains are real, especially T2
optimized partitions, but cold fits are slower in all four measured cases.
Production adoption still needs supported-version/platform parity, compiled CI,
optional-dependency fallback coverage and an approach to startup cost. No new
numerical tolerance is authorized or needed for the local results above.

### Reproduction and validation

```bash
pytest -q tests/
python benchmarks/benchmark_population_batching.py --samples 150 400 512 1000 --repeats 5
python benchmarks/prototype_exact_compiled_reductions.py
python benchmarks/benchmark_evaluator_variants.py --samples 1000 --generations 20 --variants current compiled-cold compiled-warm
python benchmarks/benchmark_evaluator_variants.py --fuzzy-type t2 --samples 1000 --generations 20 --variants current compiled-cold compiled-warm
```

The compiled variants require the optional local Numba installation and are
excluded from the benchmark's default variant list. The standalone prototype
reports availability and skips timing when Numba is absent; it does not install
anything. Timing is withheld if its parity checks fail.

Final validation of the complete working tree: **671 passed, 40 skipped**,
60 warnings, in 93.48 s (`pytest -q tests/`). The five enum serialization tests
include actual spawned-worker evaluation; 37 population tests cover exact
objective/search parity and fallback behavior. `git diff --check` passes when
respecting the existing CRLF files (`core.whitespace=cr-at-eol`).
