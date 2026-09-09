# Speedup catalogue review — 2026-09-09

All **36 defined proposals** in `SPEED_UP.md` now have a decision backed by code,
a measurement, or a stated blocker. "Inspect" means a code/dependency
assessment; "prototype" means experimental code outside the package; "measured"
means a local experiment. Deferred and rejected items are not implemented.
Native and accelerator backends are competing alternatives, not a requirement to
maintain every implementation simultaneously.

Every implemented item preserves the objective bit for bit. Nothing here changes
the search, a public option, or a default.

## Coverage and decisions

| ID | Evidence / decision | Remaining work or reason |
| --- | --- | --- |
| A01 | Measured; implemented | One gathered kernel (`rules._gather_firing_from_arrays`) now serves T1 and T2 and both the object and array paths. The earlier inconsistent T1 result came from the object path's surrounding overhead, which the array evaluator removes. |
| A02 | Measured; implemented | Evaluation-local scratch reuse, plus a packed membership table that removes the per-candidate table build. Further layout work needs a fresh profile. |
| A03 | Measured; implemented | Normalization bounds, the integer label layout and the packed membership table are all built once per problem and are read-only. |
| A04 | Measured; implemented | Object-free decoding and scoring for the built-in T1/T2 objective. The object decoder remains the oracle, the fallback and the builder of the final model, checkpoints and anything a custom loss sees. |
| A05 | Measured; implemented as packing only | Memberships are evaluated straight into the gather table by the ordinary fuzzy-set methods. Vectorising the trapezoid formula itself measured 0.85–0.87× (slower) for T1 and 1.09–1.12× for T2 and lost to packing end to end, so it was removed instead of being kept unused. |
| A06 | Measured; implemented | Consequent-run grouping with contiguous reduced axes is exact across 720 randomized T1/T2 cases in C, Fortran and strided layouts. The T2 denominator has no exact vectorised form and keeps its per-rule call. |
| A07 | Measured; implemented | Evaluation-local class masks plus `_LabelDomain`, which rebuilds the MCC's label set by counting instead of sorting `2 × samples` values per candidate. |
| A08 | Measured; implemented | Pruning masks and both complexity penalties run on arrays. The penalties are accumulated in the reference's order, one addition each: summing them first differed in the last bits. |
| A09 | Implemented | Empty phenotypes and fully pruned candidates return `0.0` without scoring, which is what the reference produces. A general zero-firing shortcut is still not established and is not implemented. |
| A10 | Implemented; tested | The final fit computes the global classification metrics once, after pruning. `tests/test_finalization_work.py` covers it; its seeded-fit test previously could not run because of an undefined name. |
| A11 | Measured; partial | `setdefault` removes duplicate hash work. The array decoder reproduces the same dictionary behavior, including the `ds_mode == 2` hash/equality inconsistency, rather than correcting it: correcting it would change which rules survive. |
| B01 | Measured; diagnostic complete | Seeded exact-genotype and decoded-fragment reuse diagnostic exists. |
| B02 | Measured; scoped implementation | Bounded exact cache for standard serial PyMoo only; custom/checkpoint/worker paths bypass it. |
| B03 | Superseded where it would apply | Under `elementwise=True` the fitness cache already skips every repeated genotype within and across generations, so within-generation sharing has no work left to save on the path where B02 runs. On the paths where B02 is deliberately disabled (workers, custom losses, checkpoints) sharing would need the ordered scatter of C01 first. |
| B04 | Measured; implemented for fixed partitions | Fit-local LRU of firing columns, 8 MiB of retained payload, installed in the same scope as B02 and only when partitions are fixed. Complete seeded fits: 1.11×–1.59× on top of the array evaluator. |
| B05 | Measured; rejected | With optimized partitions the same antecedent indexes denote different fuzzy sets per candidate. Reusing columns changed the objective of 104 of 200 candidates. `benchmarks/prototype_firing_cache.py` reproduces it. |
| B06 | Inspect; defer | Offspring provenance absent; one rule change can invalidate all winners/pruning. Needs a dependency design before incremental objective updates. |
| B07 | Prototype prerequisite; defer | The array decoder now exposes decoded arrays, but a phenotype key must still preserve weights, order and normalization and outweigh the decoding cost that B02's raw-genotype key avoids entirely. |
| C01 | Measured; not implemented | Uncached evaluation fits `a + b × samples`; the sample-independent share is 87%/58%/15% of a T1 fixed-partition candidate at 100/400/3,200 samples, and 60%/27%/4% for T2. Batching is therefore valuable only on small data, and needs `elementwise=False`, padded rule masks, per-candidate pruning and an ordered scatter that also serves the thread runner and custom-loss fallbacks. |
| C02 | Measured; follows C01 | Same headroom, plus a per-candidate membership dimension and a much larger memory budget. Optimized partitions also cannot share the packed table. |
| C03 | Numerical experiment; rejected as formulated | Chunked partial sums change rounding. Exact streaming needs an explicit accumulation design and global-pruning passes. |
| C04 | Measured | Complete serial/2-thread/4-thread diagnostic checks fit parity. Note that threads disable both fit-local caches, so the array evaluator widened the gap a worker pool has to make up. |
| C05 | Blocked; cause identified | Pickling a `FitRuleBase` fails with `Can't pickle <enum 'FUZZY_SETS'>: attribute lookup FUZZY_SETS on temporal failed`, because `temporal` re-creates the enum, and with `it's not the same object as fuzzy_sets.FUZZY_SETS` under the package's dual import modes. `FUZZY_SETS` members are also unhashable (`__eq__` without `__hash__`). Process workers need one importable enum definition first; that is a package-structure fix, not a speedup. |
| C06 | Inspect; defer | GCC/OpenMP available, but D01–D03 must clear the parity bar below before parallel scheduling over compiled kernels is meaningful. |
| C07 | Implemented; tested | Classifier-owned pools exist only during a fit and close/join on failure and success; external runners remain caller-owned. Global thread limits are not changed. |
| C08 | Measured | Diagnostic compares two independent serial-worker fits with two spawned processes; throughput, not single-fit acceleration. |
| C09 | Hardware inspected; defer | One local GPU, no multi-node test environment. No distributed scaling claim. |
| C10 | Measured | Peak RSS of complete 10,000-sample fits: +20.4 MB (T1 fixed) and +34.1 MB (T2 fixed) for the two 8 MiB fit-local caches; optimized partitions are unchanged because they get neither. Reported with `benchmarks/prototype_remaining_routes.py --memory`. |
| D01 | Measured; blocked on parity | Cython is installed, but see D02: the arithmetic bar is the same for any compiled route. |
| D02 | Measured; blocked on parity | A sequential Numba product matches `np.prod` in all 60 tested shapes; a sequential Numba sum differs from `np.sum` in 42 of them, because NumPy uses pairwise summation on contiguous reductions. A compiled kernel must reimplement pairwise summation exactly, or the project must accept a numerical-policy change. |
| D03 | Inspect; defer selection | Same parity bar as D01/D02, with more build and maintenance burden and no advantage established. |
| D04 | Kernel prototype/measured; investigate | RTX 5060 Ti available. Ordered CUDA products preserve tested parity; generic products do not. The full batched classification objective remains pending and inherits the C01 padding and pruning problems. |
| D05 | Prototype available; pending | Opt-in `--compile` probe exists; compilation not yet exercised. Depends on a stable tensor objective and warm-up accounting. |
| D06 | Inspect; defer | Custom kernels need evidence of bottlenecks in a validated D04/D05 first. |
| D07 | Dependencies checked; defer selection | JAX/JAXlib and CuPy unavailable locally. No installation or alternate-backend speed claim. |
| D08 | Installed path inspected; defer | No measured need to replace the optimizer; changed versions, operators or RNG would change the search. |

## Experiments

The scripts below are opt-in diagnostics, not imported by production code.
Kernel measurements do not include decoding, scoring or a complete fit and must
not be multiplied into the measured fit speedups.

### Evaluator variants

```bash
python benchmarks/benchmark_evaluator_variants.py --samples 1000 --generations 20
```

Runs every variant's fit in its own process in randomized order and refuses to
report timings unless all variants produced the identical fitness, chromosome,
consequents, rule scores and predictions. Complete-fit results are tabulated in
`SPEED_UP.md`: 1.16×–1.55× for optimized partitions and 1.81×–2.26× for fixed
partitions over the state before the array evaluator.

### Firing reuse

```bash
python benchmarks/prototype_firing_cache.py --fits --samples 1000
```

Reports hit rate, retained bytes, complete-fit speedup and whether the objective
stayed exact. Fixed partitions are exact and faster; optimized partitions are
reported as differing, which is the evidence against B05.

### Remaining routes

```bash
python benchmarks/prototype_remaining_routes.py --memory --overhead --numba
```

Measures the C10 memory cost, the C01/C02 batching headroom, and the D01–D03
compiled-arithmetic parity question in one run.

### CPU reductions

`python benchmarks/prototype_dominance_reductions.py --timing`

The earlier probe that rejected naive axis vectorization and chunked partial
sums. Its layout-preserving grouping is what shipped as A06, in the narrower
consequent-run form that avoids the fancy-index copies the probe still paid.

### Array decoder

`pytest -q tests/test_array_decoder_prototype.py`

The original object-oracle prototype. The production decoder in
`ex_fuzzy/ex_fuzzy/_array_fitness.py` supersedes it and is covered by
`tests/test_array_evaluation.py`, which compares complete objective values
against the object decoder rather than only the decoded phenotype.

## What is worth doing next

1. **Population batching (C01), for small datasets only.** The headroom is real
   below roughly 1,000 samples and negligible above a few thousand. It is the
   only remaining route with a large exact win, and it is a substantial change
   to the pymoo integration.
2. **Fix the `FUZZY_SETS` definition.** It unblocks C05, makes models and
   problems picklable, and is a correctness fix independent of performance.
3. **A numerical-policy decision.** Compiled and GPU routes (D01–D06) are all
   blocked on reproducing NumPy's pairwise summation. Deciding what tolerance,
   if any, is acceptable would reopen them; without it they stay closed.
