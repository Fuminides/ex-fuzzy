# Ex-Fuzzy on the KEEL classification collection

The figure in the project README places Ex-Fuzzy's two rule learners next to two
scikit-learn baselines on the KEEL classification datasets, under one protocol:
same folds, same seed, same raw columns, library defaults except for the
Ex-Fuzzy search budget recorded below. It is an illustration of out-of-the-box
behaviour across many problems, **not** a tuned comparison and not a claim that
any method is best.

The KEEL collection is not redistributed with Ex-Fuzzy. Obtain it from
[the KEEL dataset repository](https://sci2s.ugr.es/keel/datasets.php) and point
`EX_FUZZY_KEEL_ROOT` at a directory holding one subdirectory per dataset, each
containing a `<name>.dat` file.

## Reproduce

From the repository root, with the package installed from source:

```bash
python -m pip install -e .
export EX_FUZZY_KEEL_ROOT=/path/to/keel_datasets

# One (dataset, method) pair.
python benchmarks/benchmark_keel.py --dataset iris --method exfuzzy-ga

# The whole grid on an Open Grid Engine cluster, one array task per pair.
bash benchmarks/cluster/submit_keel.sh

# Aggregate, then publish docs/performance/keel.{json,md,svg}.
python benchmarks/aggregate_keel.py
```

The published figure was built with `--allow-partial` because `census` is
excluded; see below.

`benchmarks/cluster/submit_keel.sh` freezes the task list, submits it to
`all.q`, and pins each task to one core. Rerunning it without `FORCE=1` skips
pairs that already have a result file, so it is also the way to fill gaps after
a partial run. `DATASETS`, `METHODS`, `MAX_CONCURRENT`, `FOLDS` and `SEED`
override the defaults. Without a cluster, loop over
`benchmarks/benchmark_keel.py --task-index N` instead.

To redraw the figure from the published aggregate:

```bash
python benchmarks/aggregate_keel.py --plot-only
```

## Measurement design

- **Datasets.** Every dataset in the local KEEL collection, read by
  `benchmarks/keel_datasets.py`. Nominal attributes become the integer codes of
  their declared order and are reported in a categorical mask; labels are mapped
  to `0..k-1`. Rows carrying KEEL's `?` missing marker are dropped and counted
  in the result file. No scaling, encoding or feature selection is applied, so
  every method sees identical columns.
- **Protocol.** 5-fold stratified cross-validation,
  `StratifiedKFold(shuffle=True, random_state=0)`, identical folds for every
  method. Each fold's estimator is seeded with `0 + fold_index`.
- **Reported numbers.** Test accuracy, balanced accuracy and macro F1 per fold,
  averaged over folds; then averaged (accuracy) or taken as the median (rules,
  time) across datasets. Ex-Fuzzy may abstain with `-1` when no rule fires; an
  abstention counts as an error and the count is recorded separately.
- **Model size.** "Rules" means Ex-Fuzzy GA rules surviving pruning, FERL leaves,
  decision-tree leaves, or leaves summed over every tree in the forest.
  "Conditions" is the total antecedent count over those rules. FERL's soft
  inference also consults interior nodes; the leaves are what `max_rules`
  bounds.
- **Timing.** Seconds inside `fit` for one fold, on shared cluster nodes with
  numerical library threads pinned to one. The published run spread over four
  Intel Xeon models (E5-2698 v4, Gold 6152, 6238 and 6238L), and other jobs
  share those nodes, so treat the training-time panel as an order-of-magnitude
  comparison, not a benchmark of the kind in
  [the speedup record](README.md).

## Method configuration

| Method | Configuration |
| --- | --- |
| Ex-Fuzzy GA rules | `BaseFuzzyRulesClassifier(fuzzy_type=t1, nRules=30, nAnts=4)`, fitted with `n_gen=100, pop_size=100, patience=25, min_delta=1e-4` |
| Ex-Fuzzy FERL | `FERL()` with library defaults |
| Decision tree | `DecisionTreeClassifier()` with library defaults |
| Random forest | `RandomForestClassifier(n_jobs=1)` with library defaults |

The genetic learner is the only method given a non-default setting, and only for
its **search budget**, which is uniform across every dataset rather than tuned
per problem. Ex-Fuzzy's shipped defaults (70 generations, population 30,
patience 10) stop early enough to understate it: on `vehicle` the defaults
reached 0.42 accuracy where the budget above reached 0.62, for a tenth of the
time. Quoting the truncated search would misrepresent the learner; quoting a
per-dataset tuned search would misrepresent the comparison. A fixed, larger
budget is stated instead. The structural budget — 30 rules, 4 antecedent slots,
Type-1 sets, three linguistic terms per variable — stays at the library default.

## What this does and does not show

- The rule budget is a **fixed 30 rules for every problem**. On the datasets with
  many classes (`abalone` has 28, `letter` 26, `kr-vs-k` 18) that is around one
  rule per class, and the genetic learner scores accordingly. This is a real
  property of a fixed small rule base, not an artefact to be corrected, but it
  means the accuracy panel should be read together with the size panel. The
  published run shows it directly. Mean accuracy for the genetic learner is
  0.791 on the 32 binary datasets, 0.714 on the 14 with three to five classes,
  and 0.523 on the 21 with six or more. Over the same bands the decision tree
  scores 0.811, 0.818 and 0.758. Scaling the rule budget with the number of
  classes is the obvious experiment, and it has not been run.
- The baselines run at their own defaults, which for a decision tree means
  unlimited depth and for a random forest means 100 unpruned trees. Neither is
  an interpretable model at that size; that contrast is the point of the size
  panel, not an oversight.
- Interval Type-2 and general Type-2 sets, the two-stage `FuzzyRulesClassifier`,
  rule mining, and the EvoX backend are all out of scope here. Type-2 costs
  roughly three to four times Type-1 per fit in spot checks and is not measured
  in this grid.
- Only datasets with a result for **every** method enter the comparison. Missing
  or failed pairs are listed at the bottom of [the results table](keel.md), and
  `benchmarks/aggregate_keel.py` refuses to publish an incomplete grid unless
  `--allow-partial` is passed.
- **`census` is excluded, so the figure covers 67 of the 68 datasets.** Its three
  baselines finished, but the genetic learner had not finished its first fold
  after about an hour. That puts a full run at an estimated 6–10 hours, so it
  was stopped. `census` has 142,521 rows and 41 features, and the next largest
  dataset, `adult`, needed about 12.5 minutes per fold. The baseline results are
  kept under `benchmarks/results/keel/`. Nothing else failed: all 268 other pairs
  completed.
- Single seed, single fold assignment. The interquartile bands in the figure
  span **datasets**, not repeated runs, so they describe how much the methods
  vary across problems, not the uncertainty of any one number.

## Tests

`tests/test_keel_benchmark.py` covers the parts that turn into published
numbers: KEEL header and data parsing, including the singular `@output` form,
tight type specifications, nominal level ordering and missing-marker handling;
the task grid; the size accounting; failure recording; rank averaging; and the
aggregation's refusal to compare a dataset that is missing a method.

```bash
python -m pytest -q tests/test_keel_benchmark.py
```

The benchmark itself is opt-in and is not run in CI: it needs the KEEL
collection, and its timings would be meaningless on shared CI machines.
