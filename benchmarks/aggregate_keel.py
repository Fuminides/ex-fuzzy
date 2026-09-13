"""Summarise the KEEL benchmark results into the README figure and its tables.

Reads every ``<dataset>__<method>.json`` written by ``benchmark_keel.py`` and
publishes three artefacts under ``docs/performance``: ``keel.json`` (the full
aggregate, including per-dataset rows), ``keel.md`` (the per-dataset table that
serves as the figure's text equivalent) and ``keel.svg`` (the figure).

Only datasets where every method produced a result are compared, so no method
gains an advantage from a missing entry; skipped pairs are listed explicitly.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_keel import (DEFAULT_OUTPUT, EXFUZZY_METHODS, LABELS, METHODS,  # noqa: E402
                            RULELESS_METHODS)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DESTINATION = ROOT / 'docs' / 'performance' / 'keel.json'

# Colour encodes one thing only: whether a model is Ex-Fuzzy's or a reference.
# Each dot sits on a row named by its method, so identity never rests on colour,
# and seven methods need no more than two hues. Values from the validated
# reference palette: categorical slot 1 and the muted ink.
EXFUZZY_COLOR = '#2a78d6'
REFERENCE_COLOR = '#898781'
INK = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
SURFACE = '#fcfcfb'
GRID = '#e1e0d9'
# Sequential blue ramp, steps 100-700, for the class-count heatmap.
SEQUENTIAL = ('#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
              '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b')


def method_color(method: str) -> str:
    return EXFUZZY_COLOR if method in EXFUZZY_METHODS else REFERENCE_COLOR

#: Accuracy is broken out by class count because the genetic learner's rule
#: budget is fixed, so the number of classes is what most changes the picture.
CLASS_BANDS = ((2, 2, 'Binary'), (3, 5, '3-5 classes'), (6, 10 ** 6, '6+ classes'))

PANELS = (
    ('mean_accuracy', 'Test accuracy', 'Mean over datasets', False, '{:.3f}'),
    ('mean_rules', 'Rules per model', 'Median over datasets', True, '{:,.0f}'),
    ('mean_fit_seconds', 'Training time', 'Median seconds per fold', True, '{:.3g}'),
)


def read_results(directory: Path) -> tuple[dict, list[dict]]:
    """Return ``{(dataset, method): record}`` for successes and the failures."""
    successes, failures = {}, []
    for path in sorted(directory.glob('*__*.json')):
        record = json.loads(path.read_text())
        if record.get('status') == 'ok':
            successes[record['dataset'], record['method']] = record
        else:
            failures.append(dict(dataset=record.get('dataset'), method=record.get('method'),
                                 error=record.get('error', 'unknown')))
    return successes, failures


def rank_within(values: dict[str, float], higher_is_better: bool = True) -> dict[str, float]:
    """Competition ranks with ties averaged, as used for average-rank tables."""
    ordered = sorted(values.items(), key=lambda item: -item[1] if higher_is_better else item[1])
    ranks, index = {}, 0
    while index < len(ordered):
        stop = index
        while stop + 1 < len(ordered) and ordered[stop + 1][1] == ordered[index][1]:
            stop += 1
        shared = (index + stop) / 2 + 1
        for position in range(index, stop + 1):
            ranks[ordered[position][0]] = shared
        index = stop + 1
    return ranks


def aggregate(successes: dict, failures: list[dict], methods: list[str]) -> dict:
    datasets = sorted({dataset for dataset, _ in successes})
    complete = [name for name in datasets
                if all((name, method) in successes for method in methods)]
    incomplete = [name for name in datasets if name not in complete]

    rows = []
    for name in complete:
        accuracies = {method: successes[name, method]['mean_accuracy'] for method in methods}
        ranks = rank_within(accuracies)
        row = dict(dataset=name, **{key: successes[name, methods[0]][key]
                                    for key in ('n_samples', 'n_features', 'n_classes')},
                   methods={})
        for method in methods:
            record = successes[name, method]
            row['methods'][method] = dict(
                accuracy=record['mean_accuracy'], accuracy_std=record['std_accuracy'],
                balanced_accuracy=record['mean_balanced_accuracy'],
                macro_f1=record['mean_macro_f1'], rules=record.get('mean_rules'),
                conditions=record.get('mean_conditions'),
                parameters=record.get('mean_parameters'),
                fit_seconds=record['mean_fit_seconds'],
                unclassified=record['mean_unclassified'], rank=ranks[method])
        rows.append(row)

    summary = {}
    for method in methods if rows else ():  # Nothing to summarise without a complete row.
        values = {key: [row['methods'][method][key] for row in rows]
                  for key in ('accuracy', 'balanced_accuracy', 'macro_f1', 'rules',
                              'conditions', 'fit_seconds', 'rank')}
        wins = sum(1 for row in rows if row['methods'][method]['rank'] == 1.0)
        has_rules = all(value is not None for value in values['rules'] + values['conditions'])
        if not has_rules:  # A linear model: keep the columns, leave them empty.
            values['rules'] = values['conditions'] = None
        summary[method] = dict(
            label=LABELS[method], datasets=len(rows), best_on=wins,
            mean_accuracy=statistics.fmean(values['accuracy']),
            median_accuracy=statistics.median(values['accuracy']),
            accuracy_q1=_quantile(values['accuracy'], .25),
            accuracy_q3=_quantile(values['accuracy'], .75),
            mean_balanced_accuracy=statistics.fmean(values['balanced_accuracy']),
            mean_macro_f1=statistics.fmean(values['macro_f1']),
            mean_rank=statistics.fmean(values['rank']),
            mean_rules=_maybe(statistics.fmean, values['rules']),
            median_rules=_maybe(statistics.median, values['rules']),
            rules_q1=_maybe(_quantile, values['rules'], .25),
            rules_q3=_maybe(_quantile, values['rules'], .75),
            median_conditions=_maybe(statistics.median, values['conditions']),
            mean_fit_seconds=statistics.fmean(values['fit_seconds']),
            median_fit_seconds=statistics.median(values['fit_seconds']),
            fit_seconds_q1=_quantile(values['fit_seconds'], .25),
            fit_seconds_q3=_quantile(values['fit_seconds'], .75))

    bands = []
    for low, high, label in CLASS_BANDS:
        members = [row for row in rows if low <= row['n_classes'] <= high]
        if not members:
            continue
        bands.append(dict(label=label, min_classes=low,
                          max_classes=max(row['n_classes'] for row in members),
                          datasets=len(members),
                          mean_accuracy={method: statistics.fmean(
                              row['methods'][method]['accuracy'] for row in members)
                              for method in methods}))

    environments = sorted({json.dumps(successes[key]['environment'], sort_keys=True)
                           for key in successes})
    return dict(created_utc=datetime.now(timezone.utc).isoformat(),
                methods=list(methods), method_labels={m: LABELS[m] for m in methods},
                configurations={m: successes[complete[0], m]['configuration']
                                for m in methods} if complete else {},
                folds=successes[next(iter(successes))]['folds'] if successes else None,
                seed=successes[next(iter(successes))]['seed'] if successes else None,
                n_datasets_compared=len(complete), datasets_compared=complete,
                datasets_incomplete=incomplete, failures=failures,
                environments=[json.loads(text) for text in environments],
                summary=summary, class_bands=bands, per_dataset=rows)


def _maybe(function, values, *args):
    """Apply a statistic, or return ``None`` when the quantity is undefined."""
    return None if values is None else function(values, *args)


def _fmt(value, spec: str, missing: str = '—') -> str:
    return missing if value is None else format(value, spec)


def _quantile(values: list[float], fraction: float) -> float:
    """Linear-interpolated quantile; ``statistics.quantiles`` needs n >= 2."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def write_table(report: dict, destination: Path) -> None:
    """The per-dataset table: the figure's text equivalent, not a summary."""
    methods = report['methods']
    lines = ['# KEEL benchmark results',
             '',
             f'Generated {report["created_utc"]} from '
             f'`benchmarks/results/keel/`. {report["n_datasets_compared"]} datasets have a '
             f'result for every method and are the ones the figure compares. '
             f'See [the methodology](KEEL.md) for the protocol.',
             '',
             '## Summary',
             '',
             '| Method | Mean accuracy | Mean rank | Best on | Median rules | Median conditions | Median fit (s) |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for method in methods:
        entry = report['summary'][method]
        lines.append(f'| {entry["label"]} | {entry["mean_accuracy"]:.4f} | '
                     f'{entry["mean_rank"]:.2f} | {entry["best_on"]}/{entry["datasets"]} | '
                     f'{_fmt(entry["median_rules"], ",.1f")} | '
                     f'{_fmt(entry["median_conditions"], ",.1f")} | '
                     f'{entry["median_fit_seconds"]:,.2f} |')
    lines += ['', '## Accuracy per dataset',
              '', 'Mean test accuracy over the cross-validation folds. '
                  'The best value in each row is bold.', '',
              '| Dataset | Rows | Features | Classes | ' +
              ' | '.join(LABELS[method] for method in methods) + ' |',
              '| --- | ---: | ---: | ---: |' + ' ---: |' * len(methods)]
    for row in report['per_dataset']:
        best = max(row['methods'][method]['accuracy'] for method in methods)
        cells = []
        for method in methods:
            value = row['methods'][method]['accuracy']
            cells.append(f'**{value:.4f}**' if value == best else f'{value:.4f}')
        lines.append(f'| {row["dataset"]} | {row["n_samples"]:,} | {row["n_features"]} | '
                     f'{row["n_classes"]} | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Rules per model',
              '', 'Mean rule count over the folds: Ex-Fuzzy rules, FERL and decision-tree '
                  'leaves, and leaves summed over the forest. Logistic regression is not a '
                  'rule model and has no entry.', '',
              '| Dataset | ' + ' | '.join(LABELS[method] for method in methods) + ' |',
              '| --- |' + ' ---: |' * len(methods)]
    for row in report['per_dataset']:
        cells = [_fmt(row['methods'][method]['rules'], ',.1f') for method in methods]
        lines.append(f'| {row["dataset"]} | ' + ' | '.join(cells) + ' |')
    if report['datasets_incomplete'] or report['failures']:
        lines += ['', '## Not compared', '']
        for name in report['datasets_incomplete']:
            lines.append(f'- `{name}`: missing at least one method.')
        for failure in report['failures']:
            lines.append(f'- `{failure["dataset"]}` / `{failure["method"]}`: {failure["error"]}')
    destination.write_text('\n'.join(lines) + '\n')


def plot(report: dict, destination: Path) -> None:
    """Draw the README figure: three dot panels and a class-count heatmap.

    Dots rather than bars because two panels are logarithmic, where a bar's
    length is not proportional to what it represents. Rows are methods in the
    same order in every panel; colour only separates Ex-Fuzzy from reference
    models. The heatmap uses one sequential hue with the value printed in each
    cell, so it needs no categorical colours at all.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap
    from matplotlib.lines import Line2D
    import numpy as np

    methods = report['methods']
    labels = [LABELS[method] for method in methods]
    summary = report['summary']
    bands = report.get('class_bands') or []
    positions = np.arange(len(methods))[::-1]  # First method at the top.
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none',
                         'font.family': ['DejaVu Sans', 'sans-serif'],
                         'text.color': INK, 'axes.labelcolor': INK_SECONDARY,
                         'xtick.color': INK_SECONDARY, 'ytick.color': INK_SECONDARY})
    figure = plt.figure(figsize=(13.2, 7.4), facecolor=SURFACE, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, width_ratios=(1, 1))
    layout = {'mean_accuracy': grid[0, 0], 'mean_rules': grid[0, 1],
              'mean_fit_seconds': grid[1, 0]}
    labelled = {'mean_accuracy', 'mean_fit_seconds'}  # Left column carries the names.

    for key, title, subtitle, logarithmic, fmt in PANELS:
        axis = figure.add_subplot(layout[key])
        _style(axis)
        statistic = 'mean' if key == 'mean_accuracy' else 'median'
        base = key.removeprefix('mean_')
        values = [summary[method][f'{statistic}_{base}'] for method in methods]
        low = [summary[method][f'{base}_q1'] for method in methods]
        high = [summary[method][f'{base}_q3'] for method in methods]
        present = [v for v in values + low + high if v is not None]
        if logarithmic:
            axis.set_xscale('log')
            axis.set_xlim(min(present) / 4, max(present) * 14)
        else:
            axis.set_xlim(0, 1.12)
        floor = axis.get_xlim()[0]
        for position, method, value, left, right in zip(positions, methods, values, low, high):
            if value is None:
                axis.annotate('not a rule model', (floor, position), xytext=(4, 0),
                              textcoords='offset points', va='center', fontsize=9,
                              color=INK_MUTED, style='italic')
                continue
            color = method_color(method)
            axis.plot([floor, value], [position, position], color=GRID,
                      linewidth=1, solid_capstyle='butt', zorder=2)
            axis.plot([left, right], [position, position], color=color,
                      linewidth=4.5, alpha=.30, solid_capstyle='round', zorder=3)
            axis.plot([value], [position], marker='o', markersize=9, color=color,
                      markeredgecolor=SURFACE, markeredgewidth=2, zorder=4)
            # Anchored past the range band so the label never sits on it.
            axis.annotate(fmt.format(value), (max(value, right), position), xytext=(12, 0),
                          textcoords='offset points', va='center', fontsize=9.5,
                          color=INK, fontweight='bold', zorder=5)
        axis.set_yticks(positions, labels if key in labelled else [''] * len(labels))
        axis.set_ylim(-0.6, len(methods) - 0.4)
        _titles(axis, title, subtitle)
        axis.set_xlabel({'mean_accuracy': 'Accuracy', 'mean_rules': 'Rules (log scale)',
                         'mean_fit_seconds': 'Seconds (log scale)'}[key])

    if bands:
        axis = figure.add_subplot(grid[1, 1])
        axis.set_facecolor(SURFACE)
        matrix = np.array([[band['mean_accuracy'][method] for band in bands]
                           for method in methods])
        low, high = np.floor(matrix.min() * 20) / 20, np.ceil(matrix.max() * 20) / 20
        colormap = LinearSegmentedColormap.from_list('keel_blue', SEQUENTIAL)
        edges = np.linspace(low, high, len(SEQUENTIAL) + 1)
        axis.pcolormesh(np.arange(len(bands) + 1), np.arange(len(methods) + 1) - 0.5,
                        matrix[::-1], cmap=colormap,
                        norm=BoundaryNorm(edges, colormap.N), edgecolors=SURFACE,
                        linewidth=2)
        midpoint = low + (high - low) * 0.55
        for row, method in enumerate(methods):
            for column, value in enumerate(matrix[row]):
                axis.text(column + .5, positions[row], f'{value:.3f}', ha='center',
                          va='center', fontsize=9.5,
                          color=SURFACE if value >= midpoint else INK,
                          fontweight='bold' if method in EXFUZZY_METHODS else 'normal')
        axis.set_xticks(np.arange(len(bands)) + .5,
                        [f'{band["label"]}\n{band["datasets"]} datasets' for band in bands])
        axis.set_yticks(positions, [''] * len(methods))
        axis.set_xlim(0, len(bands))
        axis.set_ylim(-0.6, len(methods) - 0.4)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        # Group labels sit below, like the neighbouring panel's axis, so the
        # heatmap rows stay level with that panel's method rows.
        _titles(axis, 'Accuracy by number of classes', 'Mean over the datasets in each group')

    figure.legend(handles=[Line2D([], [], marker='o', linestyle='none', markersize=9,
                                  color=color, markeredgecolor=SURFACE, markeredgewidth=2,
                                  label=label)
                           for color, label in ((EXFUZZY_COLOR, 'Ex-Fuzzy'),
                                                (REFERENCE_COLOR, 'Reference model'))],
                  loc='upper right', bbox_to_anchor=(0.995, 0.995), ncols=2, frameon=False,
                  fontsize=10, labelcolor=INK_SECONDARY, handletextpad=.3)
    figure.suptitle(
        f'Ex-Fuzzy on {report["n_datasets_compared"]} KEEL classification datasets',
        x=0.006, ha='left', fontsize=15, fontweight='bold', color=INK)
    figure.supxlabel(
        f'{report["folds"]}-fold stratified cross-validation with one shared seed. FERL uses the '
        'compact, medium and deep presets of fuzzy_greedy_tree; the GA uses a stated search '
        'budget; baselines use library defaults.\nDots are the mean (accuracy) or median '
        '(rules, time) across datasets. The band through each dot spans the interquartile '
        'range across datasets, not a confidence interval.',
        x=0.006, ha='left', fontsize=8.8, color=INK_SECONDARY)
    figure.savefig(destination, dpi=160, facecolor=SURFACE)
    plt.close(figure)


def _style(axis) -> None:
    """Recessive chrome: hairline solid grid, no box, no y ticks."""
    axis.set_facecolor(SURFACE)
    axis.xaxis.grid(True, color=GRID, linewidth=.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines[['top', 'right', 'left']].set_visible(False)
    axis.spines['bottom'].set_color(GRID)
    axis.tick_params(axis='y', length=0)
    axis.tick_params(axis='x', length=3, color=GRID)


def _titles(axis, title: str, subtitle: str, pad: float = 15) -> None:
    axis.set_title(title, loc='left', fontsize=12, fontweight='bold', color=INK, pad=pad)
    axis.annotate(subtitle, (0, 1), xytext=(0, 6), xycoords='axes fraction',
                  textcoords='offset points', fontsize=9.5, color=INK_SECONDARY)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--output', type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument('--methods', nargs='*', default=list(METHODS))
    parser.add_argument('--plot-only', action='store_true',
                        help='Redraw the figure from an existing aggregate.')
    parser.add_argument('--allow-partial', action='store_true',
                        help='Publish even when some pairs are still missing.')
    options = parser.parse_args(argv)

    if options.plot_only:
        plot(json.loads(options.output.read_text()), options.output.with_suffix('.svg'))
        return 0

    if not options.results.is_dir():
        parser.error(f'No results directory at {options.results}')
    successes, failures = read_results(options.results)
    if not successes:
        parser.error(f'No successful results in {options.results}')
    report = aggregate(successes, failures, options.methods)
    expected = len({dataset for dataset, _ in successes}) * len(options.methods)
    if report['datasets_incomplete'] and not options.allow_partial:
        parser.error(f'{len(report["datasets_incomplete"])} dataset(s) are missing a method '
                     f'({len(successes)}/{expected} pairs present): '
                     f'{", ".join(report["datasets_incomplete"])}. '
                     f'Finish the run or pass --allow-partial.')
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + '\n')
    write_table(report, options.output.with_suffix('.md'))
    plot(report, options.output.with_suffix('.svg'))
    print(f'{report["n_datasets_compared"]} datasets compared across '
          f'{len(options.methods)} methods')
    for method in options.methods:
        entry = report['summary'][method]
        print(f'  {entry["label"]:22s} acc={entry["mean_accuracy"]:.4f} '
              f'rank={entry["mean_rank"]:.2f} rules={_fmt(entry["median_rules"], ">10,.1f"):>10} '
              f'fit={entry["median_fit_seconds"]:>8,.2f}s')
    print(f'wrote {options.output}, {options.output.with_suffix(".md")}, '
          f'{options.output.with_suffix(".svg")}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
