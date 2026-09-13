"""Validate a completed T1 scaling campaign and update its README figure.

This never trains a model. It refuses partial, inconsistent or nonmatching
reports, and can run after a long benchmark without keeping an interactive
session open.
"""
import argparse
import json
import math
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET

from benchmark_speedup import LABELS, ROOT, plot, source_digest

START = '<!-- T1-SCALING:START -->'
END = '<!-- T1-SCALING:END -->'


def validate(report):
    """Require the complete requested grid and internally consistent evidence."""
    expected = {(n, d, f) for n in (1000, 10000, 100000)
                for d in (10, 50, 200) for f in (False, True)}
    rows = report['results']
    observed = {(r['samples'], r['settings']['features'], r['fixed']) for r in rows}
    if len(rows) != len(expected) or observed != expected:
        raise ValueError('Incomplete or unexpected scaling grid')
    if len(report['seeds']) != 3 or len(set(report['seeds'])) != 3:
        raise ValueError('Expected three distinct search seeds')
    for r in rows:
        c = r['settings']
        if r['fuzzy_type'] != 't1' or r['exact_parity'] is not True:
            raise ValueError('Missing T1 exact parity')
        if (c['samples'], c['fuzzy_type'], c['fixed']) != (r['samples'], 't1', r['fixed']):
            raise ValueError('Inconsistent workload settings')
        if (c['rules'], c['antecedents'], c['population'], c['generations']) != (20, 4, 40, 5):
            raise ValueError('Unexpected training budget')
        if set(r['seconds']) != set(LABELS) or set(r['median_seconds']) != set(LABELS):
            raise ValueError('Missing version timing')
        for label in LABELS:
            times = r['seconds'][label]
            if len(times) != 3 or not all(math.isfinite(t) and t > 0 for t in times):
                raise ValueError('Incomplete or invalid fit timings')
            if statistics.median(times) != r['median_seconds'][label]:
                raise ValueError('Incorrect timing median')
        if r['speedup'] != r['median_seconds'][LABELS[0]] / r['median_seconds'][LABELS[1]]:
            raise ValueError('Incorrect speedup ratio')
        if set(r['evidence']) != {str(seed) for seed in report['seeds']}:
            raise ValueError('Missing seed evidence')
        for evidence in r['evidence'].values():
            if evidence['n_eval'] != 200 or len(evidence['trace']) != 5:
                raise ValueError('Incomplete evaluation evidence')
            if any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                   for h in evidence['trace']):
                raise ValueError('Invalid objective trace digest')


def section(report, report_link='docs/performance/t1_scaling.json'):
    """Render the measured summary without manually copying timing values."""
    speedups = [r['speedup'] for r in report['results']]
    return f'''{START}
#### Larger T1 scaling benchmark

![T1 complete-fit scaling from 1,000 to 100,000 samples and 10 to 200 features]({str(Path(report_link).with_suffix('.svg'))})

This expanded T1-only campaign crosses **1,000 / 10,000 / 100,000 samples**
with **10 / 50 / 200 features**, for both fixed and optimized partitions.
All **54 reference/current pairs (108 complete fits)** passed exact parity,
covering **10,800 evaluated candidate positions** checked pairwise.
Measured median speedups range from **{min(speedups):.1f}× to {max(speedups):.1f}×**
on the recorded CPU; gains vary with both sample count and feature count.

The chart uses **logarithmic axes**. Points are median complete-fit seconds;
whiskers show the observed minimum and maximum across three matched seeds.
The rule and search budgets stay fixed at 20 rules, four antecedent slots,
population 40 and five generations. Each dataset has three classes and
`features - 1` informative features, so increasing width also changes the
learning problem. These timings do not measure predictive quality.

See the [raw scaling measurements and environment]({report_link})
and [full methodology and resume instructions](docs/performance/README.md#larger-t1-scaling-campaign).
The largest reference fits can each take over an hour. Reproduce on an otherwise
idle machine with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
  python benchmarks/benchmark_speedup.py --fuzzy-types t1 \\
  --samples 1000 10000 100000 --features 10 50 200 \\
  --output docs/performance/t1_scaling.json --resume
```
{END}

'''


def publish(report_path, readme=ROOT / 'README.md'):
    report = json.loads(report_path.read_text())
    validate(report)
    if report['source_sha256'] != source_digest():
        raise ValueError('Evaluator/worker source changed since the campaign started')
    report_link = report_path.resolve().relative_to(readme.resolve().parent).as_posix()
    figure = report_path.with_suffix('.svg')
    plot(report, figure)
    ET.parse(figure)
    with readme.open(newline='') as stream:
        original = stream.read()
    content = section(report, report_link)
    if START in original:
        if END not in original:
            raise ValueError('Unterminated scaling section in README')
        begin = original.index(START)
        end = original.index(END, begin) + len(END)
        updated = original[:begin] + content.rstrip('\n') + original[end:]
    else:
        anchor = '### Backend Comparison'
        if anchor not in original:
            raise ValueError('README performance insertion point not found')
        updated = original.replace(anchor, content + anchor, 1)
    with readme.open('w', newline='') as stream:
        stream.write(updated)
    print(f'Validated 18 workloads / 108 fits; updated {figure} and {readme}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path,
                        default=ROOT / 'docs/performance/t1_scaling.json')
    args = parser.parse_args()
    publish(args.report)
