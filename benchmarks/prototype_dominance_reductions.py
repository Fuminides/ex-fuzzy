"""Numerical parity probe for A06 reductions and the C03 chunking boundary.

This is a benchmark-only experiment; it is not imported by production code.
Default execution performs correctness comparisons only.  ``--timing`` is
reserved for a later local measurement and is intentionally opt-in.
"""
import argparse
from collections import defaultdict
from pathlib import Path
import sys
import time
from statistics import median

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
INNER = ROOT / "ex_fuzzy" / "ex_fuzzy"
if str(INNER) not in sys.path:
    sys.path.insert(0, str(INNER))
import _fitness


def naive_axis_vectorized(firing, y, consequents):
    """Natural axis-vectorized formulation; expected to be parity-sensitive."""
    mask = y[:, None] == consequents[None, :]
    if firing.ndim == 2:
        support = np.mean(firing * mask, axis=0)
        denominator = np.sum(firing, axis=0)
        numerator = np.sum(firing * mask, axis=0)
    else:
        values = firing * mask[:, :, None]
        support = np.mean(np.mean(values, axis=0), axis=1)
        denominator = np.sum(firing, axis=(0, 2))
        numerator = np.sum(values, axis=(0, 2))
    confidence = np.divide(numerator, denominator, out=np.zeros_like(denominator),
                           where=denominator != 0)
    return support * confidence


def layout_preserving_grouped(firing, y, consequents):
    """Class-grouped data-layout proposal retaining each reference reduction.

    Rule-major contiguous copies make the selected sample rows explicit, but
    denominator and each support mean retain the reference's individual-rule
    call shape.  This is intentionally less aggressive than the naive version.
    """
    result = np.empty(len(consequents), dtype=float)
    groups = defaultdict(list)
    for index, consequent in enumerate(consequents):
        groups[consequent].append(index)
    for consequent, indexes in groups.items():
        mask = y == consequent
        # Copy in rule-major layout. Advanced indexing keeps increasing sample
        # order, matching ``values[mask]`` in _dominance.
        block = np.ascontiguousarray(firing[:, indexes].transpose(
            (1, 0) if firing.ndim == 2 else (1, 0, 2)))
        matched = np.ascontiguousarray(block[:, mask])
        for local, index in enumerate(indexes):
            original = firing[:, index]
            values = block[local]
            if firing.ndim == 2:
                support = np.mean(values * mask)
            else:
                # Preserve precisely mean([mean(lower), mean(upper)]), not an
                # all-axis mean, because the latter changes reduction layout.
                support = np.mean([np.mean(values[:, 0] * mask),
                                   np.mean(values[:, 1] * mask)])
            denominator = np.sum(original)
            numerator = np.sum(matched[local])
            confidence = numerator / denominator if denominator != 0 else 0.0
            result[index] = support * confidence
    return result


def chunked_sum(values, chunk):
    """The straightforward C03 partial-sum reduction, for equality checking."""
    total = 0.0
    for start in range(0, len(values), chunk):
        total += np.sum(values[start:start + chunk])
    return total


def cases():
    """C/F/strided inputs with non-contiguous and absent consequents."""
    rng = np.random.default_rng(20260909)
    for n in (1, 180, 1000, 10000):
        for r in (1, 7, 20):
            y = np.resize(np.array([10, 77, 77, 10]), n)
            consequents = np.resize(np.array([10, 77, 999]), r)
            for t2 in (False, True):
                shape = (n, r, 2) if t2 else (n, r)
                base = rng.random(shape)
                base.reshape(-1)[::17] = 0.0
                variants = {
                    "C": np.array(base, order="C", copy=True),
                    "F": np.array(base, order="F", copy=True),
                }
                # Stride only along samples, preserving the requested N.
                expanded = rng.random((n * 2,) + shape[1:])
                expanded.reshape(-1)[::17] = 0.0
                variants["strided"] = expanded[::2]
                for layout, firing in variants.items():
                    yield n, r, "T2" if t2 else "T1", layout, firing, y, consequents


def equal(a, b):
    return np.array_equal(a, b, equal_nan=True)


def run_checks():
    report = {"cases": 0, "naive_mismatches": 0, "layout_mismatches": 0,
              "chunk_mismatches": 0, "examples": []}
    for n, r, kind, layout, firing, y, consequents in cases():
        reference = _fitness._dominance(firing, y, consequents)
        naive = naive_axis_vectorized(firing, y, consequents)
        proposal = layout_preserving_grouped(firing, y, consequents)
        report["cases"] += 1
        if not equal(reference, naive):
            report["naive_mismatches"] += 1
            if len(report["examples"]) < 4:
                report["examples"].append(("naive", n, r, kind, layout,
                                           float(np.max(np.abs(reference - naive)))))
        if not equal(reference, proposal):
            report["layout_mismatches"] += 1
            if len(report["examples"]) < 4:
                report["examples"].append(("layout", n, r, kind, layout,
                                           float(np.max(np.abs(reference - proposal)))))
        # C03 demonstration: each smaller chunk has a distinct summation tree.
        # Check only first rule; all shapes/layouts are exercised by the loop.
        original = np.sum(firing[:, 0])
        for chunk in (1, 7, 64, n):
            if not equal(np.asarray(original), np.asarray(chunked_sum(firing[:, 0], chunk))):
                report["chunk_mismatches"] += 1
                if len(report["examples"]) < 4:
                    report["examples"].append(("chunk", n, r, kind, layout, chunk,
                                               float(abs(original - chunked_sum(firing[:, 0], chunk)))))
    return report


def optional_timing():
    """Compare reference/grouped reductions; local medians, not a full harness."""
    rng = np.random.default_rng(20260910)
    results = []
    for samples in (1000, 10000):
        for t2 in (False, True):
            firing = rng.random((samples, 20, 2) if t2 else (samples, 20))
            firing.reshape(-1)[::17] = 0.0
            y = np.resize(np.array([10, 77, 77, 10]), samples)
            consequents = np.resize(np.array([10, 77, 999]), 20)
            timings = {}
            for name, func in (("original", _fitness._dominance),
                               ("grouped", layout_preserving_grouped)):
                runs = []
                for _ in range(3):
                    started = time.perf_counter()
                    func(firing, y, consequents)
                    runs.append(time.perf_counter() - started)
                timings[name] = median(runs)
            results.append({"samples": samples, "rules": 20,
                            "kind": "T2" if t2 else "T1", **timings})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    result = run_checks()
    print(result)
    if result["layout_mismatches"]:
        raise SystemExit("layout-preserving proposal failed exact comparison")
    if args.timing:
        print({"optional_local_median_seconds": optional_timing()})
