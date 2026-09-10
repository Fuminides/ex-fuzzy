"""Probe exact compiled CPU kernels for D01--D03.

This benchmark-only module asks a narrower question than a complete alternate
backend: can a bounded optional kernel reproduce the reduction tree used by
NumPy 2.x and pay for its call overhead on evaluator-shaped arrays?

NumPy's floating-point add reduction uses an eight-lane leaf up to 128 values,
then recursively splits at an eight-value boundary.  The Numba kernels below
spell out that tree and deliberately use neither ``fastmath`` nor parallel
reductions.  The public wrappers require C-contiguous float64 inputs and fall
back to the production NumPy implementation for every other layout/dtype.  The
fallback is part of the parity contract, not a benchmark convenience.

No package module imports this file.  Numba is optional and is never installed
by the probe.

    python benchmarks/prototype_exact_compiled_reductions.py
    python benchmarks/prototype_exact_compiled_reductions.py --timing
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
from pathlib import Path
import platform
from statistics import median
import sys
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ex_fuzzy import evolutionary_fit as evf, fuzzy_sets as fs, utils

_fitness = importlib.import_module(evf.__package__ + "._fitness")


PAIRWISE_BLOCK = 128


def _make_kernels():
    if importlib.util.find_spec("numba") is None:
        return None
    from numba import njit

    @njit(cache=False, fastmath=False)
    def pairwise_sum(values, start, count):
        """Float64 add loop matching NumPy's pairwise_sum template."""
        if count < 8:
            result = 0.0
            for index in range(count):
                result += values[start + index]
            return result
        if count <= PAIRWISE_BLOCK:
            r0 = values[start]
            r1 = values[start + 1]
            r2 = values[start + 2]
            r3 = values[start + 3]
            r4 = values[start + 4]
            r5 = values[start + 5]
            r6 = values[start + 6]
            r7 = values[start + 7]
            index = 8
            while index <= count - 8:
                r0 += values[start + index]
                r1 += values[start + index + 1]
                r2 += values[start + index + 2]
                r3 += values[start + index + 3]
                r4 += values[start + index + 4]
                r5 += values[start + index + 5]
                r6 += values[start + index + 6]
                r7 += values[start + index + 7]
                index += 8
            result = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
            while index < count:
                result += values[start + index]
                index += 1
            return result
        half = count // 2
        half -= half % 8
        return (pairwise_sum(values, start, half)
                + pairwise_sum(values, start + half, count - half))

    @njit(cache=False, fastmath=False)
    def firing_t1(table, indexes, disabled):
        samples = table.shape[0]
        rules, features = indexes.shape
        result = np.empty((samples, rules), dtype=np.float64)
        for sample in range(samples):
            for rule in range(rules):
                value = 1.0
                for feature in range(features):
                    value *= table[sample, indexes[rule, feature]]
                result[sample, rule] = 0.0 if disabled[rule] else value
        return result

    @njit(cache=False, fastmath=False)
    def firing_t2(table, indexes, disabled):
        samples = table.shape[0]
        rules, features = indexes.shape
        result = np.empty((samples, rules, 2), dtype=np.float64)
        for sample in range(samples):
            for rule in range(rules):
                lower = 1.0
                upper = 1.0
                for feature in range(features):
                    lower *= table[sample, indexes[rule, feature], 0]
                    upper *= table[sample, indexes[rule, feature], 1]
                if disabled[rule]:
                    lower = 0.0
                    upper = 0.0
                result[sample, rule, 0] = lower
                result[sample, rule, 1] = upper
        return result

    @njit(cache=False, fastmath=False)
    def dominance_t1(firing, y, consequents):
        samples, rules = firing.shape
        result = np.empty(rules, dtype=np.float64)
        scratch = np.empty(samples, dtype=np.float64)
        selected = np.empty(samples, dtype=np.float64)
        for rule in range(rules):
            consequent = consequents[rule]
            selected_count = 0
            for sample in range(samples):
                value = firing[sample, rule]
                scratch[sample] = value
                if y[sample] == consequent:
                    selected[selected_count] = value
                    selected_count += 1
            denominator = pairwise_sum(scratch, 0, samples)
            numerator = pairwise_sum(selected, 0, selected_count)
            # NumPy materializes values * mask before reducing support.
            for sample in range(samples):
                scratch[sample] = (firing[sample, rule]
                                   * (y[sample] == consequent))
            support = pairwise_sum(scratch, 0, samples) / samples
            confidence = numerator / denominator if denominator != 0.0 else 0.0
            result[rule] = support * confidence
        return result

    @njit(cache=False, fastmath=False)
    def dominance_t2(firing, y, consequents):
        samples, rules, _ = firing.shape
        result = np.empty(rules, dtype=np.float64)
        scratch = np.empty(samples * 2, dtype=np.float64)
        selected = np.empty(samples * 2, dtype=np.float64)
        for rule in range(rules):
            consequent = consequents[rule]
            selected_count = 0
            for sample in range(samples):
                lower = firing[sample, rule, 0]
                upper = firing[sample, rule, 1]
                scratch[2 * sample] = lower
                scratch[2 * sample + 1] = upper
                if y[sample] == consequent:
                    selected[selected_count] = lower
                    selected[selected_count + 1] = upper
                    selected_count += 2
            # ``firing[:, rule]`` is strided even when ``firing`` is C-order.
            # NumPy's iterator feeds that reduction in 8192-value buffers; each
            # buffer uses pairwise_sum and its subtotal is added left-to-right.
            denominator = 0.0
            offset = 0
            while offset < samples * 2:
                count = min(8192, samples * 2 - offset)
                denominator += pairwise_sum(scratch, offset, count)
                offset += count
            numerator = pairwise_sum(selected, 0, selected_count)
            for sample in range(samples):
                match = y[sample] == consequent
                scratch[sample] = firing[sample, rule, 0] * match
            lower_support = pairwise_sum(scratch, 0, samples) / samples
            for sample in range(samples):
                match = y[sample] == consequent
                scratch[sample] = firing[sample, rule, 1] * match
            upper_support = pairwise_sum(scratch, 0, samples) / samples
            support = (lower_support + upper_support) / 2.0
            confidence = numerator / denominator if denominator != 0.0 else 0.0
            result[rule] = support * confidence
        return result

    return pairwise_sum, firing_t1, firing_t2, dominance_t1, dominance_t2


KERNELS = _make_kernels()


def numpy_firing(table, indexes, disabled):
    # Match production's bounded packed-table route, including its 16-rule
    # gather chunks.  Chunking does not alter the feature-axis products.
    tail = (2,) if table.ndim == 3 else ()
    result = np.empty((table.shape[0], len(indexes)) + tail)
    for first in range(0, len(indexes), 16):
        last = first + 16
        gathered = np.take(table, indexes[first:last], axis=1)
        result[:, first:last] = np.prod(gathered, axis=2)
    result[:, disabled] = 0.0
    return result


def compiled_firing(table, indexes, disabled):
    """Use the bounded kernel only for its explicitly supported ABI."""
    supported = (KERNELS is not None and table.dtype == np.float64
                 and table.flags.c_contiguous and indexes.dtype == np.int64
                 and indexes.flags.c_contiguous and disabled.dtype == np.bool_
                 and disabled.flags.c_contiguous and table.ndim in (2, 3))
    if not supported:
        return numpy_firing(table, indexes, disabled)
    return KERNELS[1 if table.ndim == 2 else 2](table, indexes, disabled)


def compiled_dominance(firing, y, consequents, mask_cache=None):
    """Exact optional dispatch; unsupported operands retain NumPy behavior."""
    supported = (KERNELS is not None and firing.dtype == np.float64
                 and firing.flags.c_contiguous and firing.ndim in (2, 3)
                 and y.dtype == np.int64 and y.flags.c_contiguous
                 and consequents.dtype == np.int64
                 and consequents.flags.c_contiguous)
    if not supported:
        return _fitness._dominance(firing, y, consequents, mask_cache)
    return KERNELS[3 if firing.ndim == 2 else 4](firing, y, consequents)


def _compiled_array_firing(original):
    """Build an adapter with the production ``firing_strengths`` signature."""
    def adapter(antecedents, truth, n_samples, interval, cache=None, packed=None):
        if cache is not None or packed is None:
            return original(antecedents, truth, n_samples, interval, cache, packed)
        table, offsets, lengths = packed
        if (antecedents.shape[1] != len(lengths)
                or np.any(antecedents >= lengths)):
            return original(antecedents, truth, n_samples, interval, cache, packed)
        indexes = np.ascontiguousarray(np.where(
            antecedents >= 0, antecedents + offsets, table.shape[1] - 1),
            dtype=np.int64)
        disabled = np.ascontiguousarray(
            np.all(antecedents < 0, axis=1), dtype=np.bool_)
        return compiled_firing(table, indexes, disabled)
    return adapter


def _case(seed, samples, rules, features, interval):
    rng = np.random.default_rng(seed)
    terms = features * 3 + 1
    shape = (samples, terms, 2) if interval else (samples, terms)
    table = rng.random(shape)
    table[:, -1] = 1.0
    indexes = rng.integers(0, terms, size=(rules, features), dtype=np.int64)
    disabled = rng.random(rules) < 0.08
    y = np.ascontiguousarray(np.resize(np.array([0, 1, 2], dtype=np.int64), samples))
    consequents = np.ascontiguousarray(np.resize(
        np.array([0, 0, 1, 1, 2], dtype=np.int64), rules))
    return table, indexes, disabled, y, consequents


def parity_checks():
    if KERNELS is None:
        return {"available": False}
    report = {"available": True, "cases": 0, "sum_cases": 0,
              "firing_mismatches": 0, "dominance_mismatches": 0,
              "fallback_cases": 0, "fallback_mismatches": 0}
    rng = np.random.default_rng(20260909)

    compile_started = time.perf_counter()
    KERNELS[0](np.ones(8), 0, 8)
    for interval in (False, True):
        table, indexes, disabled, y, consequents = _case(
            2 + int(interval), 8, 4, 3, interval)
        firing = compiled_firing(table, indexes, disabled)
        compiled_dominance(firing, y, consequents)
    report["compile_seconds"] = round(time.perf_counter() - compile_started, 3)

    # Threshold boundaries and adversarial dynamic range exercise the exact
    # summation tree directly, including an empty reduction.
    lengths = list(range(0, 18)) + [31, 63, 127, 128, 129, 255, 256,
                                     257, 1000, 10000]
    for length in lengths:
        for dynamic in (False, True):
            values = rng.random(length)
            if dynamic and length:
                values[::3] *= 1e-12
                values[1::7] *= 1e12
            expected = np.sum(values)
            actual = KERNELS[0](values, 0, length)
            report["sum_cases"] += 1
            if np.asarray(expected).view(np.uint64) != np.asarray(actual).view(np.uint64):
                raise AssertionError(("pairwise sum", length, expected, actual))

    for interval in (False, True):
        for samples in (1, 7, 17, 127, 128, 129, 1000, 4097, 10000):
            for rules, features in ((1, 1), (7, 4), (20, 10), (33, 33)):
                table, indexes, disabled, y, consequents = _case(
                    10_000 + samples + rules + int(interval), samples, rules,
                    features, interval)
                expected_firing = numpy_firing(table, indexes, disabled)
                actual_firing = compiled_firing(table, indexes, disabled)
                expected_scores = _fitness._dominance(
                    expected_firing, y, consequents)
                actual_scores = compiled_dominance(
                    actual_firing, y, consequents)
                report["cases"] += 1
                if not np.array_equal(expected_firing, actual_firing, equal_nan=True):
                    report["firing_mismatches"] += 1
                if not np.array_equal(expected_scores, actual_scores, equal_nan=True):
                    report["dominance_mismatches"] += 1

                # The wrapper must preserve exact behavior on layouts outside
                # the narrow ABI by dispatching back to NumPy.
                variants = [np.asfortranarray(expected_firing)]
                expanded = np.repeat(expected_firing, 2, axis=0)
                variants.append(expanded[::2])
                variants.append(expected_firing.astype(np.float32))
                for variant in variants:
                    report["fallback_cases"] += 1
                    if not np.array_equal(
                            _fitness._dominance(variant, y, consequents),
                            compiled_dominance(variant, y, consequents),
                            equal_nan=True):
                        report["fallback_mismatches"] += 1
    return report


def _measure(function, minimum_seconds=0.15):
    iterations = 1
    while True:
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        elapsed = time.perf_counter() - started
        if elapsed >= minimum_seconds:
            break
        iterations *= 2
    runs = []
    for _ in range(5):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        runs.append((time.perf_counter() - started) / iterations)
    return median(runs), iterations


def timings():
    # Parity checks compile every signature before warm timings begin.
    for interval in (False, True):
        table, indexes, disabled, y, consequents = _case(
            1 + int(interval), 8, 4, 3, interval)
        firing = compiled_firing(table, indexes, disabled)
        compiled_dominance(firing, y, consequents)
    rows = []
    for interval in (False, True):
        for samples in (100, 1000, 10000):
            table, indexes, disabled, y, consequents = _case(
                100 + samples + int(interval), samples, 20, 10, interval)
            numpy_result = numpy_firing(table, indexes, disabled)
            compiled_result = compiled_firing(table, indexes, disabled)
            assert np.array_equal(numpy_result, compiled_result)
            assert np.array_equal(_fitness._dominance(numpy_result, y, consequents),
                                  compiled_dominance(compiled_result, y, consequents))
            functions = {
                "numpy_firing": lambda: numpy_firing(table, indexes, disabled),
                "compiled_firing": lambda: compiled_firing(table, indexes, disabled),
                "numpy_dominance": lambda: _fitness._dominance(
                    numpy_result, y, consequents),
                "compiled_dominance": lambda: compiled_dominance(
                    compiled_result, y, consequents),
                "numpy_combined": lambda: _fitness._dominance(
                    numpy_firing(table, indexes, disabled), y, consequents),
                "compiled_combined": lambda: compiled_dominance(
                    compiled_firing(table, indexes, disabled), y, consequents),
            }
            measured = {}
            iterations = {}
            for name, function in functions.items():
                measured[name], iterations[name] = _measure(function)
            rows.append({
                "kind": "T2" if interval else "T1",
                "samples": samples,
                "rules": 20,
                "features": 10,
                "seconds": {key: round(value, 9)
                            for key, value in measured.items()},
                "speedup": {
                    "firing": round(measured["numpy_firing"]
                                    / measured["compiled_firing"], 2),
                    "dominance": round(measured["numpy_dominance"]
                                       / measured["compiled_dominance"], 2),
                    "combined": round(measured["numpy_combined"]
                                      / measured["compiled_combined"], 2),
                },
                "iterations": iterations,
            })
    candidate_rows = _candidate_timings()
    return {"note": "warm medians; compile_seconds is reported under parity",
            "kernel_rows": rows, "candidate_rows": candidate_rows}


def _candidate_timings():
    """Time complete object-free candidate scores with temporary dispatch."""
    arrfit = importlib.import_module(evf.__package__ + "._array_fitness")
    original_firing = arrfit.firing_strengths
    original_dominance = arrfit._dominance
    compiled_array_firing = _compiled_array_firing(original_firing)
    rows = []
    for interval in (False, True):
        kind = fs.FUZZY_SETS.t2 if interval else fs.FUZZY_SETS.t1
        cases = ((True, 100), (True, 1000), (True, 10000),
                 (False, 1000))
        for fixed, samples in cases:
            rng = np.random.default_rng(30_000 + samples + int(interval))
            X = rng.normal(size=(samples, 10))
            y = np.ascontiguousarray(np.arange(samples, dtype=np.int64) % 3)
            partitions = utils.construct_partitions(X, kind) if fixed else None
            problem = evf.FitRuleBase(
                X, y, 20, 4, 3, linguistic_variables=partitions,
                fuzzy_type=kind)
            genes = rng.integers(
                problem.xl.astype(int), problem.xu.astype(int) + 1,
                size=(20, problem.n_var))
            problem._packed_memberships()  # exclude one-time packing from both paths

            try:
                arrfit.firing_strengths = original_firing
                arrfit._dominance = original_dominance
                expected = np.asarray([problem._array_score(gene) for gene in genes])
                numpy_seconds, numpy_iterations = _measure(
                    lambda: [problem._array_score(gene) for gene in genes])

                arrfit.firing_strengths = compiled_array_firing
                arrfit._dominance = compiled_dominance
                actual = np.asarray([problem._array_score(gene) for gene in genes])
                if not np.array_equal(expected, actual, equal_nan=True):
                    raise AssertionError(("complete candidate parity", interval,
                                          samples, expected, actual))
                compiled_seconds, compiled_iterations = _measure(
                    lambda: [problem._array_score(gene) for gene in genes])
            finally:
                arrfit.firing_strengths = original_firing
                arrfit._dominance = original_dominance
            rows.append({
                "kind": "T2" if interval else "T1",
                "partitions": "fixed" if fixed else "optimized",
                "samples": samples,
                "candidates": len(genes),
                "numpy_seconds_per_candidate": round(
                    numpy_seconds / len(genes), 9),
                "compiled_seconds_per_candidate": round(
                    compiled_seconds / len(genes), 9),
                "speedup": round(numpy_seconds / compiled_seconds, 2),
                "batch_iterations": {
                    "numpy": numpy_iterations,
                    "compiled": compiled_iterations,
                },
                "exact_fitness": True,
                "note": ("standalone uncached candidate; complete fixed-partition "
                         "fits may reuse firing columns"),
            })
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timing", action="store_true")
    args = parser.parse_args()
    numba_version = None
    if KERNELS is not None:
        import numba
        numba_version = numba.__version__
    result = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "numba": numba_version,
            "machine": platform.machine(),
        },
        "parity": parity_checks(),
    }
    parity = result["parity"]
    failed = any(parity.get(key, 0) for key in (
        "firing_mismatches", "dominance_mismatches", "fallback_mismatches"))
    if args.timing and parity["available"] and not failed:
        result["timing"] = timings()
    print(json.dumps(result, indent=2))
    parity = result["parity"]
    if (parity.get("firing_mismatches", 0)
            or parity.get("dominance_mismatches", 0)
            or parity.get("fallback_mismatches", 0)):
        raise SystemExit("exact parity failed")
