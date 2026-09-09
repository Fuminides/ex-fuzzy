"""Benchmark-only firing-reduction backend prototypes.

This module deliberately does not integrate with ex-fuzzy production code.  It
compares an ordered feature-axis NumPy oracle with optional Numba and Torch
implementations for the reduction shape used by T1/T2-like firing arrays.

The default command performs numerical parity checks only.  Timing is opt-in
with ``--timing`` so that a parity run cannot accidentally become a workload
benchmark.  Optional dependencies are detected at runtime and are never
installed by this script.
"""

from __future__ import annotations

import argparse
import importlib.util
import time
from typing import Callable, Iterable, Optional

import numpy as np


SHAPES = (1, 7, 10, 33)


def numpy_product(values: np.ndarray) -> np.ndarray:
    """Reference reduction: preserve the complete ordered feature axis."""
    return np.prod(values, axis=2)


def _make_numba_product() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create the rank-3 T1 Numba kernel, or return None if unavailable."""
    if importlib.util.find_spec("numba") is None:
        return None
    try:
        from numba import njit

        @njit(cache=False, fastmath=False)
        def product(values):
            n_samples, n_rules, n_features = values.shape
            result = np.ones((n_samples, n_rules), dtype=np.float64)
            for sample in range(n_samples):
                for rule in range(n_rules):
                    value = 1.0
                    for feature in range(n_features):
                        value *= values[sample, rule, feature]
                    result[sample, rule] = value
            return result

        return product
    except Exception:
        # Import/runtime incompatibilities are reported by the caller.
        return None


def _make_numba_product_t2() -> Optional[Callable[[np.ndarray], np.ndarray]]:
    """Create a separate rank-4 Numba kernel for interval/T2-like values."""
    if importlib.util.find_spec("numba") is None:
        return None
    try:
        from numba import njit

        @njit(cache=False, fastmath=False)
        def product_t2(values):
            n_samples, n_rules, n_features, n_intervals = values.shape
            result = np.ones((n_samples, n_rules, n_intervals), dtype=np.float64)
            for sample in range(n_samples):
                for rule in range(n_rules):
                    for interval in range(n_intervals):
                        value = 1.0
                        for feature in range(n_features):
                            value *= values[sample, rule, feature, interval]
                        result[sample, rule, interval] = value
            return result

        return product_t2
    except Exception:
        return None
def _torch_product(values: np.ndarray, device: str) -> np.ndarray:
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    result = torch.prod(tensor, dim=2)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return result.detach().cpu().numpy()


def _torch_ordered_product(values: np.ndarray, device: str) -> np.ndarray:
    """Multiply one feature plane at a time, preserving NumPy's loop order."""
    import torch

    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    result = torch.ones(
        tensor.shape[:2] + tensor.shape[3:], dtype=torch.float64, device=device
    )
    for feature in range(tensor.shape[2]):
        result = result * tensor[:, :, feature, ...]
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return result.detach().cpu().numpy()


def _torch_compiled_ordered_product(values: np.ndarray, device: str) -> np.ndarray:
    """Opt-in D05 probe; compilation is intentionally never used by default."""
    import torch

    def ordered(tensor):
        result = torch.ones(
            tensor.shape[:2] + tensor.shape[3:], dtype=tensor.dtype, device=tensor.device
        )
        for feature in range(tensor.shape[2]):
            result = result * tensor[:, :, feature, ...]
        return result

    compiled = torch.compile(ordered, fullgraph=True)
    tensor = torch.as_tensor(values, dtype=torch.float64, device=device)
    result = compiled(tensor)
    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    return result.detach().cpu().numpy()


def _case(
    seed: int, n_samples: int, n_rules: int, n_features: int, t2: bool = False
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (n_samples, n_rules, n_features, 2) if t2 else (
        n_samples,
        n_rules,
        n_features,
    )
    values = rng.uniform(0.0, 1.0, shape).astype(
        np.float64
    )
    # Exercise the common fuzzy-membership zero path without making every
    # result zero.  A second seeded case below covers all-nonzero values.
    values[0, 0, 0, ...] = 0.0
    values[-1, -1, -1, ...] = 0.0
    return values


def _compare(name: str, expected: np.ndarray, actual: np.ndarray) -> str:
    equal = np.array_equal(expected, actual)
    maxdiff = float(np.max(np.abs(expected - actual))) if expected.size else 0.0
    mismatch_count = int(np.count_nonzero(expected != actual))
    status = "PASS" if equal else "FAIL"
    return (
        f"{status} {name}: mismatches={mismatch_count} "
        f"maxdiff={maxdiff:.17g} shape={expected.shape}"
    )


def parity_check(compile_torch: bool = False) -> int:
    """Run deterministic exact NumPy-vs-backend parity checks."""
    numba_product = _make_numba_product()
    numba_product_t2 = _make_numba_product_t2()
    torch_available = importlib.util.find_spec("torch") is not None
    torch_cuda = False
    if torch_available:
        try:
            import torch

            torch_cuda = bool(torch.cuda.is_available())
        except Exception as exc:
            print(f"SKIP torch import/probe failed: {type(exc).__name__}: {exc}")

    failures = 0
    if numba_product is None:
        print("SKIP numba: unavailable or import failed")
    if numba_product_t2 is None:
        print("SKIP numba T2: unavailable or import failed")
    if not torch_available:
        print("SKIP torch: unavailable")

    for case_index, n_features in enumerate(SHAPES):
        for t2 in (False, True):
            for all_nonzero in (False, True):
                values = _case(
                    1000 + case_index * 4 + int(t2) * 2 + int(all_nonzero),
                    5,
                    4,
                    n_features,
                    t2=t2,
                )
                if all_nonzero:
                    values = np.maximum(values, np.finfo(np.float64).tiny)
                expected = numpy_product(values)
                label = (
                    f"{'T2' if t2 else 'T1'} features={n_features} "
                    f"{'nonzero' if all_nonzero else 'zeros'}"
                )

                selected_numba = numba_product_t2 if t2 else numba_product
                if selected_numba is not None:
                    try:
                        actual = selected_numba(values)
                        line = _compare(f"numba {label}", expected, actual)
                        print(line)
                        failures += not line.startswith("PASS")
                    except Exception as exc:
                        print(f"FAIL numba {label}: {type(exc).__name__}: {exc}")
                        failures += 1

                if torch_available:
                    try:
                        actual = _torch_product(values, "cpu")
                        line = _compare(f"torch-cpu {label}", expected, actual)
                        print(line)
                        failures += not line.startswith("PASS")
                    except Exception as exc:
                        print(f"FAIL torch-cpu {label}: {type(exc).__name__}: {exc}")
                        failures += 1

                    if torch_cuda:
                        try:
                            actual = _torch_product(values, "cuda")
                            line = _compare(f"torch-cuda {label}", expected, actual)
                            print(line)
                            failures += not line.startswith("PASS")
                        except Exception as exc:
                            print(f"FAIL torch-cuda {label}: {type(exc).__name__}: {exc}")
                            failures += 1

                        try:
                            actual = _torch_ordered_product(values, "cuda")
                            line = _compare(
                                f"torch-cuda-ordered {label}", expected, actual
                            )
                            print(line)
                            failures += not line.startswith("PASS")
                        except Exception as exc:
                            print(
                                f"FAIL torch-cuda-ordered {label}: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            failures += 1

                        if compile_torch:
                            try:
                                actual = _torch_compiled_ordered_product(values, "cuda")
                                line = _compare(
                                    f"torch-cuda-compiled-ordered {label}", expected, actual
                                )
                                print(line)
                                failures += not line.startswith("PASS")
                            except Exception as exc:
                                print(
                                    f"FAIL torch-cuda-compiled-ordered {label}: "
                                    f"{type(exc).__name__}: {exc}"
                                )
                                failures += 1

    return int(failures)


def timing_check() -> None:
    """Optional lightweight timing diagnostic; never runs by default.

    These are approximate kernel timings only, not complete-fit measurements.
    """
    numba_product = _make_numba_product()
    numba_product_t2 = _make_numba_product_t2()
    print("TIMING diagnostic (not a production benchmark)")
    for t2 in (False, True):
        values = _case(777 + int(t2), 1024, 64, 33, t2=t2)
        suffix = "T2" if t2 else "T1"
        funcs = [("numpy", lambda values=values: numpy_product(values))]
        selected_numba = numba_product_t2 if t2 else numba_product
        if selected_numba is not None:
            funcs.append(("numba", lambda f=selected_numba, values=values: f(values)))
        if importlib.util.find_spec("torch") is not None:
            funcs.append(("torch-cpu-prod", lambda values=values: _torch_product(values, "cpu")))
            funcs.append(("torch-cpu-ordered", lambda values=values: _torch_ordered_product(values, "cpu")))
            try:
                import torch

                if torch.cuda.is_available():
                    # Transfer to/from CUDA is intentionally included in these
                    # timings; synchronize is performed inside each function.
                    funcs.append(("torch-cuda-prod-transfer", lambda values=values: _torch_product(values, "cuda")))
                    funcs.append(("torch-cuda-ordered-transfer", lambda values=values: _torch_ordered_product(values, "cuda")))
            except Exception:
                pass
        for name, function in funcs:
            start = time.perf_counter()
            function()
            cold = time.perf_counter() - start
            warm_samples = []
            for _ in range(3):
                start = time.perf_counter()
                function()
                warm_samples.append(time.perf_counter() - start)
            median_warm = float(np.median(warm_samples))
            print(
                f"{suffix} {name} cold_seconds={cold:.9f} "
                f"warm_median_seconds={median_warm:.9f} repeats=3"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="opt-in torch.compile CUDA parity probe (may compile; not a timing run)",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="also run an explicitly requested timing diagnostic",
    )
    args = parser.parse_args()
    failures = parity_check(compile_torch=args.compile)
    if args.timing:
        timing_check()
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
