"""Summarize the EvoX GPU comparison against the recorded CPU scaling campaign.

Reads the per-task JSON files written by ``benchmarks/cluster/submit_evox_gpu.sh``
and prints one row per (samples, features, partitions) workload:

- EvoX ``cpu`` and ``device`` median seconds across seeds, measured on the same
  GPU node, and their ratio (the GPU speedup of an identical search);
- whether every seed's EvoX routes produced identical searches, whether every
  device verification passed, and how many generations the device scored;
- the recorded Ex-Fuzzy 3.0 PyMoo median from ``docs/performance/t1_scaling.json``
  for context only: it ran a different genetic algorithm on another machine.

    python benchmarks/summarize_evox_gpu.py
    python benchmarks/summarize_evox_gpu.py --json benchmarks/results/evox_gpu/summary.json
    python benchmarks/summarize_evox_gpu.py --plot docs/performance/evox_gpu.svg
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CURRENT = "Ex-Fuzzy 3.0"


def _recorded_cpu(path: Path) -> dict:
    if not path.exists():
        return {}
    report = json.loads(path.read_text())
    return {(row["samples"], row["settings"]["features"],
             "fixed" if row["fixed"] else "optimized"): row["median_seconds"][CURRENT]
            for row in report.get("results", []) if row.get("fuzzy_type") == "t1"}


def plot(rows: list, groups: dict, destination: Path) -> None:
    """Draw EvoX CPU and GPU fit times for the largest workload, one bar pair per partitioning."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    size = max((row["samples"], row["features"]) for row in rows)
    rows = [row for row in rows if (row["samples"], row["features"]) == size and row.get("gpu_speedup")]
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for index, (route, label, color) in enumerate((("cpu", "EvoX CPU", "#536878"),
                                                   ("device", "EvoX GPU", "#007d73"))):
        values = [row[f"{route}_seconds"] for row in rows]
        seeds = [[w["median_seconds"][route] for w in groups[(row["samples"], row["features"], row["partitions"])]]
                 for row in rows]
        low = [value - min(times) for value, times in zip(values, seeds)]
        high = [max(times) - value for value, times in zip(values, seeds)]
        x = [i + (index - .5) * .36 for i in range(len(rows))]
        ax.bar(x, values, .36, label=label, color=color, yerr=[low, high], capsize=3)
        for position, value, times in zip(x, values, seeds):
            ax.text(position, max(times) + 10, f"{value:.1f} s", ha="center", va="bottom", fontsize=9)
    for i, row in enumerate(rows):
        top = max(w["median_seconds"]["cpu"] for w in groups[(row["samples"], row["features"], row["partitions"])])
        ax.text(i, top * 1.12, f"{row['gpu_speedup']:.2f}× faster", ha="center", weight="bold")
    ax.set_xticks(range(len(rows)), [f"{row['partitions'].capitalize()} partitions" for row in rows])
    ax.set_ylabel("Complete fit (seconds)")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper left")
    ax.set_title(f"EvoX CPU vs GPU · T1 · {size[0]:,} samples · {size[1]} features\n"
                 f"Median complete fit over {len(rows[0]['seeds'])} seeds; whiskers show min–max")
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/evox_gpu")
    parser.add_argument("--cpu", type=Path, default=ROOT / "docs/performance/t1_scaling.json")
    parser.add_argument("--json", type=Path, help="also write the summary to this file")
    parser.add_argument("--plot", type=Path, help="also draw the CPU/GPU figure to this file")
    args = parser.parse_args()

    groups = defaultdict(list)
    hosts = set()
    for path in sorted(args.results.glob("n*_d*_seed*.json")):
        report = json.loads(path.read_text())
        hosts.add((report.get("host"), report.get("gpu"), report.get("cpu")))
        for workload in report["workloads"]:
            key = (workload["samples"], workload["features"], workload["partitions"])
            groups[key].append(workload)
    failures = sorted(path.name for path in args.results.glob("*.failed"))
    recorded = _recorded_cpu(args.cpu)

    rows = []
    for key in sorted(groups):
        workloads = groups[key]
        identical = all(w["evox_identical"] for w in workloads)
        row = {"samples": key[0], "features": key[1], "partitions": key[2],
               "seeds": sorted(w["seed"] for w in workloads), "identical": identical,
               "recorded_pymoo_seconds": recorded.get(key)}
        if identical:
            for route in ("cpu", "device", "pymoo"):
                values = [w["median_seconds"][route] for w in workloads
                          if route in w.get("median_seconds", {})]
                row[f"{route}_seconds"] = statistics.median(values) if values else None
            device_runs = [run for w in workloads for run in w["runs"].get("device", [])]
            verifications = [v for run in device_runs for v in run.get("verifications", [])]
            row["verified"] = bool(verifications) and all(v["verified"] for v in verifications)
            row["device_generations"] = sum(run["device_generations"] for run in device_runs)
            row["scored_generations"] = sum(run["scored_generations"] for run in device_runs)
            if row.get("cpu_seconds") and row.get("device_seconds"):
                row["gpu_speedup"] = row["cpu_seconds"] / row["device_seconds"]
        rows.append(row)

    for host, gpu, cpu in sorted(hosts, key=str):
        print(f"node {host}: gpu={gpu} cpu={cpu}")
    print("\n| Samples | Features | Partitions | Seeds | EvoX CPU (s) | EvoX GPU (s) | GPU speedup "
          "| Identical | Verified | Device gens | Recorded PyMoo 3.0 (s) |")
    print("| ---: | ---: | --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |")
    for row in rows:
        def seconds(name):
            value = row.get(name)
            return f"{value:.2f}" if value is not None else "-"
        speedup = f"{row['gpu_speedup']:.2f}x" if row.get("gpu_speedup") else "-"
        generations = (f"{row['device_generations']}/{row['scored_generations']}"
                       if "scored_generations" in row else "-")
        print(f"| {row['samples']:,} | {row['features']} | {row['partitions']} | "
              f"{','.join(map(str, row['seeds']))} | {seconds('cpu_seconds')} | "
              f"{seconds('device_seconds')} | {speedup} | {row['identical']} | "
              f"{row.get('verified', '-')} | {generations} | {seconds('recorded_pymoo_seconds')} |")
    if failures:
        print("\nFailed tasks: " + ", ".join(failures))
    if args.json is not None:
        args.json.write_text(json.dumps({"nodes": sorted(map(list, hosts), key=str),
                                         "rows": rows, "failures": failures}, indent=2))
    if args.plot is not None:
        plot(rows, groups, args.plot)


if __name__ == "__main__":
    main()
