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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=ROOT / "benchmarks/results/evox_gpu")
    parser.add_argument("--cpu", type=Path, default=ROOT / "docs/performance/t1_scaling.json")
    parser.add_argument("--json", type=Path, help="also write the summary to this file")
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


if __name__ == "__main__":
    main()
