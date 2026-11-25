"""Read single-process CUDA-correlated NVTX phases and Tensor Active from an Nsight Systems SQLite export."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import sqlite3
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with sqlite3.connect(args.profile) as db:
        samples = db.execute("""
            SELECT g.timestamp, g.value
            FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m
            USING (metricId, typeId)
            WHERE m.metricName = 'Tensor Active [Throughput %]'
            ORDER BY g.timestamp
        """).fetchall()
        rows = db.execute("""
            SELECT n.text, n.start, k.start, k.end, names.value
            FROM NVTX_EVENTS n
            JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                ON r.start BETWEEN n.start AND n.end
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId = r.correlationId
            JOIN StringIds names ON names.id = k.shortName
            WHERE n.text IN ('resplat/knn', 'resplat/point-fwd', 'resplat/point-bwd')
            ORDER BY n.start, k.start
        """).fetchall()
    if not samples or not rows:
        parser.error("profile must contain Tensor Active samples and CUDA-correlated Resplat NVTX phases")
    assert all(0 <= value <= 100 for _, value in samples)
    timestamps, values = zip(*samples)
    kernels = defaultdict(list)
    for phase, launch, start, end, name in rows:
        kernels[(phase, launch)].append((start, end, name))

    records = defaultdict(list)
    top = defaultdict(Counter)
    steps = []
    for (phase, _), ks in kernels.items():
        lo, hi = min(k[0] for k in ks), max(k[1] for k in ks)
        metrics = values[bisect_left(timestamps, lo):bisect_left(timestamps, hi)]
        records[phase].append({"gpu_span_ms": (hi - lo) / 1e6, "tensor_active_pct": mean(metrics)})
        for start, end, name in ks:
            top[phase][name] += (end - start) / 1e6
        if phase == "resplat/knn":
            steps.append([lo, hi])
        else:
            steps[-1][1] = hi
    assert len(steps) == len(records["resplat/point-fwd"]) == len(records["resplat/point-bwd"])
    for lo, hi in steps:
        entries = values[bisect_left(timestamps, lo):bisect_left(timestamps, hi)]
        records["step"].append({"gpu_span_ms": (hi - lo) / 1e6, "tensor_active_pct": mean(entries)})
    report = {phase: {"median_gpu_span_ms": median(r["gpu_span_ms"] for r in vals),
                      "median_tensor_active_pct": median(r["tensor_active_pct"] for r in vals),
                      "steps": vals,
                      "top_kernels_mean_ms": [(name, ms / len(vals)) for name, ms in top[phase].most_common(10)]}
              for phase, vals in records.items()}
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
