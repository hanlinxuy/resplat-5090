"""Extract phases, kernel hotspots and hardware metrics from a ReSplat NSYS export."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sqlite3
from statistics import mean, median


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--chrome-trace", type=Path,
                        help="Export only ReSplat stage GPU spans for Chrome/Perfetto")
    args = parser.parse_args()
    with sqlite3.connect(args.profile) as db:
        events = db.execute("""
            SELECT DISTINCT n.text, n.start, k.start, k.end, s.value
            FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                ON r.start BETWEEN n.start AND n.end
            JOIN CUPTI_ACTIVITY_KIND_KERNEL k USING (correlationId)
            JOIN StringIds s ON s.id = k.demangledName
            WHERE n.text LIKE 'resplat/%'
            ORDER BY n.start, k.start
        """).fetchall()
        launches = db.execute("""
            SELECT n.text, n.start, r.start, r.end, s.value
            FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r
                ON r.start BETWEEN n.start AND n.end
            JOIN StringIds s ON s.id = r.nameId
            WHERE n.text LIKE 'resplat/train_%' AND LOWER(s.value) LIKE '%launch%'
        """).fetchall()
        metrics = defaultdict(list)
        if db.execute("SELECT 1 FROM sqlite_master WHERE name = 'GPU_METRICS'").fetchone():
            for name, time, value in db.execute("""
                SELECT m.metricName, g.timestamp, g.value
                FROM GPU_METRICS g JOIN TARGET_INFO_GPU_METRICS m USING (metricId, typeId)
                WHERE m.metricName LIKE '%[Throughput %]'
                ORDER BY g.timestamp
            """):
                if 0 <= value <= 100:
                    metrics[name].append((time, value))
    if not events:
        parser.error("capture contains no CUDA launches in ReSplat NVTX ranges")
    metric_arrays = {name: tuple(zip(*data)) for name, data in metrics.items()}

    def hardware(lo, hi):
        result = {}
        for name, (times, values) in metric_arrays.items():
            sample = values[bisect_left(times, lo):bisect_left(times, hi)]
            if sample:
                result[name] = mean(sample)
        return result

    windows = defaultdict(list)
    for name, cpu_start, gpu_start, gpu_end, kernel in events:
        windows[name, cpu_start].append((gpu_start, gpu_end, kernel))
    cpu = defaultdict(list)
    for name, start, lo, hi, _ in launches:
        cpu[name, start].append((hi - lo) / 1e6)

    stages = defaultdict(list)
    stage_kernels = defaultdict(Counter)
    steps = []
    step_kernels = Counter()
    spans = []
    for (name, start), kernels in windows.items():
        gpu_start = kernels[0][0]
        gpu_end = max(item[1] for item in kernels)
        spans.append((name, gpu_start, gpu_end, len(kernels)))
        stages[name].append({
            "gpu_span_ms": (gpu_end - gpu_start) / 1e6,
            "kernel_sum_ms": sum(hi - lo for lo, hi, _ in kernels) / 1e6,
            "kernel_count": len(kernels),
            "cpu_launch_api_sum_ms": sum(cpu.get((name, start), ())),
        })
        for lo, hi, kernel in kernels:
            stage_kernels[name][kernel] += (hi - lo) / 1e6
        if name == "resplat/train_forward_loss":
            steps.append([])
        if name.startswith("resplat/train_"):
            steps[-1].extend(kernels)
            for lo, hi, kernel in kernels:
                step_kernels[kernel] += (hi - lo) / 1e6
    if not steps:
        parser.error("capture contains no whole training steps")

    step_rows = []
    for kernels in steps:
        kernels.sort()
        lo, hi = kernels[0][:2]
        total_active = 0
        for start, end, _ in kernels:
            if start > hi:
                total_active += hi - lo
                lo = start
            hi = max(hi, end)
        total_active += hi - lo
        start = kernels[0][0]
        span = hi - start
        step_rows.append(dict(gpu_span_ms=span / 1e6, kernel_count=len(kernels),
                              cuda_kernel_gap_pct=(1 - total_active / span) * 100,
                              hardware=hardware(start, hi)))

    result = dict(steps=step_rows, stages={}, top_kernels_mean_ms=[
        [name, ms / len(steps)] for name, ms in step_kernels.most_common(35)
    ])
    for name, values in stages.items():
        result["stages"][name] = {key: median(row[key] for row in values) for key in values[0]}
        result["stages"][name]["top_kernels_mean_ms"] = [
            [kernel, ms / len(values)] for kernel, ms in stage_kernels[name].most_common(8)
        ]
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.chrome_trace:
        spans = [span for span in spans if re.fullmatch(r"resplat/[A-Za-z0-9_.]+", span[0])]
        origin = min(lo for _, lo, _, _ in spans)
        tracks = {"Training phases": 1}
        trace = [{"ph": "M", "name": "process_name", "pid": 1, "args": {"name": "ReSplat GPU spans"}}]
        for name, lo, hi, count in sorted(spans, key=lambda span: span[1]):
            stage = name.removeprefix("resplat/")
            track = "Training phases" if stage.startswith("train_") else stage
            tid = tracks.setdefault(track, len(tracks) + 1)
            trace.append({"ph": "X", "name": stage, "cat": "ReSplat GPU span", "pid": 1, "tid": tid,
                          "ts": (lo - origin) / 1000, "dur": (hi - lo) / 1000,
                          "args": {"CUDA kernels": count}})
        trace.extend({"ph": "M", "name": "thread_name", "pid": 1, "tid": tid, "args": {"name": label}}
                     for label, tid in tracks.items())
        args.chrome_trace.write_text(json.dumps({"traceEvents": trace}, indent=2) + "\n")


if __name__ == "__main__":
    main()
