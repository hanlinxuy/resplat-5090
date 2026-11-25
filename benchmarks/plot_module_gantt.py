"""Plot relative GPU event spans exported from one profiled training update."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("timeline", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--metrics", type=Path, help="Optional separate Nsight hardware capture summary")
    args = parser.parse_args()
    trace = json.loads(args.timeline.read_text())
    lanes = {row["label"]: row["intervals_ms"] for row in trace["lanes"]}
    rows = [
        ("Depth predictor", "Mixed: matrix, scalar ALU, memory", "#7b65b8"),
        ("Gaussian regressor / head", "Matrix: convolution", "#2e76a7"),
        ("Point Transformer", "Matrix + layout + normalization", "#2e76a7"),
        ("gsplat", "Raster / irregular memory access", "#408b78"),
        ("VGG / loss", "Matrix: convolution, plus loss", "#2e76a7"),
        ("AdamW / update", "Streaming ALU + memory", "#408b78"),
        ("Other", "Mixed / unattributed", "#9aa4b1"),
    ]
    plt.rcParams["svg.hashsalt"] = "resplat-module-gantt"
    fig, ax = plt.subplots(figsize=(13, 5.0))
    fig.patch.set_facecolor("#fafbfe")
    ax.set_facecolor("#fafbfe")
    fig.subplots_adjust(left=0.205, right=0.75, top=0.76, bottom=0.20)
    fig.text(0.045, 0.933, "RTX 4090  |  ReSplat training step", fontsize=19, weight="bold", color="#182132")
    fig.text(0.046, 0.873, "6 layers   •   256×448   •   B1   •   8 input + 6 target views", fontsize=10, color="#4e5c70")
    if args.metrics:
        stages = json.loads(args.metrics.read_text())["stages"]
        step_tcu = stages["whole_update"]["median"]["tensor_pct"]
        backward_tcu = stages["backward"]["median"]["tensor_pct"]
        fig.text(0.046, 0.825, f"Tensor Active in separate Nsight capture: {step_tcu:.2f}% whole step, {backward_tcu:.2f}% backward",
                 fontsize=9, color="#4e5c70")
    for key, label, color in [
        ("train_forward_loss", "Forward + loss", "#e1eafa"),
        ("train_backward", "Backward", "#ede6fa"),
        ("train_update", "Update", "#e1f0ea"),
    ]:
        lo, hi = lanes[key][0]
        ax.axvspan(lo, hi, color=color, linewidth=0, zorder=0)
        if key == "train_update":
            ax.text(hi + 1, -0.78, label, ha="right", fontsize=9, color="#4e5c70")
        else:
            ax.text((lo + hi) / 2, -0.78, label, ha="center", fontsize=10, weight="bold", color="#39445a")
    for y, (name, kind, color) in enumerate(rows):
        ax.axhline(y + 0.49, color="#dbe2ed", linewidth=0.45, zorder=1)
        bars = [(lo, hi - lo) for lo, hi in lanes[name]]
        ax.broken_barh(bars, (y - 0.22, 0.44), facecolors=color, edgecolors="none", zorder=2)
        ax.text(1.023, y, kind, transform=ax.get_yaxis_transform(), va="center", fontsize=9, color="#35445a")
    ax.set_ylim(len(rows) - 0.44, -0.98)
    ax.set_xlim(0, lanes["train_update"][-1][-1] + 6)
    ax.set_yticks(range(len(rows)), [row[0] for row in rows], fontsize=9)
    ax.set_xlabel("GPU timeline (ms from first kernel)", color="#35445a", fontsize=10)
    ax.grid(axis="x", alpha=0.32, linewidth=0.7)
    ax.tick_params(length=0, colors="#35445a")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(0.046, 0.083, "Colored bars show GPU events by originating module. Work from different modules can interleave.",
             fontsize=9, color="#4e5c70")
    fig.text(0.046, 0.043, "Right-side labels describe dominant operations. A module label alone does not establish hardware saturation.",
             fontsize=9, color="#4e5c70")
    fig.savefig(args.output, dpi=160, facecolor=fig.get_facecolor())
    if args.output.suffix == ".svg":
        svg = args.output.read_text()
        svg = re.sub(r'<path([^>]*?) d="([^"]+)"', lambda m: f'<path{m[1]} d="{" ".join(m[2].split())}"', svg)
        args.output.write_text(svg)
