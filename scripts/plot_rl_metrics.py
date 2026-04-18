#!/usr/bin/env python3
import argparse
import math
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+):(-?\d+(?:\.\d+)?)")
INVALID_RE = re.compile(r"Round\s+(\d+):\s+invalid draft format count\s*=\s*(\d+)/(\d+)")


DEFAULT_METRICS = [
    "global_seqlen/mean",
    "actor/pg_loss",
    "actor/pg_clipfrac",
    "actor/ppo_kl",
    "actor/kl_loss",
    "actor/entropy_loss",
    "actor/grad_norm",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "response_length/mean",
    "prompt_length/mean",
    "timing_s/gen",
    "timing_s/ref",
    "timing_s/adv",
    "timing_s/update_actor",
    "timing_s/step",
    "mfu/actor",
]


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s).strip()


def find_latest_log(default_dir: str) -> str:
    """Find the most recently modified training log under project root."""
    if not os.path.isdir(default_dir):
        raise RuntimeError(f"Project directory not found: {default_dir}")

    candidates: List[str] = []
    for fn in os.listdir(default_dir):
        if not fn.endswith(".log"):
            continue
        # Prefer paper-writing training logs but keep generic fallback.
        if fn.startswith("paper-writing-") or "paper-writing" in fn or "grpo" in fn:
            candidates.append(os.path.join(default_dir, fn))

    if not candidates:
        # fallback: any .log
        for fn in os.listdir(default_dir):
            if fn.endswith(".log"):
                candidates.append(os.path.join(default_dir, fn))

    if not candidates:
        raise RuntimeError(f"No .log files found under: {default_dir}")

    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def parse_log(log_path: str) -> Tuple[List[int], List[int], Dict[str, List[float]], Dict[int, List[float]]]:
    step_ids: List[int] = []
    x_idx: List[int] = []
    metrics: Dict[str, List[float]] = defaultdict(list)
    invalid_by_round: Dict[int, List[float]] = defaultdict(list)

    row_count = 0
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = strip_ansi(raw)
            if not line:
                continue

            im = INVALID_RE.search(line)
            if im:
                rnd = int(im.group(1))
                bad = int(im.group(2))
                total = int(im.group(3))
                invalid_by_round[rnd].append((bad / total) if total > 0 else 0.0)
                continue

            sm = STEP_RE.search(line)
            if not sm:
                continue

            step = int(sm.group(1))
            payload = sm.group(2)
            kvs = {m.group(1): float(m.group(2)) for m in KV_RE.finditer(payload)}

            step_ids.append(step)
            x_idx.append(row_count)
            row_count += 1

            all_keys = set(metrics.keys()) | set(kvs.keys())
            for k in all_keys:
                if k not in metrics:
                    metrics[k] = [math.nan] * (row_count - 1)
                metrics[k].append(kvs.get(k, math.nan))
            for k in metrics:
                if len(metrics[k]) < row_count:
                    metrics[k].append(math.nan)

    return x_idx, step_ids, metrics, invalid_by_round


def _group_key(metric_name: str) -> str:
    """Extract the first prefix as group name, e.g. 'actor/pg_loss' -> 'actor'."""
    if "/" in metric_name:
        return metric_name.split("/", 1)[0]
    return "other"


def _plot_group(
    group_name: str,
    keys: List[str],
    x_idx: List[int],
    step_ids: List[int],
    metrics: Dict[str, List[float]],
    out_dir: str,
) -> str:
    """Plot all metrics in one group onto a single figure, return the saved path."""
    ncols = min(3, len(keys))
    nrows = math.ceil(len(keys) / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, k in enumerate(keys):
        ax = axes_flat[i]
        y = metrics[k]
        ax.plot(x_idx, y, linewidth=1.4)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("step index")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)

    for j in range(len(keys), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"{group_name} | steps={len(step_ids)} | first={step_ids[0]} | last={step_ids[-1]}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png = os.path.join(out_dir, f"rl_metrics_{group_name}.png")
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png


def plot_metrics(
    x_idx: List[int],
    step_ids: List[int],
    metrics: Dict[str, List[float]],
    invalid_by_round: Dict[int, List[float]],
    selected_metrics: List[str],
    out_png: str,
) -> None:
    plot_keys = [k for k in selected_metrics if k in metrics]
    if not plot_keys:
        raise RuntimeError("No selected metrics found in parsed log.")

    # ---- per-group figures ----
    out_dir = os.path.dirname(out_png) or "."
    groups: Dict[str, List[str]] = defaultdict(list)
    for k in plot_keys:
        groups[_group_key(k)].append(k)

    saved_files: List[str] = []
    for gname in sorted(groups.keys()):
        gkeys = sorted(groups[gname])
        path = _plot_group(gname, gkeys, x_idx, step_ids, metrics, out_dir)
        saved_files.append(path)
        print(f"[OK] Group figure saved: {path}")

    # ---- combined overview figure (legacy) ----
    extra = 1 if invalid_by_round else 0
    total_panels = len(plot_keys) + extra
    ncols = 3
    nrows = math.ceil(total_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(6.2 * ncols, 3.6 * nrows), squeeze=False)
    axes_flat = [ax for row in axes for ax in row]

    for i, k in enumerate(plot_keys):
        ax = axes_flat[i]
        y = metrics[k]
        ax.plot(x_idx, y, linewidth=1.4)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("step index")
        ax.set_ylabel("value")
        ax.grid(alpha=0.3)

    cursor = len(plot_keys)
    if invalid_by_round:
        ax = axes_flat[cursor]
        for rnd in sorted(invalid_by_round.keys()):
            y = invalid_by_round[rnd]
            x = list(range(len(y)))
            ax.plot(x, y, marker="o", linewidth=1.2, markersize=3, label=f"round{rnd}")
        ax.set_title("invalid_draft_ratio_by_round", fontsize=10)
        ax.set_xlabel("occurrence index")
        ax.set_ylabel("ratio")
        ax.set_ylim(0.0, 1.0)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for j in range(total_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(
        f"RL Metrics Trend | parsed_steps={len(step_ids)} | first_step={step_ids[0]} | last_step={step_ids[-1]}",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def save_csv(out_csv: str, x_idx: List[int], step_ids: List[int], metrics: Dict[str, List[float]]) -> None:
    keys = sorted(metrics.keys())
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("parsed_index,step," + ",".join(keys) + "\n")
        for i in range(len(x_idx)):
            row = [str(x_idx[i]), str(step_ids[i])]
            for k in keys:
                v = metrics[k][i]
                row.append("" if math.isnan(v) else f"{v}")
            f.write(",".join(row) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot RL training metric trends from a log.")
    parser.add_argument(
        "--log",
        default=None,
        help="Path to training log file. If omitted, auto-pick latest .log under project root.",
    )
    parser.add_argument(
        "--out-dir",
        default="/home/wangzixu/Search-R1/monitor_outputs",
        help="Directory to save plots and parsed csv.",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        default=True,
        help="Plot all parsed metrics (default). Use --no-all-metrics for curated subset.",
    )
    parser.add_argument(
        "--no-all-metrics",
        action="store_false",
        dest="all_metrics",
        help="Plot only curated key metrics instead of all.",
    )
    args = parser.parse_args()

    if args.log is None:
        args.log = find_latest_log("/home/wangzixu/Search-R1")

    os.makedirs(args.out_dir, exist_ok=True)
    x_idx, step_ids, metrics, invalid_by_round = parse_log(args.log)
    if not step_ids:
        raise RuntimeError(f"No step lines parsed from log: {args.log}")

    selected = sorted(metrics.keys()) if args.all_metrics else DEFAULT_METRICS
    out_png = os.path.join(args.out_dir, "rl_metrics_trend.png")
    out_csv = os.path.join(args.out_dir, "rl_metrics_parsed.csv")

    plot_metrics(x_idx, step_ids, metrics, invalid_by_round, selected, out_png)
    save_csv(out_csv, x_idx, step_ids, metrics)

    print(f"[OK] Figure saved: {out_png}")
    print(f"[OK] CSV saved:    {out_csv}")
    print(f"[OK] Parsed steps: {len(step_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
