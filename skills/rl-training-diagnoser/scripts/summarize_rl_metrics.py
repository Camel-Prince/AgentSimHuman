#!/usr/bin/env python3
"""Summarize RL training metrics from a parsed W&B/local CSV."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any


KEY_METRICS = [
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/ppo_kl",
    "actor/pg_clipfrac",
    "actor/pg_loss",
    "actor/entropy_loss",
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "response_length/mean",
    "global_seqlen/mean",
]

CONFIG_PATTERNS = {
    "use_kl_loss": r"(?:actor_rollout_ref\.actor\.use_kl_loss|['\"]use_kl_loss['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "actor_kl_loss_coef": r"(?:actor_rollout_ref\.actor\.kl_loss_coef|['\"]kl_loss_coef['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "kl_ctrl_coef": r"(?:algorithm\.kl_ctrl\.kl_coef|['\"]kl_ctrl['\"]\s*:\s*\{[^}]*['\"]kl_coef['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "adv_estimator": r"(?:algorithm\.adv_estimator|['\"]adv_estimator['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "state_masking": r"(?:actor_rollout_ref\.actor\.state_masking|['\"]state_masking['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "task_type": r"(?:task_type|['\"]task_type['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
    "reward_type": r"(?:reward_type|['\"]reward_type['\"])\s*[:=]\s*(['\"]?[A-Za-z0-9_.+-]+['\"]?)",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-csv", required=True, help="Path to parsed metrics CSV.")
    parser.add_argument("--log", help="Optional training log path for config and invalid draft counts.")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown")
    parser.add_argument("--window", type=int, default=5, help="Rows per first/last summary window.")
    return parser.parse_args()


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(value)
    except ValueError:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def load_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, reader.fieldnames or []


def metric_values(rows: list[dict[str, str]], name: str) -> list[tuple[int, float]]:
    values: list[tuple[int, float]] = []
    for idx, row in enumerate(rows):
        value = to_float(row.get(name))
        if value is not None:
            values.append((idx, value))
    return values


def avg(items: list[float]) -> float | None:
    return mean(items) if items else None


def slope(values: list[tuple[int, float]]) -> float | None:
    if len(values) < 2:
        return None
    xs = [float(x) for x, _ in values]
    ys = [y for _, y in values]
    x_mean = mean(xs)
    y_mean = mean(ys)
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom


def corr(values: list[tuple[int, float]]) -> float | None:
    if len(values) < 2:
        return None
    xs = [float(x) for x, _ in values]
    ys = [y for _, y in values]
    x_mean = mean(xs)
    y_mean = mean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var == 0 or y_var == 0:
        return None
    return sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / math.sqrt(x_var * y_var)


def pct_change(first: float | None, last: float | None) -> float | None:
    if first is None or last is None or first == 0:
        return None
    return (last - first) / abs(first)


def summarize_metric(rows: list[dict[str, str]], name: str, window: int) -> dict[str, Any] | None:
    values = metric_values(rows, name)
    if not values:
        return None
    ys = [v for _, v in values]
    first_values = [v for _, v in values[:window]]
    last_values = [v for _, v in values[-window:]]
    mid_start = max(0, (len(values) // 2) - (window // 2))
    mid_values = [v for _, v in values[mid_start : mid_start + window]]
    first_avg = avg(first_values)
    last_avg = avg(last_values)
    return {
        "count": len(values),
        "first": ys[0],
        "last": ys[-1],
        "min": min(ys),
        "max": max(ys),
        "first_window_mean": first_avg,
        "mid_window_mean": avg(mid_values),
        "last_window_mean": last_avg,
        "pct_change_first_to_last_window": pct_change(first_avg, last_avg),
        "slope_per_row": slope(values),
        "step_correlation": corr(values),
    }


def discover_metrics(fieldnames: list[str]) -> list[str]:
    metrics = list(KEY_METRICS)
    for field in fieldnames:
        lowered = field.lower()
        if "invalid" in lowered and ("draft" in lowered or "format" in lowered):
            metrics.append(field)
        elif lowered.startswith("paper_writing/"):
            metrics.append(field)
    seen: set[str] = set()
    out: list[str] = []
    for metric in metrics:
        if metric not in seen:
            seen.add(metric)
            out.append(metric)
    return out


def extract_log_info(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    redacted = re.sub(r"(?i)(api[_-]?key|token|secret)(\s*[:=]\s*)\S+", r"\1\2<redacted>", text)
    config: dict[str, str] = {}
    for key, pattern in CONFIG_PATTERNS.items():
        match = re.search(pattern, redacted, flags=re.DOTALL)
        if match:
            config[key] = match.group(1).strip("'\"")
    invalid_patterns = [
        r"round\s*([123])[^0-9\n]*(?:invalid|invalid draft)[^0-9\n]*(\d+)\s*/\s*(\d+)",
        r"(?:invalid|invalid draft)[^0-9\n]*round\s*([123])[^0-9\n]*(\d+)\s*/\s*(\d+)",
    ]
    invalid_counts: list[dict[str, int]] = []
    for pattern in invalid_patterns:
        for round_id, invalid, total in re.findall(pattern, redacted, flags=re.IGNORECASE):
            invalid_counts.append(
                {"round": int(round_id), "invalid": int(invalid), "total": int(total)}
            )
    return {"path": str(path), "config": config, "invalid_counts": invalid_counts[:200]}


def flag_findings(summary: dict[str, Any]) -> list[str]:
    metrics = summary["metrics"]
    findings: list[str] = []

    def last(name: str) -> float | None:
        item = metrics.get(name)
        return item.get("last_window_mean") if item else None

    def change(name: str) -> float | None:
        item = metrics.get(name)
        return item.get("pct_change_first_to_last_window") if item else None

    def corr_value(name: str) -> float | None:
        item = metrics.get(name)
        return item.get("step_correlation") if item else None

    if (change("response_length/mean") or 0) < -0.05 or (change("global_seqlen/mean") or 0) < -0.05:
        findings.append("Response length or global sequence length is materially decreasing; inspect short-output reward hacking or overly strong penalties.")
    if (change("actor/kl_loss") or 0) > 1.0 and (corr_value("actor/kl_loss") or 0) > 0.4:
        findings.append("Actor KL loss rises strongly; if use_kl_loss=true, check whether the KL coefficient is large enough to control policy drift.")
    if (corr_value("actor/pg_clipfrac") or 0) > 0.4 and (corr_value("actor/ppo_kl") or 0) > 0.4:
        findings.append("PPO clip fraction and PPO KL both trend upward; actor updates may be drifting or becoming constrained.")
    elif (corr_value("actor/pg_clipfrac") or 0) > 0.4:
        findings.append("PPO clip fraction trends upward; more token updates are reaching the clipping region.")
    if (corr_value("actor/grad_norm") or 0) > 0.4:
        findings.append("Actor grad norm trends upward; inspect learning rate, reward variance, KL control, and batch composition.")
    if (change("actor/entropy_loss") or 0) < -0.2 and (corr_value("actor/entropy_loss") or 0) < -0.4:
        findings.append("Actor entropy loss decreases strongly; output diversity may be collapsing or becoming template-like.")
    score_delta = change("critic/score/mean")
    reward_delta = change("critic/rewards/mean")
    if score_delta is not None and reward_delta is not None and score_delta > 0 and reward_delta < 0:
        findings.append("Score improves while reward declines; if reward-path KL is enabled, KL penalty may exceed task-score gains.")
    if last("critic/advantages/mean") is not None and (change("critic/advantages/mean") or 0) < -0.1:
        findings.append("Advantages trend downward; inspect next-rollout quality, reward normalization, and whether penalties dominate.")
    return findings


def build_summary(metrics_csv: Path, log_path: Path | None, window: int) -> dict[str, Any]:
    rows, fieldnames = load_csv(metrics_csv)
    selected = discover_metrics(fieldnames)
    metrics: dict[str, Any] = {}
    missing: list[str] = []
    for name in selected:
        item = summarize_metric(rows, name, window)
        if item is None:
            if name in KEY_METRICS:
                missing.append(name)
            continue
        metrics[name] = item
    summary = {
        "metrics_csv": str(metrics_csv),
        "row_count": len(rows),
        "column_count": len(fieldnames),
        "metrics": metrics,
        "missing_key_metrics": missing,
        "log": extract_log_info(log_path),
    }
    summary["findings"] = flag_findings(summary)
    return summary


def fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# RL Metrics Summary",
        "",
        f"- Metrics CSV: `{summary['metrics_csv']}`",
        f"- Rows: {summary['row_count']}",
        f"- Columns: {summary['column_count']}",
    ]
    log = summary.get("log") or {}
    if log.get("path"):
        lines.append(f"- Log: `{log['path']}`")
    if log.get("config"):
        lines.append("")
        lines.append("## Extracted Config")
        for key, value in sorted(log["config"].items()):
            lines.append(f"- `{key}`: `{value}`")
    if summary["missing_key_metrics"]:
        lines.append("")
        lines.append("## Missing Key Metrics")
        for metric in summary["missing_key_metrics"]:
            lines.append(f"- `{metric}`")
    lines.append("")
    lines.append("## Metric Windows")
    lines.append("| metric | first_mean | mid_mean | last_mean | pct_change | slope | corr_step | min | max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name, item in summary["metrics"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{name}`",
                    fmt(item["first_window_mean"]),
                    fmt(item["mid_window_mean"]),
                    fmt(item["last_window_mean"]),
                    fmt(item["pct_change_first_to_last_window"]),
                    fmt(item["slope_per_row"]),
                    fmt(item["step_correlation"]),
                    fmt(item["min"]),
                    fmt(item["max"]),
                ]
            )
            + " |"
        )
    if summary["findings"]:
        lines.append("")
        lines.append("## Automatic Findings")
        for finding in summary["findings"]:
            lines.append(f"- {finding}")
    invalid_counts = (summary.get("log") or {}).get("invalid_counts") or []
    if invalid_counts:
        lines.append("")
        lines.append("## Invalid Draft Counts From Log")
        for item in invalid_counts[:30]:
            lines.append(f"- Round {item['round']}: {item['invalid']}/{item['total']}")
        if len(invalid_counts) > 30:
            lines.append(f"- ... {len(invalid_counts) - 30} additional entries omitted")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    metrics_csv = Path(args.metrics_csv)
    log_path = Path(args.log) if args.log else None
    summary = build_summary(metrics_csv, log_path, args.window)
    if args.format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_markdown(summary))


if __name__ == "__main__":
    main()
