#!/usr/bin/env python3
import argparse
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+)\s*-\s*(.*)")
KV_RE = re.compile(r"([A-Za-z0-9_./-]+):(-?\d+(?:\.\d+)?)")
INVALID_RE = re.compile(r"Round\s+(\d+):\s+invalid draft format count\s*=\s*(\d+)/(\d+)")
EPOCH_STEP_RE = re.compile(r"epoch\s+(\d+),\s+step\s+(\d+)")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s).strip()


@dataclass
class ParsedStep:
    step: int
    metrics: Dict[str, float]


@dataclass
class MonitorState:
    recent_steps: Deque[ParsedStep]
    all_steps: int = 0
    last_step: Optional[ParsedStep] = None
    last_epoch: Optional[int] = None
    last_epoch_step: Optional[int] = None
    warning_count: int = 0
    error_count: int = 0
    parse_score_warning_count: int = 0
    invalid_round_latest: Dict[int, Dict[str, float]] = field(default_factory=dict)
    invalid_round_total: Dict[int, int] = field(default_factory=dict)
    invalid_total_seen: int = 0
    invalid_total_count: int = 0

    def ingest_line(self, line: str) -> None:
        s = strip_ansi(line)
        if not s:
            return

        if "[WARNING]" in s:
            self.warning_count += 1
            if "Failed to parse scores from:" in s:
                self.parse_score_warning_count += 1
        if "[ERROR]" in s:
            self.error_count += 1

        em = EPOCH_STEP_RE.search(s)
        if em:
            self.last_epoch = int(em.group(1))
            self.last_epoch_step = int(em.group(2))

        im = INVALID_RE.search(s)
        if im:
            round_id = int(im.group(1))
            bad = int(im.group(2))
            total = int(im.group(3))
            ratio = (bad / total) if total > 0 else 0.0
            self.invalid_round_latest[round_id] = {
                "bad": bad,
                "total": total,
                "ratio": ratio,
            }
            self.invalid_round_total[round_id] = self.invalid_round_total.get(round_id, 0) + 1
            self.invalid_total_seen += bad
            self.invalid_total_count += total

        sm = STEP_RE.search(s)
        if not sm:
            return

        step = int(sm.group(1))
        payload = sm.group(2)
        kvs = {}
        for m in KV_RE.finditer(payload):
            key = m.group(1)
            val = float(m.group(2))
            kvs[key] = val

        parsed = ParsedStep(step=step, metrics=kvs)
        self.last_step = parsed
        self.recent_steps.append(parsed)
        self.all_steps += 1


def load_initial_state(log_path: str, window: int) -> MonitorState:
    st = MonitorState(recent_steps=deque(maxlen=window))
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            st.ingest_line(line)
    return st


def metric_avg(recent: Deque[ParsedStep], key: str) -> Optional[float]:
    vals = [x.metrics[key] for x in recent if key in x.metrics]
    if not vals:
        return None
    return sum(vals) / len(vals)


def fmt(v: Optional[float], ndigits: int = 4) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{ndigits}f}"


def _recent_values(recent: Deque[ParsedStep], key: str) -> List[float]:
    vals: List[float] = []
    for x in recent:
        if key in x.metrics:
            vals.append(float(x.metrics[key]))
    return vals


def build_alerts(st: MonitorState, window: int) -> List[str]:
    alerts: List[str] = []
    if st.last_step is None:
        return alerts

    last = st.last_step.metrics
    recent_rewards = _recent_values(st.recent_steps, "critic/rewards/mean")
    recent_kl = _recent_values(st.recent_steps, "actor/kl_loss")
    recent_grad = _recent_values(st.recent_steps, "actor/grad_norm")
    recent_step_time = _recent_values(st.recent_steps, "timing_s/step")

    # 1) absolute threshold alerts
    if "actor/kl_loss" in last and last["actor/kl_loss"] > 0.20:
        alerts.append(f"KL high: actor/kl_loss={last['actor/kl_loss']:.4f} > 0.20")
    if "actor/grad_norm" in last and last["actor/grad_norm"] > 3.0:
        alerts.append(f"Grad high: actor/grad_norm={last['actor/grad_norm']:.4f} > 3.0")

    # 2) reward plateau in window
    min_window = min(10, window)
    if len(recent_rewards) >= min_window:
        half = len(recent_rewards) // 2
        head = recent_rewards[:half]
        tail = recent_rewards[half:]
        if head and tail:
            head_best = max(head)
            tail_best = max(tail)
            if tail_best <= head_best + 1e-4:
                alerts.append(
                    f"Reward plateau: best_tail={tail_best:.4f}, best_head={head_best:.4f} (window={len(recent_rewards)})"
                )

    # 3) invalid draft ratio too high in latest rounds
    if st.invalid_round_latest:
        bad_rounds = []
        for r, x in sorted(st.invalid_round_latest.items()):
            if x["ratio"] > 0.25:
                bad_rounds.append(f"R{r}:{100.0*x['ratio']:.1f}%")
        if bad_rounds:
            alerts.append("Invalid-draft high: " + ", ".join(bad_rounds))

    # 4) step-time spike
    if len(recent_step_time) >= 5:
        base = sum(recent_step_time[:-1]) / (len(recent_step_time) - 1)
        latest = recent_step_time[-1]
        if base > 0 and latest > base * 1.25:
            alerts.append(f"Step-time spike: latest={latest:.1f}s > 1.25x recent_avg={base:.1f}s")

    # 5) rising-trend heuristics
    if len(recent_kl) >= 8:
        half = len(recent_kl) // 2
        early = sum(recent_kl[:half]) / max(1, half)
        late = sum(recent_kl[half:]) / max(1, len(recent_kl) - half)
        if late - early > 0.05:
            alerts.append(f"KL rising fast: late_mean={late:.4f}, early_mean={early:.4f}")
    if len(recent_grad) >= 8:
        half = len(recent_grad) // 2
        early = sum(recent_grad[:half]) / max(1, half)
        late = sum(recent_grad[half:]) / max(1, len(recent_grad) - half)
        if late - early > 0.7:
            alerts.append(f"Grad rising fast: late_mean={late:.3f}, early_mean={early:.3f}")

    return alerts


def render_dashboard(st: MonitorState, window: int) -> str:
    lines: List[str] = []
    lines.append("=" * 92)
    lines.append("RL Training Log Monitor")
    lines.append(f"Parsed step lines: {st.all_steps} | Rolling window: {window}")
    lines.append(
        f"Epoch/Step cursor: {st.last_epoch if st.last_epoch is not None else 'N/A'} / "
        f"{st.last_epoch_step if st.last_epoch_step is not None else 'N/A'}"
    )
    lines.append(
        f"Warnings: {st.warning_count} (parse-score warnings: {st.parse_score_warning_count}) | "
        f"Errors: {st.error_count}"
    )

    if st.invalid_total_count > 0:
        lines.append(
            f"Invalid draft overall: {st.invalid_total_seen}/{st.invalid_total_count} "
            f"({100.0 * st.invalid_total_seen / st.invalid_total_count:.2f}%)"
        )
    else:
        lines.append("Invalid draft overall: N/A")

    if st.invalid_round_latest:
        parts = []
        for r in sorted(st.invalid_round_latest.keys()):
            x = st.invalid_round_latest[r]
            parts.append(f"R{r}={x['bad']}/{x['total']} ({100.0 * x['ratio']:.2f}%)")
        lines.append("Latest invalid by round: " + " | ".join(parts))
    else:
        lines.append("Latest invalid by round: N/A")

    lines.append("-" * 92)
    alerts = build_alerts(st, window)
    if alerts:
        lines.append("ALERTS:")
        for a in alerts:
            lines.append(f"  - {a}")
    else:
        lines.append("ALERTS: none")
    lines.append("-" * 92)

    key_order = [
        "global_seqlen/mean",
        "actor/pg_loss",
        "actor/pg_clipfrac",
        "actor/ppo_kl",
        "actor/kl_loss",
        "actor/entropy_loss",
        "actor/grad_norm",
        "critic/score/mean",
        "critic/rewards/mean",
        "response_length/mean",
        "prompt_length/mean",
        "timing_s/gen",
        "timing_s/ref",
        "timing_s/adv",
        "timing_s/update_actor",
        "timing_s/step",
        "mfu/actor",
    ]

    if st.last_step is None:
        lines.append("No step metrics parsed yet.")
        lines.append("=" * 92)
        return "\n".join(lines)

    lines.append(f"Latest parsed step: {st.last_step.step}")
    lines.append("Metric                               latest        avg(window)")
    for k in key_order:
        latest = st.last_step.metrics.get(k)
        avgv = metric_avg(st.recent_steps, k)
        lines.append(f"{k:<36} {fmt(latest, 4):>12} {fmt(avgv, 4):>16}")
    lines.append("=" * 92)
    return "\n".join(lines)


def follow_file(log_path: str, st: MonitorState, window: int, refresh_sec: float) -> None:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        f.seek(0, os.SEEK_END)
        last_print = 0.0
        while True:
            where = f.tell()
            line = f.readline()
            if not line:
                time.sleep(0.2)
                f.seek(where)
            else:
                st.ingest_line(line)

            now = time.time()
            if now - last_print >= refresh_sec:
                os.system("clear")
                print(render_dashboard(st, window))
                last_print = now


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor RL training key metrics from log file.")
    parser.add_argument(
        "--log",
        type=str,
        default="/home/wangzixu/Search-R1/paper-writing-grpo-qwen2_5-3B-instruct-arxiv-writing.log",
        help="Path to training log file.",
    )
    parser.add_argument("--window", type=int, default=20, help="Rolling window size for averages.")
    parser.add_argument("--follow", action="store_true", help="Follow log in real time.")
    parser.add_argument(
        "--refresh-sec",
        type=float,
        default=3.0,
        help="Dashboard refresh interval in seconds when --follow is enabled.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.log):
        print(f"[ERROR] Log file not found: {args.log}", file=sys.stderr)
        return 1

    st = load_initial_state(args.log, args.window)
    if args.follow:
        try:
            follow_file(args.log, st, args.window, args.refresh_sec)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
    else:
        print(render_dashboard(st, args.window))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
