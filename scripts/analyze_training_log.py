#!/usr/bin/env python3
"""
Analyze training log file and extract key metrics.

Usage:
    python scripts/analyze_training_log.py paper-writing-grpo-qwen2_5-7b-instruct-debug.log
"""

import re
import sys
from collections import defaultdict

def parse_log(log_file):
    """Parse training log and extract metrics."""

    metrics = defaultdict(list)

    with open(log_file, 'r') as f:
        for line in f:
            # Extract reward scores
            if 'reward' in line.lower() or 'score' in line.lower():
                # Pattern: reward: 0.75, score: 0.80, etc.
                matches = re.findall(r'(reward|score):\s*([0-9.]+)', line, re.IGNORECASE)
                for metric_name, value in matches:
                    metrics[metric_name.lower()].append(float(value))

            # Extract loss values
            if 'loss' in line.lower():
                matches = re.findall(r'loss:\s*([0-9.]+)', line, re.IGNORECASE)
                for value in matches:
                    metrics['loss'].append(float(value))

            # Extract KL divergence
            if 'kl' in line.lower():
                matches = re.findall(r'kl:\s*([0-9.]+)', line, re.IGNORECASE)
                for value in matches:
                    metrics['kl'].append(float(value))

            # Extract learning rate
            if 'lr' in line.lower() or 'learning_rate' in line.lower():
                matches = re.findall(r'(?:lr|learning_rate):\s*([0-9.e-]+)', line, re.IGNORECASE)
                for value in matches:
                    metrics['lr'].append(float(value))

            # Extract step/epoch info
            if 'step' in line.lower() or 'epoch' in line.lower():
                step_match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
                if step_match:
                    metrics['step'].append(int(step_match.group(1)))

                epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
                if epoch_match:
                    metrics['epoch'].append(int(epoch_match.group(1)))

    return metrics

def print_summary(metrics):
    """Print summary statistics."""

    print("\n" + "="*60)
    print("Training Log Summary")
    print("="*60)

    for metric_name, values in sorted(metrics.items()):
        if not values:
            continue

        if metric_name in ['step', 'epoch']:
            print(f"\n{metric_name.upper()}:")
            print(f"  Current: {values[-1] if values else 'N/A'}")
            print(f"  Total: {len(values)} records")
        else:
            print(f"\n{metric_name.upper()}:")
            print(f"  Latest: {values[-1]:.6f}")
            print(f"  Mean: {sum(values)/len(values):.6f}")
            print(f"  Min: {min(values):.6f}")
            print(f"  Max: {max(values):.6f}")
            print(f"  Total records: {len(values)}")

    print("\n" + "="*60)

def plot_metrics(metrics):
    """Plot metrics over time (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training Metrics')

        plot_configs = [
            ('reward', 'Reward', axes[0, 0]),
            ('loss', 'Loss', axes[0, 1]),
            ('kl', 'KL Divergence', axes[1, 0]),
            ('lr', 'Learning Rate', axes[1, 1])
        ]

        for metric_name, title, ax in plot_configs:
            if metric_name in metrics and metrics[metric_name]:
                ax.plot(metrics[metric_name])
                ax.set_title(title)
                ax.set_xlabel('Step')
                ax.set_ylabel(title)
                ax.grid(True)
            else:
                ax.text(0.5, 0.5, f'No {title} data',
                       ha='center', va='center', transform=ax.transAxes)

        plt.tight_layout()
        output_file = 'training_metrics.png'
        plt.savefig(output_file)
        print(f"\nPlot saved to: {output_file}")

    except ImportError:
        print("\nNote: Install matplotlib to generate plots: pip install matplotlib")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_training_log.py <log_file>")
        sys.exit(1)

    log_file = sys.argv[1]

    print(f"Analyzing log file: {log_file}")
    metrics = parse_log(log_file)
    print_summary(metrics)
    plot_metrics(metrics)
