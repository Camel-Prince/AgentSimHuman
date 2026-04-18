#!/usr/bin/env python3
"""
Enhanced GRPO training log analyzer - extracts key metrics to verify training progress.

Usage:
    python scripts/analyze_grpo_training.py <log_file> [--plot] [--recent N]

Examples:
    # Basic analysis
    python scripts/analyze_grpo_training.py paper-writing-grpo-qwen2_5-7b-instruct-debug.log

    # With plots
    python scripts/analyze_grpo_training.py paper-writing-grpo-qwen2_5-7b-instruct-debug.log --plot

    # Show only recent N steps
    python scripts/analyze_grpo_training.py paper-writing-grpo-qwen2_5-7b-instruct-debug.log --recent 10
"""

import re
import sys
import argparse
from collections import defaultdict
import numpy as np

def parse_grpo_log(log_file):
    """Parse GRPO training log and extract all key metrics."""
    metrics = defaultdict(list)
    steps = []
    current_step = None

    with open(log_file, 'r') as f:
        for line in f:
            # Extract step/epoch info
            step_match = re.search(r'epoch\s+(\d+),\s+step\s+(\d+)', line)
