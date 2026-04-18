#!/usr/bin/env python3
"""Clean SFT candidate JSONL files: remove redundant fields (e.g. `comments`).

Usage:
    # 一次性清理指定目录下所有 .jsonl 文件
    python3 scripts/clean_sft_candidates.py

    # 清理指定目录
    python3 scripts/clean_sft_candidates.py --dir outputs/sft_candidates

    # 以 watch 模式持续运行，每隔 N 秒扫描一次（配合正在运行的训练使用）
    python3 scripts/clean_sft_candidates.py --watch --interval 120

    # Dry-run：只打印会改动什么，不实际写入
    python3 scripts/clean_sft_candidates.py --dry-run
"""

import argparse
import json
import os
import time

# 需要从每条记录中删除的冗余字段
FIELDS_TO_REMOVE = ['comments']


def clean_file(path: str, dry_run: bool = False) -> int:
    """Remove redundant fields from a JSONL file in-place.

    Returns the number of records that were actually modified.
    """
    records = []
    modified = 0
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            changed = False
            for field in FIELDS_TO_REMOVE:
                if field in rec:
                    rec.pop(field)
                    changed = True
            if changed:
                modified += 1
            records.append(rec)

    if modified == 0:
        return 0

    if not dry_run:
        with open(path, 'w', encoding='utf-8') as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')

    return modified


def scan_and_clean(sft_dir: str, dry_run: bool = False) -> None:
    if not os.path.isdir(sft_dir):
        return
    for fname in os.listdir(sft_dir):
        if not fname.endswith('.jsonl'):
            continue
        path = os.path.join(sft_dir, fname)
        modified = clean_file(path, dry_run=dry_run)
        if modified > 0:
            tag = '[dry-run] ' if dry_run else ''
            print(f'{tag}Cleaned {modified} record(s) in {path}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Remove redundant fields from SFT candidate JSONL files.')
    parser.add_argument('--dir', default='outputs/sft_candidates',
                        help='Directory containing .jsonl files (default: outputs/sft_candidates)')
    parser.add_argument('--watch', action='store_true',
                        help='Keep running and re-scan periodically')
    parser.add_argument('--interval', type=int, default=120,
                        help='Seconds between scans in watch mode (default: 120)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would change without writing files')
    args = parser.parse_args()

    sft_dir = args.dir

    if args.watch:
        print(f'Watching {sft_dir} every {args.interval}s. Press Ctrl+C to stop.')
        try:
            while True:
                scan_and_clean(sft_dir, dry_run=args.dry_run)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print('Stopped.')
    else:
        scan_and_clean(sft_dir, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
