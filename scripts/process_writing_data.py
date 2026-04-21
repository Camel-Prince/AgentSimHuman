"""
Process paper writing data for multi-turn abstract revision experiment.

This script:
1. Cleans extra_info to keep only 'domain' and 'topic'
2. Constructs chat-format prompts with system and user messages
3. Task is simplified to writing an abstract (150-250 words) to avoid length truncation
"""

import pandas as pd
import sys
from pathlib import Path

SYSTEM_PROMPT = """You are an academic writing assistant specialized in writing research paper abstracts.

## Your Role
You will iteratively write and refine an abstract draft based on reviewer feedback.

## Interaction Format
- You submit a draft by outputting: <draft>your abstract content</draft>
- The system will respond with reviewer feedback: <comment>reviewer feedback</comment>
- You may submit multiple drafts to incorporate feedback and improve your writing.
- When you are satisfied with your draft, submit your final version by outputting: <camera-ready>your final abstract content</camera-ready>
- The interaction ends immediately after you output <camera-ready>.

## Decision Guidelines
- Start by writing a complete initial draft.
- After receiving <comment> feedback, decide whether to revise further (<draft>) or finalize (<camera-ready>).
- You are free to finalize at any point — there is no fixed number of rounds.
- Only output one tag block per turn. Do not mix <draft> and <camera-ready> in the same response.

## Abstract Writing Guidelines
- Length: 150-250 words (approximately 200-350 tokens)
- Structure: Cover motivation, method, key results, and conclusion
- Use formal academic language appropriate for top-tier venues (e.g., NeurIPS, ICML, ICLR, ACL)
- Be concise and precise — every sentence should convey essential information
- Avoid citations, figures, or tables in the abstract
- Write in a self-contained manner that can be understood independently"""


def process_row(row):
    """Process a single row of data.

    Replaces the system prompt in existing arxiv_writing_*.parquet rows.
    These rows use keywords/title (not domain/topic) in extra_info.
    The user message is kept as-is; only the system message is updated.
    """
    try:
        existing_prompt = row['prompt']

        # Find the existing user message content
        user_message = None
        for msg in existing_prompt:
            if msg['role'] == 'user':
                user_message = msg['content']
                break

        if user_message is None:
            print(f"Warning: Skipping row {row['data_source']} — no user message found")
            return None

        # Rebuild prompt with new system prompt, keeping original user message
        new_prompt = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_message},
        ]

        # Update extra_info: preserve all existing keys, update query
        new_extra_info = dict(row['extra_info'])
        new_extra_info['query'] = SYSTEM_PROMPT + user_message

        return {
            'data_source': row['data_source'],
            'ability': row['ability'],
            'prompt': new_prompt,
            'extra_info': new_extra_info,
            'reward_model': row['reward_model'],
        }

    except Exception as e:
        print(f"Error processing row {row.get('data_source', 'unknown')}: {e}")
        return None


def process_file(input_path, output_path):
    """Process a single parquet file."""
    print(f"\nProcessing: {input_path}")
    
    # Read input file
    df = pd.read_parquet(input_path)
    print(f"  Loaded {len(df)} rows")
    
    # Process each row
    processed_rows = []
    skipped_count = 0
    
    for idx, row in df.iterrows():
        result = process_row(row)
        if result is not None:
            processed_rows.append(result)
        else:
            skipped_count += 1
    
    # Create new dataframe
    new_df = pd.DataFrame(processed_rows)
    
    # Save to output file
    new_df.to_parquet(output_path, index=False)
    
    print(f"  Processed {len(new_df)} rows (skipped {skipped_count})")
    print(f"  Saved to: {output_path}")
    
    return len(new_df), skipped_count


def main():
    """Main processing function."""
    base_dir = Path('/home/wangzixu/Search-R1/data_paper_writing/processed')

    # In-place update of arxiv_writing_*.parquet files
    files_to_process = [
        ('arxiv_writing_train.parquet', 'arxiv_writing_train.parquet'),
        ('arxiv_writing_valid.parquet', 'arxiv_writing_valid.parquet'),
    ]
    
    total_processed = 0
    total_skipped = 0
    
    for input_file, output_file in files_to_process:
        input_path = base_dir / input_file
        output_path = base_dir / output_file
        
        if not input_path.exists():
            print(f"Warning: Input file not found: {input_path}")
            continue
        
        processed, skipped = process_file(str(input_path), str(output_path))
        total_processed += processed
        total_skipped += skipped
    
    print(f"\n{'='*60}")
    print(f"Total processed: {total_processed} rows")
    print(f"Total skipped: {total_skipped} rows")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
