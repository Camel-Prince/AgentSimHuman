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
You will write or revise abstract drafts based on the input you receive.

## Input Format
You may receive previous drafts and reviewer comments in the following format:
- Previous draft: <draft>previous abstract content</draft>
- Reviewer comment: <comment>reviewer feedback</comment>

## Input Scenarios and Expected Actions

**Scenario 1: No previous draft or comment**
- Situation: You only see the writing task description
- Action: Write a complete initial abstract draft from scratch
- Output format: <draft>your abstract content</draft>

**Scenario 2: Previous comment exists (with <comment> tags), but no previous draft**
- Situation: You see a comment but no draft before it
- Action: Write a new abstract draft from scratch, taking the comment's suggestions into account
- Output format: <draft>your abstract content</draft>

**Scenario 3: Both previous draft (with <draft> tags) and comment (with <comment> tags) exist**
- Situation: You see both a previous draft and a comment about it
- Action: Revise the previous abstract draft based on the comment's feedback
- Output format: <draft>your revised abstract content</draft>

**Scenario 4: Generating final camera-ready version**
- Situation: You are asked to produce the final version after all revisions
- Action: Produce the final polished abstract incorporating all previous feedback
- Output format: <draft>your final abstract content</draft>

## Abstract Writing Guidelines
- Length: 150-250 words (approximately 200-350 tokens)
- Structure: Cover motivation, method, key results, and conclusion
- Use formal academic language appropriate for top-tier venues (e.g., NeurIPS, ICML, ICLR, ACL)
- Be concise and precise - every sentence should convey essential information
- Avoid citations, figures, or tables in the abstract
- Write in a self-contained manner that can be understood independently"""


def process_row(row):
    """Process a single row of data."""
    try:
        # Extract required fields
        domain = row['extra_info'].get('domain')
        topic = row['extra_info'].get('topic')

        # Skip if any required field is missing
        if not domain or not topic:
            print(f"Warning: Skipping row {row['data_source']} due to missing fields")
            return None

        # Construct user message - task is to write an abstract
        user_message = f"""## Writing Task

**Domain**: {domain}
**Topic**: {topic}

**Task Description**: Write an abstract for an academic paper about {topic} in {domain}. The abstract should be 150-250 words and cover: motivation, method, key results, and conclusion.

---

Please write your draft now."""

        # Construct new prompt (chat format)
        new_prompt = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            },
            {
                'role': 'user',
                'content': user_message
            }
        ]

        # Clean extra_info (keep domain, topic, and full query for reference)
        new_extra_info = {
            'domain': domain,
            'topic': topic,
            'query': SYSTEM_PROMPT + user_message
        }

        return {
            'data_source': row['data_source'],
            'ability': row['ability'],
            'prompt': new_prompt,
            'extra_info': new_extra_info,
            'reward_model': row['reward_model']
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
    
    # Define input and output files
    files_to_process = [
        ('train_prompts_chat.parquet', 'train_prompts_query_only.parquet'),
        ('val_prompts_chat.parquet', 'val_prompts_query_only.parquet')
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
