---
name: git-commit-and-push
description: "Use when you want to commit and push recent changes to GitHub. Automatically summarizes what changed into a short commit message, then runs git add ., git commit, and git push. Triggers: commit, push, sync to github, save changes, git commit and push."
argument-hint: "Optionally describe what you changed. If omitted, the skill will infer the message from git diff."
---

# Git Commit and Push

Summarize recent changes into a concise commit message, then stage, commit, and push to origin.

## Procedure

### Step 1 — Check working tree status

Run:
```bash
git -C /home/wangzixu/Search-R1 status --short
```

If the output is empty ("nothing to commit"), stop and tell the user there is nothing to push.

### Step 2 — Inspect the diff to understand what changed

Run:
```bash
git -C /home/wangzixu/Search-R1 diff HEAD --stat
git -C /home/wangzixu/Search-R1 diff HEAD -- '*.py' '*.sh' '*.yaml' '*.md' | head -200
```

Focus on:
- Which files changed (from `--stat`)
- The nature of each change (new feature, fix, config tweak, data update)

Do NOT read binary files or large generated files.

### Step 3 — Draft the commit message

Rules:
- **One subject line**, ≤ 72 characters, written in imperative mood ("Add X", "Fix Y", "Update Z")
- If there are 2–4 logically distinct changes, append a blank line + short bullet list (each ≤ 60 chars, no period)
- Do NOT mention file names alone — describe intent/behavior
- Do NOT use vague words like "misc changes", "update files", "various fixes"

If the user provided a description as an argument, use it verbatim or refine it slightly to match the format above.

Examples of good messages:
```
Add git-commit-and-push skill

- Auto-generates commit message from diff
- Follows imperative style with optional bullet body
```
```
Fix GRPO advantage normalization for state masking
```
```
Update paper writing reward to weight arena score 0.7
```

### Step 4 — Stage, commit, and push

Run these commands sequentially, waiting for each to succeed:

```bash
git -C /home/wangzixu/Search-R1 add .
git -C /home/wangzixu/Search-R1 commit -m "<subject line>" -m "<optional bullet body>"
git -C /home/wangzixu/Search-R1 push
```

If `git push` fails with "rejected" or "non-fast-forward", do NOT force-push. Tell the user to resolve divergence manually (`git pull --rebase` first).

### Step 5 — Confirm

After a successful push, report:
- The commit hash (short) and message
- How many files changed, insertions, deletions (from commit output)
- The remote branch that was updated

## Safety Rules

- Never use `--force` or `--force-with-lease` without explicit user confirmation
- Never amend already-pushed commits
- Never commit secrets, tokens, API keys, or credential files — if any appear in the diff, abort and warn the user
- Skip binary files (`.pt`, `.ckpt`, `.parquet`) — they should already be in `.gitignore`
