---
name: update-project-status
description: "Use when summarizing a session and updating PROJECT_STATUS.md with important changes, decisions, active experiment state, new constraints, or new files. Triggers: update project status, summarize session, handoff, record changes, keep PROJECT_STATUS current, end of session."
argument-hint: "Describe what was done this session, or leave empty to auto-detect from session memory and recent file changes."
---

# Update Project Status

Compact the important outcomes of the current session into PROJECT_STATUS.md. Do NOT write a diary. Replace stale bullets; add only what is new and durable.

## When to Use

- At the end of a working session before switching context
- After code changes that affect architecture, data flow, or constraints
- After design decisions that future agents must respect
- After adding new scripts, files, or skills to the workspace
- After a running experiment reports early-phase metrics

## Procedure

### Step 1 — Gather session context

Read session memory if it exists:
- `/memories/session/plan.md` or any file under `/memories/session/`

Then read the current PROJECT_STATUS.md in full to understand what is already there.

### Step 2 — Identify what changed this session

Categorize each change into one of:

| Category | Goes Into |
|---|---|
| New or moved key file / script | Workspace Map |
| Code path added or behavior changed | Critical Code Paths |
| Algorithm rule, constraint, or "do not" | Algorithm Notes or Do Not Forget |
| Active run metrics / experiment state | Active Run And Monitoring |
| High-level focus shift | Current Focus |

Discard: timestamps, full log excerpts, chat history, API keys, W&B run IDs.

### Step 3 — Decide what to write per section

Consult [section-rules.md](./references/section-rules.md) for per-section update rules.

The core rule: **replace stale bullets, do not append history.**

- If an existing bullet is now outdated, replace it.
- If new information belongs to an existing bullet's topic, extend that bullet.
- Add a new bullet only when the topic is genuinely absent.
- Never add more than 3 new bullets per session without removing an equal number of stale ones.

### Step 4 — Apply edits

Use `multi_replace_string_in_file` for all edits in a single call when possible.

Include ≥ 3 lines of surrounding context in each `oldString` so replacements are unambiguous.

Do NOT rewrite entire sections. Make surgical replacements.

### Step 5 — Verify

After writing, re-read the edited sections and confirm:
- No secrets, API keys, or private URLs were written
- No full log output or experiment transcript was written
- Every new bullet is ≤ 2 lines
- The file still reads as a current-state map, not a changelog
- `name` field in this SKILL.md still matches the folder name (`update-project-status`)
