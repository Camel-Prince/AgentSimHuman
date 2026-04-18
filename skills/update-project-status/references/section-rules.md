# Section Update Rules

Rules for each section in PROJECT_STATUS.md. Apply these during Step 3 of the skill procedure.

---

## Current Focus

**Update when**: the main task area shifts, a new preferred rollout mode is adopted, or the evaluation strategy changes.

**Replace** the stale bullet describing the old focus. Do not keep both old and new side by side.

**Do not add**: experiment run names, timestamps, or intermediate states.

---

## Workspace Map

**Update when**: a new file is created that future agents need to know about, a file is moved or renamed, or a file's role changes significantly.

**Format**: one bullet per file/directory, starting with the path in backticks, followed by a colon and a one-sentence role description.

**Replace** the bullet for a file if its role description is now stale.

**Add** a new bullet when a new file is introduced that is load-bearing for the project workflow.

**Do not add**: temp files, output directories that need no explanation, log files.

---

## Critical Code Paths

**Update when**: a function is added, removed, renamed, or its behavior/contract changes in a way that affects downstream reasoning.

**Format**: one bullet per function name (in backticks), followed by a colon and its current contract summary.

**Replace** the bullet when behavior changes (e.g., a field is no longer written, a mask rule changes).

**Do not add**: internal helper functions that are not entry points, functions with no cross-component effect.

---

## Algorithm Notes

**Update when**: a new algorithm rule is confirmed (not hypothesized), a previous note turns out to be wrong, or a new constraint is discovered from debugging.

**Replace** wrong notes immediately; do not leave contradictory bullets.

**Add** a new bullet only when the insight is stable and not implied by existing bullets.

**Do not add**: hunches, TODOs, or experiment hypotheses — those belong in session memory, not here.

---

## Active Run And Monitoring

**Update when**: a new experiment is started, early-phase metrics are observed, or a monitoring pattern proves important.

**Format for active run**: one bullet starting with `Active run (YYYY-MM-DD):` followed by task type, model, hardware, total steps, and a 1-line metric summary.

**Replace** the previous active run bullet when a new run supersedes it.

**Metric summary format**: list 3–5 key metrics with their step-1 values and a 1-word trend (e.g., `score 0.61→0.61, KL ~0.001 stable, entropy ~0.12 decreasing`).

**Do not add**: full metric tables, per-step breakdowns, or rubric subscores — those belong in wandb or log files.

---

## Do Not Forget

**Update when**: a new hard constraint is discovered (e.g., a field must not be re-added, a code pattern breaks something), or when a team member might plausibly make the same mistake again.

**Format**: one bullet per constraint, starting with "Do not...".

**Add** sparingly — only when the constraint is non-obvious and has been violated or nearly violated.

**Replace** a bullet if it is now handled by code enforcement (e.g., a removed field is now guarded by a test).

**Do not add**: reminders that duplicate what is already obvious from the code.
