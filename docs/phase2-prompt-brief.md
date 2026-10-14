# Follow-up: explicit prompt baseline

The initial two synthetic calls are complete. Extraction exposed schema, offset,
attribution and temporal failures. Some failures reflect underspecified prompt
semantics, especially the meaning of speaker role and date bounds. Improve the
prompt-only comparator before treating the result as evidence for training.

Keep the original prompt, payloads, receipts and metrics unchanged. Add a separate
`explicit-v2` renderer with general schema and memory-policy guidance; do not
insert the diagnostic cases' expected claims into its instructions. The same 24
cases are development diagnostics, not an independent evaluation of this revision.

One further synthetic extraction call is within the continuing inexpensive
inference scope and existing $2 operational target. Preserve the original single
T4 container, 900-second execution and 300-second startup limits, no retries,
source mounts, warm pool, training or private data. No provider billing cap exists.

File ownership:

- Evidence worker: `prompts.py` and its tests, at most 180 production lines.
- Compute worker: integrate an explicit prompt-version option into payload building,
  local evaluation and anchoring evaluation after its current slice; preserve v1
  hashes and default behavior. At most 150 added production lines, no refactor.
- Root: inspect both prompts and all changes, launch, compare raw and anchored
  results, audit semantic failures and update measured evidence.

Acceptance: v1 hashes remain pinned, each evaluator rejects the other version's
receipt, prompts omit expected labels, actual generation stays within the same
resource envelope, and the report distinguishes prompt development, schema,
span integrity and semantic support. Neither variant auto-accepts claims.
