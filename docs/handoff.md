# TraceMem handoff

Updated 2026-10-15 UTC. Verify by: 2026-10-15. This checkpoint leads with the
frozen cross-session turn-retrieval diagnostic. Its fixture SHA-256 is
`dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf`. Read the
[data card](cross-session-test-data-card.md), [frozen replay](cross-session-test-retrieval.json),
and [validation record](validation-cross-session-test.json) for scope, measured
results, limits, and receipts.

## Resume safely

Keep Context Kit, original conversations, historical receipts, and
`~/.llm-context` read-only. Keep private conversation data, vectors, claims, and
checkpoints local. Reviewed code, docs, and fictional fixtures may be signed
and pushed only after a fresh check confirms the repository is PRIVATE. Do not
launch inference, training, Modal, or GPU work for this slice. Recheck active
jobs before future runtime work; never duplicate a live call. Training still
requires separate approval.

`.local/active-cycle.json` is authoritative for signed delivery and exact-head
hosted CI. Verify its SHA, signature, and CI result before treating work as
delivered; read it alongside fresh git, process, worker, and remote-job state
before resuming operational work.

## Current phase and gate

Phase 0 inventory and Phase 1 infrastructure are complete; Phase 2 remains
synthetic diagnostics. No model is selected or trained. Gate A remains unmet,
with zero independently verified human labels. The project README architecture
and plain-language explanation are complete.

## Cross-session test slice

The six fictional cases contain 72 turns across 24 sessions. Four answerable
questions contribute eight provisional evidence-turn labels; two `UNKNOWN`
cases are excluded from turn-label recall. The unblinded, agent-only root review
provisionally supported the four answerable cases and retained both unknowns.
It is not independent validation, and the assigned `test` split is bookkeeping,
not held-out evidence. No QA or abstention accuracy was measured.

The source and installed focused runs each passed three tests. Their replay
reports matched byte-for-byte; all 37 runtime package files matched the pinned
wheel and installed environment, and all 50 historical artifact pins matched.
The final local `make check` passed 602 of 603 tests with one optional skip;
Ruff, formatting across 134 files, mypy across 32 source files, actionlint, and
`git diff --check` passed. The validation record records that 54 retrieval
cells were audited and points to the raw receipts; the replay report contains
the per-question cells. Local validation does not establish signed
delivery or hosted CI; use `.local/active-cycle.json` for those states.

## Prior longitudinal v2 context

The prior v2 data revision, review status, and limits remain in the
[longitudinal revision card](longitudinal-revision-v2.md). Its byte-identical
pre-compaction handoff is preserved in the
[2026-10-15 v2 handoff archive](handoff-history-2026-10-15-v2.md). This current
checkpoint supersedes that handoff for the active cross-session slice.

## Next useful work

After exact-head CI is verified, pursue distinct license-clean scenarios and
independent annotation. Rights, scenario adequacy, and independent validation
remain open; do not infer Gate A progress from these diagnostics. Check the
documentation index for detailed architecture, experiment receipts, and
validation records before repeating earlier work.

## History

The [documentation index](README.md) links the detailed records and prior
validation history.
