# TraceMem handoff

Updated 2026-10-15 UTC. Verify by: 2026-10-15. This checkpoint leads with the
longitudinal v2 targeted review slice. Read its [review card](longitudinal-v2-review.md),
[validation record](validation-longitudinal-v2-review.json), and
[copied synthetic artifacts](../examples/synthetic/reviews/longitudinal-v2/manifest.json).
The probe covers four explicitly selected cases and 10 tasks; it is not
representative. It had 9/9 decisive agreement with provisional authored labels,
zero human labels, and no Gate A approval. Source and installed replay parity and
focused tests passed; details and raw local receipts are in the validation record.

## Resume safely

Keep Context Kit, original conversations, historical receipts, and
`~/.llm-context` read-only. Keep private conversation data, vectors, claims, and
checkpoints local. Reviewed code, docs, and fictional fixtures may be signed
and pushed only after a fresh check confirms the repository is PRIVATE. No new
model-serving inference, training, Modal, or GPU work is needed for this slice;
the completed agent annotation is recorded separately. Recheck active jobs
before future runtime work; never duplicate a live call. Training still
requires separate approval.

`.local/active-cycle.json` is authoritative for signed delivery and exact-head
hosted CI. Verify its SHA, signature, and CI result before treating work as
delivered; read it alongside fresh git, process, worker, and remote-job state
before resuming operational work.

## Current phase and gate

Phase 0 inventory and Phase 1 infrastructure are complete; Phase 2 remains
synthetic diagnostics. No system model is selected or trained. Gate A remains unmet,
with zero independently verified human labels. The project README architecture
and plain-language explanation are complete.

## Longitudinal v2 review slice

The fresh fork-none Astra agent was shown only the probe blind JSON. Its labels
were 6 supported, 3 unsupported, and 1 uncertain; agreement is 9/9 only on the
decisive tasks with authored labels, excluding the unlabeled task. The prior
expectations were authored/reviewed by Luna, so the comparison does not establish
model-family or version independence. Root's unblinded audit agrees on all 10
labels and checked 13 exact evidence spans; it is not independent or human
ground truth. The full 26-task blind page is available locally at
`.local/longitudinal-v2-review-ready/reviewer/review.html`; share only its
`reviewer/` directory and keep the answer key in `curator/`. The next validation
step is human annotation of the full packet and a separate label/readiness
review. Preserve the frozen fixture bytes.

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

## Longitudinal v2 dataset details

The v2 data revision and fixture history remain in the
[longitudinal revision card](longitudinal-revision-v2.md). Its byte-identical
pre-compaction handoff is preserved in the
[2026-10-15 v2 handoff archive](handoff-history-2026-10-15-v2.md); this
checkpoint carries the current review status above.

## Next useful work

Proceed with human annotation of the full blinded longitudinal v2 packet, then
review those labels and dataset readiness criteria separately. Rights, scenario
adequacy, and independent validation remain open; do not infer Gate A progress
from this agent probe or other diagnostics. The frozen cross-session retrieval
slice remains documented in its [data card](cross-session-test-data-card.md),
[replay](cross-session-test-retrieval.json), and
[validation record](validation-cross-session-test.json). Check the documentation
index before repeating prior work.

## History

The [documentation index](README.md) links the detailed records and prior
validation history.
