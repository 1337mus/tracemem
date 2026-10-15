# Current checkpoint: first trained adapter, negative quality result

Updated 2026-10-15 UTC. Verify live jobs before resuming.

Raj explicitly approved the prepared training pilot. The single guarded call
completed on personal Modal, using fictional data only. It trained Qwen3-4B
with QLoRA for 32 optimizer steps on 128 examples and saved a 132,187,888-byte
adapter plus tokenizer in the private Volume `tracemem-sft-pilot`, under
`adapters/pilot-20261015-01`. No weights were downloaded to this Mac.

Training took 426.58 seconds; the worker took 768.13 seconds and the full client
call 855.38 seconds. All losses were finite, all 32 steps completed, and adapter
parameter hashes changed. Estimated worker execution resources cost about
$0.33 before build/startup/storage; actual billing was not retrieved.

The matched eight-prompt development result is negative for exact extraction:
zero exact claim matches before and after training. Parse failures fell from
two to zero; schema failures remained one; truncations fell from two to zero;
empty-target abstentions increased from 1/2 to 2/2. Raw evidence integrity was
0/9 base claims and 0/10 tuned claims. The existing unique-quote matcher repaired
base integrity to 9/9 and tuned integrity to 8/10, but exact matches stayed zero.
It repairs offsets only and never establishes semantic support. Attribution,
temporal bounds, missed negation/intention, and false ownership remain failures.

See [pilot results](sft-pilot-results.md), [aggregate results](sft-pilot-results.json),
and [training validation](validation-sft-pilot-training.json). Exact fictional
job/receipt bytes are preserved in `examples/synthetic/sft-pilot-{job,remote}-v1.json`.
The local scoring CLI replays the same report bytes without a GPU. Raw local
receipts are unchanged under `.local/sft-pilot-20261015/`.

The final local `make check` passed: 702 tests passed, one optional skip;
Ruff, formatting across 155 files, and strict mypy across 41 source files passed.
Source and report changes are being prepared for signed private delivery;
`.local/active-cycle.json` is authoritative for actual commit, push, and CI state.
Do not infer hosted CI from these local checks.

Personal inventory at 2026-10-15T21:15:34Z showed the app stopped with zero tasks.
No training client remains active. Recheck before another job. The completed app
was `ap-ptp7tOd8DmaNrRsgcaqnI3`; do not reuse run ID `pilot-20261015-01`.
The exact job approval hash was
`99cbe61613373a487eda3894e862b47dd72f9f64e3dc212bf3e4ca88ec5a4155`;
owner authorization is recorded in `.local/sft-pilot-20261015/training-owner-authorization-v1.json`.

Next research work should target the observed attribution, temporal and ownership
errors, compare candidate base checkpoints, and strengthen distinct training
scenarios and independent annotation. Preserve the original dev/test split;
any use of these development failures to revise examples must be disclosed.
More training steps alone are not established as the remedy. Gate A remains
unmet, labels are provisional agent-authored, zero human reviewers contributed,
and the private test split was not used. No production model is selected.

Context Kit, original conversations, historical receipts and `~/.llm-context`
remain read-only. Private data and checkpoints must not enter GitHub. Only
reviewed code, documentation and fictional fixtures may be signed and pushed
to `1337mus/tracemem` after a fresh PRIVATE check. Publication remains unauthorized.
Continuation automation stays paused. Computer-use tools were unused.

## Earlier checkpoint and history

# TraceMem handoff

Updated 2026-10-15 UTC. Verify by: 2026-10-15. The current diagnostic is the
[cross-session MiniLM comparison](cross-session-minilm.md), its replay report
([cross-session-minilm-retrieval.json](cross-session-minilm-retrieval.json)),
and [local validation record](validation-cross-session-minilm.json). Source and
isolated installed checks passed; signed/private delivery and exact-head hosted
CI remain governed by `.local/active-cycle.json`. The comparison uses the
existing six fictional cases and does not establish held-out quality or answer
accuracy. The earlier checkpoint below remains the
[scoped defaults/exceptions data card](scope-exceptions-data-card.md), its
[source replay](scope-exceptions-report.json), [unblinded semantic audit](scope-exceptions-semantic-audit.json),
and [validation record](validation-scope-exceptions.json). The six fictional
cases cover branch scope, attribution, dated exceptions, canceled proposals,
and prerequisites. Root's local `make check` passed 617 tests with one optional
skip; source and installed focused runs each passed 5 tests. These are local
results; verify signed delivery and exact-head hosted CI in
`.local/active-cycle.json`. The combined audit found no lexical duplicates or
declared cross-split identifier overlaps; semantic isolation and independence are
not established. Gate A remains unmet.

The prior [dataset isolation audit](dataset-isolation.md), its [report](dataset-isolation-report.json),
and [validation record](validation-dataset-isolation.json) remain useful. The CLI now audits
multiple claim fixture files together and reports declared scenario, template,
and entity identifier reuse across splits. Its original no-argument output is
byte-identical. Metadata is optional, with explicit missing-case coverage.

The 84-case pool has metadata for 60 cases and lacks it for 24 diagnostic cases.
No declared overlaps or lexical duplicate pairs were found. These literal IDs
cannot prove semantic isolation. Assigned test labels are bookkeeping, not held-out
quality. Gate A remains unmet, with zero human labels and no model selected or trained.
Source validation passed 612 tests with one optional skip; a fresh installed wheel
passed all 20 focused audit tests and produced identical report bytes. Runtime
package files matched source, wheel, and installation. No GPU or inference ran.

The prior [longitudinal v2 review](longitudinal-v2-review.md) remains provisional
agent-only evidence. The full 26-task human review packet is still available.

## Resume safely

Keep Context Kit, original conversations, historical receipts, and
`~/.llm-context` read-only. Keep private conversation data, vectors, claims, and
checkpoints local. Reviewed code, docs, and fictional fixtures may be signed
and pushed only after a fresh check confirms the repository is PRIVATE. Exactly
one reviewed synthetic-only embedding call completed for this slice. The
postcall inventory at 2026-10-15T09:16:23Z showed the worker stopped and zero
tasks. No further runtime work is needed for this slice. Recheck active jobs
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

## Cross-session MiniLM comparison

The new [comparison note](cross-session-minilm.md) records the remote MiniLM
embedding receipt and compares its vectors with the existing BM25, hash-vector,
and hybrid methods on the same 72 turns and six questions. MiniLM reaches 0.875
macro labeled-turn recall and 3/4 complete coverage at k=3; BM25 and both hybrid
methods reach 1.0 and 4/4. All methods reach complete coverage at k=5. These
provisional labels are agent-only, with zero human ground truth; the assigned
`test` split is bookkeeping. The two `UNKNOWN` cases are excluded from recall,
and no QA or abstention accuracy was measured. Local source and isolated
installed validation passed; see the [validation record](validation-cross-session-minilm.json)
for full counts, parity, the independent ranking audit, and preserved setup
corrections. Signed/private delivery and exact-head hosted CI remain unverified;
`.local/active-cycle.json` is authoritative. The raw receipt is
[`minilm-cross-session-remote.json`](../examples/synthetic/minilm-cross-session-remote.json).

## Longitudinal v2 dataset details

The v2 data revision and fixture history remain in the
[longitudinal revision card](longitudinal-revision-v2.md). Its byte-identical
pre-compaction handoff is preserved in the
[2026-10-15 v2 handoff archive](handoff-history-2026-10-15-v2.md); this
checkpoint carries the current review status above.

## Next useful work

Proceed with human review of the full longitudinal v2 packet and the scoped
defaults/exceptions packet, then review labels and dataset readiness separately.
Do not repeat this embedding call or create another near-duplicate retrieval
slice. Keep rights and scenario adequacy open; do not infer Gate A progress from
agent reviews or other diagnostics.
The original CPU cross-session retrieval slice
remains documented in its [data card](cross-session-test-data-card.md),
[replay](cross-session-test-retrieval.json), and
[validation record](validation-cross-session-test.json). Check the documentation
index before repeating prior work.

## History

The [documentation index](README.md) links the detailed records and prior
validation history.
