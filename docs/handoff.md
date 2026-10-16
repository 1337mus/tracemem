# Current checkpoint: quote-v4 complete; criteria failed

Updated 2026-10-16 UTC. The [fourth experiment report](sft-quote-v4-results.md)
records a negative result against its frozen success criteria. The test fixture
is spent; the development fixture remains an adaptive diagnostic and cannot
establish held-out quality. No fifth target-tweak SFT is planned. No production
model was selected, and Gate A remains unmet.

The run trained 192 steps over 768 fictional rows from 72 correlated families
(696 claims, 140 empty rows); seven reviewed family corrections affect 64 rows.
Training took 1,147.795 seconds; worker and client elapsed times were 2,945.737
and 2,963.787 seconds. The reported device was NVIDIA A10; the request was
A10G. The adapter changed, was saved in the separate personal Modal volume,
and was not downloaded to this Mac. The prior adapter hashes matched before
and after comparison; it was unloaded without merge, RNG state was restored,
and no prior LoRA parameters remained. The recorded post-run inventory had
zero active tasks. Verify `.local/active-cycle.json` and live Modal state before
any future call. Verify by: 2026-10-23.

One fresh-context Luna reviewer assessed 108 shuffled tasks covering 332 claims
across all 12 development and 24 test cases and all three arms. Root completed
an independent first pass and unblinded adjudication; the final counts use
root's labels, with changes from the blind review kept separately. The reviewer
and root are from the same model family; there were zero human reviewers. The
labels are provisional. They do not establish independent truth, broad
generalization, QA benefit, or model selection. Blind-review misses included
treating a literal base-arm Lumen quote as an actor swap and missing an adapter
quilt actor swap. The report preserves these limits and further examples.

The job, receipt, structural report, semantic summary, coverage, adjudicated
review, and adjudication files are linked from the results report. Only fiction
entered training and inference. No personal conversation data or derived data
was uploaded; Context Kit and original conversations remain read-only. Billing
was not measured. The exact-tip source CI run tested
`a0e82729fa165eef5b1a304dbf38a770ca6c0361`; it passed 809 source tests (one
skipped) and 399 installed tests (two skipped). It predates these result-doc
changes and does not verify the current documentation. This local update is
not committed or pushed.

The dev-only prompt baseline is implemented and reviewed. The Qwen3-4B call
is running on personal Modal, app `ap-BEYr1cfiI5nDhnqxFCFQLi`, client session
56291. Do not duplicate it. Qwen3-8B has not launched and must wait for this
call to finish. Exact prepared payloads, runner hashes, client log and receipt
paths are in `.local/quote-baseline-run-20261016/` and `.local/active-cycle.json`.
The full source check passed 854 tests with one optional skip; lint and
configured type checks passed. A subsequent typing-only helper fix passed its
25 focused tests and preserved summary bytes; extra strict checking of all
three new source modules passed. Signed delivery is pending. See the
[baseline protocol](quote-baseline-protocol.md).
Use fresh test cases and human review before another quality claim. Automation
remains paused. Verify baseline and remote state by: 2026-10-23.

## Historical checkpoint: quote-v4 training in progress

This launch snapshot is retained for chronology; its running status is
superseded by the completed-run record above. On 2026-10-16, one guarded call
was running on personal Modal under app `ap-jOI0vmcXqoCG6Fpj9A5Gej`, run
`quote-v4-20261016-01`, client session 50311. The corpus had seven corrected
families affecting 64 of 768 rows, 696 claims, and 140 empty rows across 72
families. The 36 evaluation cases had been frozen and reviewed: 12 dev and 24
test, with zero human reviewers. The prepared job hash was
`9bb99c82d119ae356d53b5576d0345ec43d1b5950823954f7e33f566b61ab59a`.
The signed preparation commits were pushed to private main at
`a0e82729fa165eef5b1a304dbf38a770ca6c0361`; the exact-tip CI result appears
above. The launch note correctly said not to start a duplicate; the completion
receipt now records that this call finished.

## Previous checkpoint: quote-v3 results adjudicated

Updated 2026-10-16 UTC. V3 training, structural scoring, blind test review, and
root adjudication are complete. The result is mixed: adverse test responses
declined, but required-fact coverage also fell and the predeclared promising
criteria were not met. There are zero human reviewers, no product selection,
and no Gate A progress. See the [results report](sft-quote-v3-results.md) and
[validation record](validation-sft-quote-v3.json).

The semantic review covered prior and tuned outputs on 48 frozen test cases
(96 tasks, 77 claims). There was no full semantic review of the base arm, dev
cohort, or adaptive regression cohort. Root's two-case spot-check found the
original v2 D14 chronology error and D21 subject ambiguity remain; it was not
a full regression review. The test also contains structural similarities and
ambiguous cases described in the report. Frozen synthetic data and receipts
remain unchanged.

Next research should target omitted required facts and explicit actor/report
attribution, with stronger annotations and longer independent conversations.
This frozen test is spent and must not be retuned against.

The recorded post-run inventory shows the v3 app stopped with zero tasks.
Before any future remote call, verify `.local/active-cycle.json` and live
Modal state. Successful hosted CI run 38828735844 tested source commit
a396d48da65b059bde55d0088e5996ec689cf300; it did not test the final
documentation working-tree tip. The final doc tip still needs its hosted CI
check after delivery.

## Older checkpoints and history

Everything below is historical context through 2026-10-15. Use the current
quote-v4 checkpoint and links above for current status; the detailed older
project snapshot is also preserved in the [2026-10-15 v2 handoff
archive](handoff-history-2026-10-15-v2.md).

### Historical checkpoint: quote-v2 completed


Updated 2026-10-15 UTC. The second synthetic-only QLoRA run and its frozen
development scoring, blind agent review, and root adjudication are complete.
The new [results report](sft-quote-v2-results.md), [aggregate JSON](sft-quote-v2-results.json),
and [validation record](validation-sft-quote-v2.json) keep strict raw scoring,
fence-only normalization, and semantic labels separate.

The run used 768 rows from 48 families with 16 correlated variants each and
192 optimizer steps; the 24 development families are a small adaptive slice,
not held out. Strict scoring found 23/24 base parse failures from Markdown
fences versus 0/24 tuned, intact citations 1/1 versus 19/19, reference-string
matches 0/19 versus 2/19, and correct empty-source abstentions 0/6 versus 6/6.
The separate fence-only analysis changes base presentation metrics only; it
does not establish meaning or support.

One fresh-context blinded agent read all 48 readable raw responses and judged
45 claims. The packet builder's strict parser mechanically marked the 23
fenced base tasks unreviewable and left their generated-claim arrays empty, but
the reviewer read every raw response, including fenced JSON. Root's later
unblinded review covered all sources and all 45 judgments and accepted every
label. These are agent-only provisional labels: zero human reviewers, visible
formatting differences may reveal the arms, and no recall or generalization
claim follows. D14 has unsupported invented chronology; D21's keeper/subject
attachment remains uncertain; D01 and D03 show better cause or attribution
handling. Gate A is unmet and no production checkpoint was selected.

The adapter (132,187,888 bytes; SHA-256
`3b30cbf98a685792734d8c16ffd7f31e000707f8f2b72a74e5ca9bc364214093`) remains
in the personal private Modal volume at
`tracemem-sft-quote-v2/adapters/quote-v2-20261015-01/adapter`; no weights were
downloaded to this Mac. Training took 901.998 seconds, the worker 1,169.920
seconds, and the client 1,184.560 seconds. The approximate `$0.50` worker
resource estimate excludes startup, build, and storage; billing was not
available. The recorded 2026-10-15 23:01 UTC inventory showed both apps stopped
with zero active tasks. Verify live operational state by 2026-10-15.

The preserved local source check attempted 734 tests, passed 733, and skipped
one optional test; Ruff covered 161 formatted files and mypy checked 44 source
files. It predates this documentation closeout. Check
`.local/active-cycle.json` for the current signed tip, push, and hosted-CI
state; this handoff makes no delivery claim for the new results documents.
No new training or model-runtime calls were needed; CPU replay and exact-copy
checks passed.

Next research work is paired chronology and subject-attachment corrections
with distinct scenario families and a separate frozen test, alongside stronger
annotations. Do not assume additional optimizer steps are the remedy. The
historical checkpoints below are retained for context; use the current status
above and `.local/active-cycle.json` for live state. Original `.local` receipts
remain unchanged.

### Earlier historical records

#### Historical checkpoint: first trained adapter, negative quality result

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
Source, reports, and fictional receipts were pushed to the private repository
in commit `a2c2439aa9d5022047e7e412d56b17538c11e2da`. Its GPG signature
verifies locally; GitHub reports `bad_email` because the commit email differs
from the key UID. A documentation follow-up uses the existing verified signing
identity without changing the key, account, or earlier history.
`.local/active-cycle.json` is authoritative for the latest signed tip, push,
GitHub signature verification, and exact-head CI state.
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

#### Older project details

##### TraceMem handoff (historical)

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

#### Historical resume notes (2026-10-15)

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

#### Historical project status (as of 2026-10-15)

Phase 0 inventory and Phase 1 infrastructure are complete; Phase 2 remains
synthetic diagnostics. No system model is selected or trained. Gate A remains unmet,
with zero independently verified human labels. The project README architecture
and plain-language explanation are complete.

#### Longitudinal v2 review slice

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

#### Cross-session test slice

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

#### Cross-session MiniLM comparison

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

#### Longitudinal v2 dataset details

The v2 data revision and fixture history remain in the
[longitudinal revision card](longitudinal-revision-v2.md). Its byte-identical
pre-compaction handoff is preserved in the
[2026-10-15 v2 handoff archive](handoff-history-2026-10-15-v2.md); this
checkpoint carries the current review status above.

#### Historical next work (superseded)

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

#### Archive links

The [documentation index](README.md) links the detailed records and prior
validation history.
