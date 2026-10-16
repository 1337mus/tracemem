# Current checkpoint: first private comparison completed

Updated 2026-10-16 UTC. One personal Modal A10G call compared the pinned
Qwen3-4B base, prior verifier adapter and continued verifier adapter on 14
candidates from seven complete real user messages in one primary personal
conversation. Two blind-to-prediction agent annotation passes agreed on both
axes for all 14 candidates; zero human reviewers. All references are eligible.
This is a balanced development challenge, not a held-out or natural-prevalence
evaluation. It measures supplied-candidate verification, not extraction or QA.

Base/prior/continued retained 6/7, 7/7, 6/7 supported claims; rejected 6/7,
7/7, 7/7 unsupported claims; and accepted 4/7, 0/7, 4/7 useful supported
eligible memories. There is no overall useful-memory gain over base on this
pilot. All 42 outputs were valid and complete, with identical input token IDs
across arms. Root inspected all disagreements and left frozen labels unchanged.
Scope uncertainty dominates; the continued adapter also rejects a faithful
project question. See [private results](private-pilot-results.md) and
[aggregate metrics](private-pilot-results.json).

No private training or checkpoint creation occurred. The original sources,
Context Kit and `~/.llm-context` remain read-only. Source anchors were verified
against the original file with unchanged source stat. All raw private data,
labels, prompts and receipts remain restricted under ignored `.local` paths.
The first 22-source subagent selection remains preserved as a failed selection.
Tracing its other candidate parent chains found the same primary conversation,
not extra independent sessions. Do not count those children as new training data.

The private annotation/export contract, inference runner and local evaluator
passed root review. Full source validation: 1,073 attempted, 1,072 passed,
one optional skip; Ruff/format on 220 files and strict mypy on 70 source files
passed. Synthetic v2 dev/test scoring reproduced exactly from archived receipts.
See [validation](validation-private-verifier.json). Signed delivery and any
remaining live metadata worker state are recorded in `.local/active-cycle.json`;
this checkpoint does not assert hosted CI or a pushed commit.

The private call used 240.87 worker seconds (181.75 loading), 263.98 client
seconds, zero retries and no training. Post-run inventory verified five stopped
apps and zero tasks. Automation stays paused. Verify live state before another
call and by 2026-10-17. Do not rerun the completed pilot or spend the v2 frozen
test again. Next: finish the bounded ownership/primary-session metadata inventory,
collect distinct primary sessions, preserve this pilot in development and build
a separate real-dialogue training cohort. The local owner-policy review packet
has seven messages without model predictions; human review is optional for
provisional research but required before claiming human-validated labels.

# Previous checkpoint: verifier v2 completed; private data selection corrected

Updated 2026-10-16 UTC. Qwen3-4B continuation training completed on 192 fictional
rows (two epochs, 96 optimizer steps). Historical transfer inference replay
matched exactly before training. The sole final adapter then completed its
predeclared single frozen test audit. Both dev and test criteria failed; no
product model selected. Test unsupported rejection improved 18/24 prior to
23/24 continued, and useful supported-memory retention improved 5/12 to10/12,
but supported retention fell24/24 to22/24. Uncertain eligibility remained weak.
See [results](verifier-v2-results.md), [separate dev](verifier-v2-dev-results.json)
and [test](verifier-v2-test-results.json). These are provisional agent-authored
fictional diagnostics, zero human reviewers, no private-data quality evidence.

Training used one personal Modal A10 for1,155.82 worker seconds, including
686.91 training seconds. The test audit used221.97 worker seconds. Both clients
exited0 and raw fictional replay inputs/receipts are archived under
`examples/synthetic/verifier-v2-*`. Previous failed launch receipts remain
unchanged. No test retuning or relaxed criteria is authorized. The separate
historical longer-conversation candidate remains unopened. The v2 test is spent.

The owner explicitly clarified that private conversations may be used for
training and evaluation on personal Modal, while staying out of GitHub. Do not
ask for this permission again. Exact wording is saved in ignored
`.local/private-real-model-20261016/owner-authorization-v1.json`. Keep private
transcripts, claims, receipts and any resulting private-data adapters out of
GitHub/public artifacts. Preserve Context Kit, original sources and
`~/.llm-context` read-only. Automation remains paused.

The first private shortlist review excluded all four windows: injected machine
context/assistant intentions or tool-transcript deltas, with no unambiguous
direct-human positive memories. Metadata shows all22 selected source files are
subagent sessions. Do not count their36 prepared windows as usable training
data. The original snapshot and pending template remain intact. Current work
selects substantive user dialogue from primary personal-project sessions and
implements private-provenance annotation/export plus a bounded private Modal
evaluation runner. All labels remain provisional; no private remote call has
run. See [realistic data](realistic-data.md) and [private review](private-evaluation.md).

Implementation/review workers are active; new private modules have NOT yet
passed root final review or a full source check. V2-only source validation
passed1,046/1,047 tests (one optional skip), Ruff/format and strict mypy across67
source files before this new private-data work. All new changes are uncommitted;
latest signed pushed commit remains87cf59823eb78e3c97cb30b09b4c5cfa3b635914.
Exact jobs, receipts, source hashes, validation and delivery:
`.local/active-cycle.json`. Verify live worker/Modal inventory before any launch.
Verify by: 2026-10-17.

# Previous checkpoint: broader verifier support gain; eligibility remains a blocker

Updated 2026-10-16 UTC. The unchanged trained Qwen3-4B verifier completed a
matched comparison on 24 newly authored fictional dev scenarios (48 candidates).
Unsupported catches rose from 10/24 base to 22/24 tuned; both retained 24/24
supported claims. False accepts fell from eight to two and uncertainty on bad
claims from six to zero. Both arms produced 48 valid, untruncated outputs.
The frozen primary criterion **failed** because it required at least 23/24
unsupported catches. Attribution catches rose 5/12 to 12/12; conditions 5/12 to
10/12. The remaining misses confuse if/only-if (case_transfer_20) and reverse
a keep/cancel branch (case_transfer_24); root checked both labels against the
complete fictional dialogue and upheld them.

Eligibility regressed: decisive exactness 27/42 to 25/42; uncertain-reference
exactness improved 4/6 to 6/6. A descriptive posthoc conjunction of support and
eligibility retains only 10/19 supported eligible claims versus 16/19 base,
while accepted unsupported eligible claims fall four to one. Do not equate the
support gain with a better complete memory pipeline. No product model selected;
zero human reviewers; all cases adaptive, provisional agent-authored dev.
The earlier failed criterion remains failed, its test remains unrun, and the
sealed longer-conversation candidate remains unopened.

The successful inference-only call used one personal Modal A10, worker time
166.310941s, client time 180.272934s, no training. The initial launch failed
before ML imports on Modal's normal symlink volume mount; a narrowly tested
canonical-mount correction preserved data, prompts, model, criteria and artifact
hashes. Both historical receipts remain intact. Post-run inventory showed all
apps stopped and zero tasks; verify again before any remote use and by
2026-10-17. Automation remains paused. See [results](verifier-transfer-results.md),
[machine report](verifier-transfer-results.json),
[protocol](verifier-transfer-protocol.md), and
[validation](validation-verifier-transfer.json). Archived fictional payload and
receipt replay through `experiments.evaluate_verifier_transfer` without a GPU.

Local private review preparation also works: 22 closed Codex sources from one
verified personal project yielded 36 dev windows and 183 record appearances.
No held-out project cohort was available; all authorship, semantic and privacy
labels remain pending. Nineteen oversized candidate windows were skipped;
two email indicators need local review. Original sources were rehashed and
preserved. Review artifacts remain restricted under ignored `.local`; no
private model inference, training or upload occurred. See
[private evaluation](private-evaluation.md).

The full source check before the narrow mount correction attempted 999 tests:
998 passed, one skipped, Ruff/format and strict mypy passed. After the correction,
all 41 focused tests, six-file lint/format and three-module type checks passed.
Root replay matched the archived report. Fresh-checkout CI exposed test fixtures
that assumed an existing `.local` directory and macOS temporary paths; the
fixtures now use isolated temporary workspaces and portable paths. The GPU
runner, payload, data and model outputs are unchanged. `.local/active-cycle.json` is the
source for exact signed delivery and hosted CI; this text does not assert them.
Next: expand directional-condition and temporary-personal-context training
examples, audit eligibility policy consistency, and freeze a fresh comparison
before further training. Human labels and approval for a concrete private
remote slice are still required for private-model evaluation. Preserve Context
Kit, original conversations and `~/.llm-context` as read-only.

## Previous checkpoint: paired verifier misses primary gate; no model selected

Updated 2026-10-16 UTC. The completed synthetic [paired-verifier run](verifier-pairs-model-results.md)
used 48 train rows, four epochs, 192 microsteps, and 48 optimizer steps. On
12 curator-visible dev candidates, supported retention moved 5/6 to 6/6,
unsupported catches 3/6 to 4/6, and decisive scope exactness 4/8 to 7/8.
The primary criterion failed because two unsupported claims remain labeled
supported; the secondary scope criterion passed. The test split is unrun and
no product model was selected.

The gain in unsupported catches changed one unsupported reference from
`uncertain` to `unsupported`; it did not reduce source-unsupported false
accepts. Remaining misses: `case-005::candidate-010` drops the handbill's
reporting source and `case-011::candidate-022` drops the “dry by noon”
condition. `case-035::candidate-069` retained support after tuning but scope
became uncertain. The six-pair result is a tiny correlated, curator-visible
diagnostic with provisional labels and zero human reviewers; it establishes
no QA, recall, or generalization claim. Training loss fell from 1.523054 to
0.003348, which is not quality evidence.

Training used 33,030,144 trainable LoRA parameters on a reported NVIDIA A10.
Training, worker, and client times were 281.848014s, 350.175540s, and
362.648002s. The standalone baseline outputs match the training call's
pre-training baseline outputs item by item. The first baseline attempt failed
before model loading on missing Modal distribution metadata; the repaired
runtime check changed no prompts or criteria. The adapter remains in the
private Modal volume, no private conversations were uploaded or used as model
data, and the sealed longer-conversation candidate remains unopened. The four
original extractor runs are unchanged.

Post-run inventory showed all Modal apps stopped and zero active tasks;
automation remains paused. Verify the stopped-job snapshot by 2026-10-17 and
before any future remote call. The frozen test must stay unrun. Next, broaden
independent attribution and condition cases, then evaluate the frozen adapter
on a larger independent set before considering another SFT. Replay artifacts
and the evaluator command are in the [results report](verifier-pairs-model-results.md).
Check `.local/active-cycle.json` for current delivery state; this checkpoint
does not assert commit or CI status.

Final local validation attempted 961 tests: 960 passed and one was skipped;
Ruff checked 198 formatted files and mypy checked 58 source files. See the
[validation record](validation-verifier-pairs-model.json).

## Previous checkpoint: verifier seed reviewed; matched model experiment next

Updated 2026-10-16 UTC. The new [paired-data seed](verifier-pairs-data-card.md)
contains 36 fictional scenarios and 72 candidate claims. Each pair has one
supported and one minimally changed unsupported claim with identical exact
citations. Source support and personal-memory scope are separate string labels;
uncertain scope is retained and excluded from decisive scope scoring.

Implemented `training/verifier_data.py` and `training/audit_verifier_data.py`.
The validator reuses the opt-in quote-v3 source checks and rejects cross-split
scenario/template/entity reuse. Prompts omit annotations and curator metadata.
The export accepts training only and yields 48 rows; dev/test messages are
absent. The CPU [audit receipt](verifier-pairs-audit.json) pins the corpus bytes
and separates all three splits. The keep-everything dev control retains 6/6
supported and 6/6 unsupported claims and rejects zero bad claims.

Root reviewed all 36 pairs, fixed a scope-label error and missing evaluation
scope coverage, tightened one claim object, and added positive controls against
simple polarity/causal-uncertainty shortcuts. Labels remain provisional
same-family agent judgments with zero human reviewers. Each dev/test split has
only six scenarios; both are curator-visible and not blind holdouts. Matching
identifiers and literal novelty do not establish semantic independence.

No training or inference ran in this slice. The earlier verifier failure and
all historical receipts remain unchanged; no model-quality gain or model
selection is claimed. Context Kit, private conversations and `~/.llm-context`
remain read-only; the sealed longer-conversation candidate was not opened.
Automation remains paused. Local validation is recorded in
[validation-verifier-pairs.json](validation-verifier-pairs.json); signed/private
delivery and exact-head hosted CI are governed by `.local/active-cycle.json`.

Next: implement and test a strict scorer for the new two-string output schema,
then freeze matched untuned/tuned controls, the exact run limits, development
criterion and test-release policy. The [next-experiment note](verifier-pairs-next-experiment.md)
proposes criteria but does not authorize an automatic launch. Existing owner
authorization covers bounded synthetic research; root still reviews exact
payloads and verifies personal Modal plus zero active jobs before any launch.
Keep uncertain eligibility out of decisive scoring and do not train this data
through the older boolean verifier schema. Do not run a fifth target-tweak
extractor SFT or consume the sealed candidate as a shortcut.

Live inventory at slice start showed four stopped apps, zero tasks and no
training clients. Recheck before runtime use; verify by 2026-10-17. Exact local
review, data/prompt freeze, exported rows and raw validation logs live under
`.local/verifier-pairs-20261016/`. Independent human annotation and a larger,
less correlated evaluation remain necessary before product-quality claims.

## Previous checkpoint: verifier criterion failed; no model selected

Updated 2026-10-16 UTC. The [Qwen3-8B verifier report](quote-verifier-dev-results.md)
covers 39 Qwen3-4B `quote-v2` candidates from fictional development cases.
The exact input gate left 39 candidates from 12 responses: one strict JSON
failure and one citation rejection. Those two no-candidate cases were gate
failures, not model abstentions.

The frozen labels classify 36 candidates as good and three as bad. The verifier
retained 35/36 good claims, rejected one good claim, and retained all three bad
claims. It produced valid, complete verdicts for all ten batches with no
truncation, but failed the predeclared criterion because it rejected none of
the bad claims. It also rejected one supported postcard statement by conflating
support with eligibility, and retained two ineligible exercise claims plus a
refund claim that dropped the reporting words “It says.”

A separate root audit found selected-fact coverage unchanged from the generator:
17/20 facts and 6/9 complete nonempty cases. The rejected postcard claim was not
a selected target. Candidate retention and selected-fact retention are separate
measures. Labels came from same-family AI agents, with zero human reviewers;
this development result establishes no independent truth, generalization, QA
benefit, or product suitability. No model was selected and Gate A remains
unmet.

The call used Qwen3-8B BF16 on reported NVIDIA A10 hardware (A10G requested).
Worker time was 144.376159s, GPU time 85.564929s, client time 149.756547s,
and output was 1,828 tokens. Billing was not measured. There was no retry, and post-run
inventory showed the app stopped with zero tasks. No training occurred.

The failed decision is frozen. The opt-in v3 validator is implemented and
root-reviewed, with 23 focused tests passing. The final local check passed
900 tests with one skip, Ruff across 188 files and configured mypy across 53
source files; verifier replay stayed byte-identical. See the
[v3 validation](validation-quote-data-v3.json). Before further evaluation, use the
opt-in [quote-data v3 contract](quote-data-v3-contract.md), which changes the
version and 12-message bound while preserving literal citation checks and
adding length validation; and review the prospective v0.1 [verifier annotation
contract](verifier-annotation-contract.md), which proposes separate labels
for semantic support and eligibility. The annotation policy is not human-approved;
unresolved eligibility stays uncertain in future diagnostics. Independent human
review is needed for human-validated or product-quality claims; the policy
ambiguity is not resolved. The sealed longer-conversation candidate remains unopened. No further GPU
call was launched. Do not run a fifth extraction target-tweak SFT. Verify by
2026-10-23: check `.local/active-cycle.json` and live state before any new
remote call.

The source code is signed and pushed at
`69c18e06d7877029cc9a8689169eccd62d8ddf82`; exact-head CI run
[38877899360](https://github.com/1337mus/tracemem/actions/runs/38877899360)
passed with 887/888 source tests (one skipped) and 399/401 installed tests
(two skipped). That run predates the result documents and v3 validator;
`.local/active-cycle.json` records their current delivery. See the
[machine score](quote-verifier-dev-results.json), [selected-fact audit](quote-verifier-selected-fact-audit.json),
and [validation record](validation-quote-verifier.json).

Only fictional data ran on Modal. Original conversations, Context Kit, and
`~/.llm-context` remain read-only; no private data was uploaded and nothing was
published publicly. Automation remains paused. Before any signed code,
documentation, or fictional-data push, make a fresh check of
`.local/active-cycle.json`, the private repository and branch state, and
exact-head CI.

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
