# Project status and design

Updated 2026-10-17 UTC.

TraceMem is a local research prototype for finding past conversation evidence
with source citations. The current search tool works; interpreting changes in
belief, decisions, or certainty remains experimental. The [README](../README.md)
has the project overview and architecture diagram.

## Phase and boundary

Phase 0 inventory and Phase 1 infrastructure are complete. Phase 2 research
continues on fictional and bounded private development data. A balanced-support
recovery produced a scored comparison, but both predeclared development gates
failed. No model has been selected or promoted, and Gate A remains unmet because
the labels have no independent human validation. See the
[recovery results](balanced-support-results.md) and the earlier
[uncertainty-training comparison](uncertainty-training-results.md).

Claude Code and Codex adapters read local transcripts without changing them;
ChatGPT data enters only through an explicit JSON export reader. Context Kit
discovery and configuration stay unchanged. Private conversation data and
derived evidence remain restricted to the authorized local and private training
workflow; only reviewed aggregates are documented here.

## Design

Adapters normalize messages while preserving speaker roles, source locations,
and revisions. SQLite stores the messages and citations. Search combines BM25
keyword ranking with deterministic signed hash-vector ranking; either method
can also run alone. A budgeted passage packer returns complete cited messages
within the consumer's token limit.

Experiments propose claims from source messages. A separate append-only claim
ledger stores those proposals and their history. Validators check structure,
quoted source text, speaker roles, and revision references. Proposals do not
feed the search index. No trained model has been selected for this step.

## Latest balanced-support comparison

The reviewed recovery trained on 256 row exposures: 85 supported, 85
unsupported, and 86 uncertain, including repeats. These exposures came from
193 unique source rows, including 37 private rows out of 50 available. The run
used 64 optimizer steps and took 334.441137 seconds. On 32 dev prompts, all 96
outputs were valid JSON and ended with EOS; base and prior input/output token
arrays matched on 32/32 cases, and candidate inputs matched on 32/32 cases.

Compared with the historical 256-row adapter, support retention was unchanged
in both cohorts. The recovery rejected one more unsupported private claim, but
accepted one fewer useful private reference and had one fewer exact private
eligibility label. Fictional eligibility improved to 16/16 and uncertain
support to 2/2. Both frozen gates failed; no model was selected or promoted.
See the [full comparison](balanced-support-results.md) for all counts.

The first approved launch remains a separate historical failure: it stopped
during runtime setup before training, consumed no schedule rows, and produced no
score. A public-only CPU checkpoint cache and offline loader later succeeded
(ten public files, about 8.06 GB, remote-only); the cause of the first failure
remains unestablished. Labels in both scored cohorts are provisional agent
judgments with zero human reviewers. Evaluation is adaptive and not held out;
these results do not support causal or downstream QA claims. The reserved
verifier test remains unused.

## Earlier uncertainty comparison

The [historical uncertainty-training report](uncertainty-training-results.md)
documents the preceding synthetic-v2 continuation, which failed both gates, and
its separate training-fit diagnostic. Its results remain unchanged by the later
recovery.

## Earlier synthetic paired-verifier result

The [36-scenario corpus](verifier-pairs-data-card.md) pairs supported claims
with small changes in meaning and keeps source support separate from memory
scope. A Qwen3-4B adapter trained on 48 fictional rows. On 12 curator-visible
dev candidates, supported retention increased from 5/6 to 6/6 and unsupported
catches from 3/6 to 4/6; the number of unsupported claims still labeled
supported stayed at 2/6. The primary criterion failed. Decisive scope labels
correct increased from 4/8 to 7/8, passing the separate secondary criterion.
The test split remains unrun; labels have zero human reviewers. See the
[results](verifier-pairs-model-results.md), [machine-scored report](verifier-pairs-model-results.json),
[validation record](validation-verifier-pairs-model.json), and frozen
[protocol](verifier-pairs-model-protocol.md). This small diagnostic does not
establish QA benefit, extraction recall, or generalization.

## Earlier extraction experiment: quote-v4

V4 compared the pinned base, unchanged quote-v3 adapter, and a newly trained
adapter on 12 dev and 24 frozen test cases. On test, the tuned arm completely
retained all selected facts in 8/18 nonempty responses versus 9/18 for the
prior. Adverse responses rose from 3/24 to 6/24. It therefore failed the
predeclared threshold of three additional complete nonempty test cases with no
increase in adverse responses or invalid citations. These labels are
provisional agent/root judgments from one model family, with zero human
reviewers. See the [results report](sft-quote-v4-results.md) and the
[protocol](sft-quote-v4-protocol.md). The test is spent; no tuning against it
is planned. No model was selected and Gate A remains unmet.

## Earlier prompt baseline: quote-only Qwen3 comparison

The adaptive development comparison covered 12 fictional cases, with 4B and
8B Qwen3 models each receiving both `quote-v2` and `quote-coverage-v1`. Paired
input token arrays matched across all 24 items. Both 4B prompts retained 17/20
selected facts; both 8B prompts retained 16/20. Complete selected-fact cases
were 6/9, 7/9, 6/9, and 6/9 respectively. All four arms correctly abstained on
0/3 empty-target cases. These results do not establish exhaustive recall,
held-out quality, QA benefit, or a production model selection. Two blind Luna
reviewers and root reviewed the outputs; there were zero human reviewers and
the semantic labels remain provisional. See the [full baseline report](quote-baseline-results.md)
and [protocol](quote-baseline-protocol.md).

## Earlier quote-v2 verifier check: development criterion failed

Qwen3-8B judged 39 Qwen3-4B `quote-v2` candidates that passed the exact
deterministic gate. It retained 35/36 provisionally good claims, but also kept
all three provisionally bad claims and rejected one good claim. All ten batches
were valid and untruncated; the overall predeclared criterion failed. Two
no-candidate cases were generator JSON and citation failures, not model
abstentions. A separate root audit found selected-fact retention unchanged at
17/20 (6/9 complete nonempty cases); candidate retention is a different
measure. Labels are provisional same-family agent judgments with zero human
reviewers. See the [results](quote-verifier-dev-results.md), [score](quote-verifier-dev-results.json),
[fact audit](quote-verifier-selected-fact-audit.json), and
[validation record](validation-quote-verifier.json).

## Prior training experiment: quote-v3

V3 compared base, quote-v2 prior, and quote-v3 tuned arms on a 24-case adaptive
regression set, 24 development cases, and a separate 48-case test. In the
test's targeted chronology and subject group, responses with any unsupported
or uncertain claim fell from 5/32 to 2/32; overall they fell from 7/48 to
3/48. Required-fact coverage fell from 21/36 to 16/36 overall and from 13/24
to 9/24 on the targeted group, so the run failed its predeclared promising
criteria. The labels are provisional agent/root judgments with zero human
reviewers. No specialist or production checkpoint was selected, and Gate A
remains unmet. The [results report](sft-quote-v3-results.md), [evaluator
report](sft-quote-v3-results.json), and [validation record](validation-sft-quote-v3.json)
keep structural, semantic, and coverage evidence separate. The
[quote-v2 report](sft-quote-v2-results.md) and [first pilot](sft-pilot-results.md)
remain historical.

## Quote-v4 run record

The run completed 192 optimizer steps on 768 fictional rows from 72 correlated
families. The adapter changed from its verified initial state and was saved to
the separate personal Modal volume; weights were not downloaded to this Mac.
No personal conversations entered training or inference, and billing was not
measured. The [results report](sft-quote-v4-results.md) links the job, receipt,
structural output, final semantic review, and validation record.

## Current evidence

The current full check ran 1,138 tests: 1,137 passed and one optional test was
skipped. Ruff checked 229 files and configured mypy passed 74 files. Fifty-one
focused tests and nine scorer-regression tests passed. Hosted CI covers the
previously published head only; CI for the current changes is pending.

The current longitudinal review packet contains 26 tasks and is available at
`.local/longitudinal-v2-review-ready/reviewer/review.html`.
The targeted probe selected four development cases and covered 10 tasks: 6
supported, 3 unsupported, and 1 uncertain. It has zero human validation; these
labels do not measure accuracy or generalization. V2 revises the same 12
fictional scenarios as v1 and adds no independent data. The
[longitudinal v2 review card](longitudinal-v2-review.md) describes the review.

The six fictional cross-session cases are assigned to a `test` split for
bookkeeping only. Four answerable cases have complete labeled evidence by k=3
with BM25 and hybrid retrieval; hash vectors cover 2 of 4 at k=3. Two
`UNKNOWN` cases are excluded from turn-label recall, so this is not an abstention
or QA accuracy result. See the [cross-session data card](cross-session-test-data-card.md)
for the retrieval details and limits.

Focused review verification covered 23 source tests and 23 installed tests.
Reports were byte-identical, and the full exports and probe exports matched.
These checks validate the artifact slice, not human agreement or truth.

The earlier v3 hosted CI result is historical and predates this working-tree
tip. See [the validation record](validation-sft-quote-v3.json) for that run's
exact checks. Check current local state and live inventory before any remote run.

## Next gate

An eligibility-focused audit found 67 uncertain-eligibility exposures. A
54-candidate, 27-scenario fictional training set with joint 3×3 balance is being
authored; blind labels are required before training. Claude metadata verification
is complete: five older leads included three already represented in development
and two unverified; no new private rows were admitted. Keep the paired-verifier
test split sealed. Broaden independent attribution and condition cases, then
evaluate on a larger, independent set before considering more verifier training.
Independent human annotation is still required before a human-validated or
product-quality claim; the sealed longer-conversation candidate remains
unopened. The prospective v0.1
[annotation contract](verifier-annotation-contract.md) is not human-approved;
uncertain scope must remain separate. Do not run another extraction target-
tweak SFT. No production model was selected and Gate A remains unmet. Check
`.local/active-cycle.json` and live state before any remote call.
