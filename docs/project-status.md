# Project status and design

Updated 2026-10-16 UTC.

TraceMem is a local research prototype for finding past conversation evidence
with source citations. The current search tool works; interpreting changes in
belief, decisions, or certainty remains experimental. The [README](../README.md)
has the project overview and architecture diagram.

## Phase and boundary

Phase 0 inventory and Phase 1 infrastructure are complete.
Phase 2 evaluation work continues on fictional data; Phase 3 diagnostic SFT is
complete for the current four-run series. Four synthetic-only runs have trained
Qwen3-4B adapters. No specialist has been validated or selected for product
use. Gate A remains unmet because there is no independent human validation of
the claims or evidence.

Claude Code and Codex adapters read local transcripts without changing them;
ChatGPT data enters only through an explicit JSON export reader. Context Kit
discovery and configuration stay unchanged. Private conversation data and its
derived evidence stay local.

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

## Latest completed training experiment: quote-v4

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

The latest completed source CI run tested `a0e82729fa165eef5b1a304dbf38a770ca6c0361`.
GitHub reports its signature as valid; exact-head CI run
[38862933699](https://github.com/1337mus/tracemem/actions/runs/38862933699)
passed: 810 source tests attempted (809 passed, 1 skipped) and 401
installed-wheel tests attempted (399 passed, 2 skipped). This run predates the
current result-document changes and does not verify them. The prepared job hash
is `9bb99c82d119ae356d53b5576d0345ec43d1b5950823954f7e33f566b61ab59a`.
The recorded post-run inventory had zero active tasks; recheck live state before
any future remote call. No model has been selected.

## Current evidence

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

The successful hosted CI run for the v3 source changes tested commit
a396d48da65b059bde55d0088e5996ec689cf300 (run 38828735844): 760/761 source
tests and 399/401 installed tests passed, with one and two skips respectively.
It did not include this later documentation working-tree tip. See
[the validation record](validation-sft-quote-v3.json) for the exact checks.
Check `.local/active-cycle.json` before any new remote run.

## Next gate

The next comparison is a dev-only prompt baseline across the 4B and 8B models,
with two prompt variants. Its reviewed 4B call is running on personal Modal;
8B will wait for completion. See the [protocol](quote-baseline-protocol.md)
and `.local/active-cycle.json` for live job and exact payload state. Do not run
a fifth target-tweak SFT. Write the stronger prompt from the original brief and
development rules without using test outputs. A fresh test and human-reviewed
evaluation are still needed before any Gate A or product claim. Verify remote
state before any future call; the quote-v4 test is spent.
