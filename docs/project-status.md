# Project status and design

Updated 2026-10-16 UTC.

TraceMem is a local research prototype for finding past conversation evidence
with source citations. The current search tool works; interpreting changes in
belief, decisions, or certainty remains experimental. The [README](../README.md)
has the project overview and architecture diagram.

## Phase and boundary

Phase 0 inventory and Phase 1 infrastructure are complete. The project is in
Phase 2, using fictional data for diagnostics. Three technical SFT runs have
trained Qwen3-4B adapters, but no specialist has been selected for product use.
Gate A is unmet: there is no independent human validation of the claims or
evidence.

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

## Latest completed training experiment: quote-v3

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

Strengthen annotations and make paired chronology and subject-attachment
corrections with distinct scenario families and a separate frozen test. Seek
human review before making any Gate A claim. Do not treat more optimizer steps,
targeted agent probes, or retrieval diagnostics as evidence of specialist
quality or readiness.
