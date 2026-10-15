# Project status and design

Updated 2026-10-15 UTC.

TraceMem is a local research prototype for finding past conversation evidence
with source citations. The current search tool works; interpreting changes in
belief, decisions, or certainty remains experimental. The [README](../README.md)
has the project overview and architecture diagram.

## Phase and boundary

Phase 0 inventory and Phase 1 infrastructure are complete. The project is in
Phase 2, using fictional data for diagnostics. A small technical SFT pilot has
trained a Qwen3-4B adapter, but no specialist has been selected for product
use. Gate A is unmet: there is no independent human validation of the claims
or evidence.

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

## First training experiment

The [SFT pilot](sft-pilot.md) completed a Qwen3-4B adapter run on 128 fictional
examples and compared base and tuned outputs on the same eight development
prompts. Both had zero exact claim matches. The [results report](sft-pilot-results.md)
records the limitations and failure analysis. This tested the learning
pipeline while Gate A remains unmet; it did not select a production specialist
or establish held-out quality.

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

The verified hosted CI baseline at preparation time is for the delivered head
`bb5eb2582e9fa35b7b175db643cb159a0dddce17` (run
`38698341232`): source tests passed 602 with 1 skip; installed tests passed 349
with 2 skips. Check `.local/active-cycle.json` for the latest exact-delivery
and CI state.

## Next gate

The next research step is independent human review of the full longitudinal
packet, followed by a separate readiness review. Add more legally usable,
distinct scenarios before selecting or training a specialist. Do not infer
Gate A progress from targeted agent probes or retrieval diagnostics.
