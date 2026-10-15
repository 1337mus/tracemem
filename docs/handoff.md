# TraceMem handoff

Updated 2026-10-15 UTC. Verify by: 2026-10-15. Read this file and ignored `.local/active-cycle.json`
first. Verify live git/process/worker state before resuming; never duplicate an
active GPU call. Context Kit, original conversations and `~/.llm-context` remain
read-only. Private data stays local.

## Current checkpoint

The [README](../README.md) now explains the project in plain language, includes
an architecture diagram, and separates working search from model experiments.
The long reference list moved into the [documentation index](README.md).
Both documented quickstart commands, `make demo` and `make benchmark`, passed
locally with fictional data and default hash vectors. Exact README delivery
and CI are in ignored `.local/active-cycle.json`; verify live.

The new offline [response validator and accounting tool](qa-controls-review-accounting.md)
regenerates the pinned 36-task bundle and validates completed templates. It
reports reviewers separately, preserves uncertain labels and always reports
zero verified humans. Seven source and seven installed tests passed; the
procedural CLI report replayed byte-for-byte. Root independently matched 72
mechanically assigned rows, six variant tables and 27 rejection cases. No
semantic annotation or inference occurred. Full source checks passed 591/592
(one optional skip), Ruff 131 files, mypy 32 runtime files and Actionlint.
See [validation](validation-qa-controls-review-accounting.json). Exact signed
delivery and final-head CI are in ignored `.local/active-cycle.json`; verify live.

The preceding packet commit's CI failed before installed tests because its
source export omitted `PYTHONPATH=src`. A one-line signed fix was privately
pushed as `87f0dcd75edcedf69a8adf6b695592c85de26e31`; local source/installed
exports matched after the fix. Preserve the failed run and authoritative
receipts. The fix CI38688228098 and accounting CI38688823456 succeeded. The subsequent
results-docs CI38689584406 also passed:591/592 source tests and338/340 installed
tests, plus lint, types, wheel build and all three packet comparisons. These
are separate from any later README-only delivery CI.

The actual 36-task packet also has a frozen key-withheld agent review and
separate unblinded root derivative. Both report source15correct/21incorrect;
original prompt-support36/0 becomes35/1 under root. Root reviewed every
rationale and corrected one mistaken claim that the full conditional-item
prompt was empty. Six variant tables/72 reviewer-task rows were audited and
the actual report replayed byte-for-byte outside the checkout. Original labels
are preserved; no human labels exist. See [results](qa-controls-review-results.md)
and [validation](validation-qa-controls-review-results.json).

Phase 0 inventory and Phase 1 infrastructure are complete. Phase 2 remains
synthetic diagnostics; no model is selected or trained. Gate A is unmet, with
zero independently verified human labels.

The latest offline work defines a source-only request and fake receipt contract
for 36 fictional QA control prompts. Request SHA-256:
`df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871`; prepared
payload SHA-256:
`36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1`. Per-prompt
ceilings are 1,024 input and 128 generated tokens; aggregate limits are upper
bounds only. The request itself contains no measured input-token counts. The API
validates reported structure/provenance, not actual execution, correct
tokenization/decoding or semantic truth. See [contract](qa-controls-answer-contract.md) and
[validation](validation-qa-controls-answer-contract.json).

Root independently reconstructed every request item and accepted only a fake
receipt; 61 invalid mutations were rejected. The accepted stop-boundary variants
are EOS only, EOS at the cap and cap without EOS. Full checks passed 556 tests
(555 passed, one skipped), Ruff 123 files and mypy 32 files. Installed replay
passed 13 tests, with all 37 runtime members matching the existing wheel. No
model, Modal or tokenizer imports were loaded. The TDD record includes observed
failures but does not claim one RED per rejection rule.

The source-free Modal producer is
[`modal_qa_interpretation_answer.py`](../experiments/modal_qa_interpretation_answer.py).
It was delivered in signed private commit
[`7c5bf12`](https://github.com/1337mus/tracemem/commit/7c5bf12f08eee3594a4173c9cac65f3747d79a36),
with valid signature and successful exact [CI run 38683345935](https://github.com/1337mus/tracemem/actions/runs/38683345935).
CI passed 568 source tests (one skipped), 315 installed tests (two skipped),
Ruff, mypy and wheel build. Root matched all 36 request items, rejected seven
altered requests before model imports, and checked four fake call modes. The
offline Modal serialization audit round-tripped the isolated body; it did not
verify live execution.

One reviewed synthetic call then completed using the fixed request and
source-only runner. The launcher command, aggregate receipt audit and limits are
in the [QA controls answer-run results](qa-controls-answer-run-results.md).
The 36 outputs were complete and nonempty; producer-reported usage was 6,993
input tokens (80–365 per prompt) and 196 generated tokens, all ending at EOS.
These counts were structurally audited, not independently retokenized. No local
model or tokenizer was loaded. The app stopped with zero tasks. The synthetic
groups are correlated, there are zero human labels, and Gate A remains unmet;
this run is not an answer-quality score or held-out evidence.

An offline strict-literal analysis now covers all 36 outputs and three variants.
It reports 26/36 prompt matches, 6/36 full-source matches and 27 exact
`UNKNOWN` outputs using casefold-plus-whitespace normalization only. These are
dev-label string matches, not semantic scores; the identical always-`UNKNOWN`
aggregate is not equivalent semantic quality. See the [run record](qa-controls-answer-run-results.md).
Root matched every row and rejected 16 invalid cases. Final local checks passed
575 tests (one skip), Ruff and mypy; seven installed tests passed and the report
replayed byte-for-byte. See [analysis validation](validation-qa-controls-answer-analysis.json).
Exact analysis delivery and CI are recorded in `.local/active-cycle.json`.

The preceding answer-contract implementation is delivered in signed private commit
[`8b9561b`](https://github.com/1337mus/tracemem/commit/8b9561bce29934a121247f305cf83110016c9f69).
Its signature is valid, remote main matched at verification, and exact [CI run 38681376612](https://github.com/1337mus/tracemem/actions/runs/38681376612)
passed 555 source tests (one skipped) and 302 installed tests (two skipped).
The controls fixture has six correlated groups, seven sessions, 22 messages, 12
questions and 36 cells; full source has 10 answerable and two unknown questions,
while all 24 support-omitted/empty cells expect `UNKNOWN`. The always-`UNKNOWN`
baseline is 26/36 on prompt expectations and 6/36 against the full-source axis.
It is not model quality or held-out evidence. See [controls](qa-interpretation-controls.md)
and [validation](validation-qa-interpretation-controls.json). The preceding
controls preparation has valid GitHub signature and successful exact
[CI run 38679061711](https://github.com/1337mus/tracemem/actions/runs/38679061711).

The earlier 46-answer review reported source correctness 32/13/1 and prompt
support 38/7/1; Root's unblinded derivative is 32/14 and 38/8. This is not an
independent sample or consensus; the frozen literal report is unchanged. See
[review results](qa-contrast-review-results.md) and
[review accounting](qa-contrast-review-accounting.md).

## Next useful slice

Verify final README-only delivery/CI from active-cycle before repeating work.
The controls review and accounting slice is complete and privately delivered;
do not repeat its inference or build more wrappers around the same fixture.
A fresh rights check matched all five pinned primary text files already covered
by `longmemeval-rights.md` and `benchmark-contracts.md`; it adds no new dataset
permission. Retain `.local/public-dataset-rights-root-audit-v1.json` and avoid
repeating this metadata audit. Next research priority is genuinely new,
license-clean scenarios kept separate from existing diagnostics and benchmark
tests, with provisional labels and an independent annotation path. Scenario
adequacy and independent validation remain unresolved; Gate A is unmet.
Training still needs separate approval. Context Kit and private inputs remain
read-only; no GPU call is needed for documentation or dataset preparation.

## Background

[Project status](project-status.md), [roadmap](roadmap.md),
[legacy answer runner](qa-contrast-answer-runner.md),
[Context Kit inventory](context-kit-inventory.md),
[older handoff archive](handoff-history-2026-10-14.md).
