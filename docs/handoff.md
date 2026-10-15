# TraceMem handoff

Updated 2026-10-15 UTC. Verify live state by 2026-10-15. Read this file and
ignored `.local/active-cycle.json` first; inspect active ownership before work.
Original sources, Context Kit, historical receipts and `~/.llm-context` remain
read-only. Private conversations, vectors, claims and checkpoints stay local.

## Verified delivery

QA review tooling commit [`125d3660066cd610e8229fc131714fcb33019cdb`](https://github.com/1337mus/tracemem/commit/125d3660066cd610e8229fc131714fcb33019cdb)
was signed, locally GPG-verified and privately pushed after confirming personal
owner, origin and PRIVATE visibility immediately before push.
[Exact CI 38674186708](https://github.com/1337mus/tracemem/actions/runs/38674186708)
passed: 526 source tests (one skip), 274 installed tests (two skips).
The earlier [analysis CI 38673059955](https://github.com/1337mus/tracemem/actions/runs/38673059955)
also passed and reproduced the literal report byte-for-byte. All 37 runtime
members remain unchanged from wheel SHA-256
`0d005d093911a42ed850ab4972d79c3ef0daa64356556bc272438060d1674f60`.

GitHub marked the tooling commit's locally valid signature `bad_email`: the
committer identity differed from the signing key. The next documentation
checkpoint uses a per-command email matching the registered verified key;
verify its actual GitHub signature state in active-cycle. No history is rewritten.
Local GPG validity and GitHub's verified badge are separate recorded facts.

Check active-cycle for the exact newest signed delivery and CI outcome;
never infer pushed from implemented. Validation JSON files preserve the pre-push
snapshots; this handoff records subsequently verified delivery.

## Completed bounded experiment

One source-free Modal answering call completed successfully in the verified
personal workspace. App `ap-kGBUTuuMF5u3lgRYeVmdwp` stopped with zero tasks.
No duplicate call or retry ran. Raw response was flushed/fsynced before app
teardown and validation; raw and validated result objects match exactly.

- Frozen payload: `80bd1b68eb0f4fabc7262f00ba17ddfbeeaa4079b924d1228881253029bc7b4e`.
- Reviewed runner: `1e1b7409c7a19bc4c0d51db72eb9256e6b1667ebc311af01d257e395e734a3db`.
- 46 unique fictional prompts map to 72 correlated development cells.
- 23,533 input tokens, 479 generated tokens; 46 EOS stops, no truncation.
- T4 peak allocation 8,822,848,512 bytes; model load 78.372997 seconds,
  GPU event duration 39.131037 seconds, remote elapsed 129.994857 seconds,
  client elapsed 234.095551 seconds. These are not billing measurements.
- Every mode/budget group: 0/4 answerable literal matches, 2/2 unknown literal
  matches, 2/6 overall. Verbose paraphrases occur: this is not semantic accuracy.
- References are provisional agent-authored labels: six dev cases, zero human
  labels, no test partition or independent-trial claim. Scoring was frozen
  before outputs; casefold/whitespace only, with no post-hoc normalization.

See [results](qa-contrast-answer-results.md), [full cells](qa-contrast-answer-dev.json),
[raw response](../examples/synthetic/qwen3-4b-qa-contrast-answer-raw.json),
[validated receipt](../examples/synthetic/qwen3-4b-qa-contrast-answer-remote.json)
and [validation](validation-qa-contrast-answer-analysis.json).
Root independently reconstructed all 72 cells/12 groups against frozen mapping
expectations and raw outputs, without importing the analyzer.

## Validation and ownership

Analysis full checks passed: 520 tests (one skip), Ruff 115 files, mypy 32.
Six focused installed tests and fake-receipt CLI replay passed outside checkout.
Actual-receipt source/copied/expected report bytes also match outside checkout,
with all 37 runtime members unchanged and heavy imports blocked. Root reviewed
the source, tests, both installed harnesses, CI, docs and fictional artifacts.
Inspect active-cycle for any remaining delivery watcher; do not duplicate it.

Local audit references:
`.local/qa-contrast-answer-inference-root-audit.json`,
`.local/qa-contrast-answer-analysis-root-audit.json`,
`.local/qa-contrast-answer-final-modal-inventory.json`.
The first post-return audit omitted PYTHONPATH and failed setup; its corrected
rerun passed without another inference. The runner log named `red-reconstructed`
passed, so it is not RED evidence. Detailed chronology stays in validation docs.

## Qualitative review preparation

The [QA review workflow](qa-contrast-answer-review.md) now exports 46 explicit
question/prompt/source/answer tasks, a separate authored-reference/provenance key
and a completely blank response template. All 72 cell mappings are preserved.
It reuses strict parsing, reviewer checks and exclusive output creation; it does
not misrepresent QA as claim extraction or reuse the claim-only HTML schema.
Full-source correctness and supplied-prompt support are separate review fields.

Root reviewed both new files and independently checked every task against saved
preparation expectations. Full checks passed: 526 tests (one skip), Ruff 117
files and mypy 32. Six installed focused tests pass; all three actual export
files match source/copy bytes outside checkout with heavy imports blocked.
All 37 runtime members still match the prior wheel. See the
[validation snapshot](validation-qa-contrast-reviews.json) for exact hashes and
honest RED/GREEN limits. Review source and tests were privately delivered in
`125d366`; the exact CI passed as recorded above.

The actual local packet is `.local/qa-contrast-reviews-source-export/`.
`bundle.json` and `response-template.json` may go to a blinded reviewer;
withhold `key.json`. No review labels have been filled in. A lossless grouped
reading view at `.local/qa-contrast-reviews-reading-view.json` reduces compact
JSON from 140,852 to 71,429 bytes; exact reconstruction matches the bundle hash.
This is byte-size reduction, not a tokenizer measurement. Its separate local
receipt pins the view hash and reconstruction check.

## Next useful action

Verify the newest signed private delivery and exact CI in active-cycle. Then
use the prepared packet for a bounded qualitative review; report any agent labels
as agent-only and keep them separate from independent human labels. Validate the
completed response document and add descriptive per-condition accounting without
relaxing the predeclared literal metric. No new GPU inference is needed.

Phase 0/1 is complete. Gate A remains unmet: zero independent human labels,
no selected model or training. No public release, private-data upload,
destructive change or expensive training is authorized. Modal is limited to
separately reviewed, bounded, synthetic-only calls, with fresh personal workspace
and zero-active-job checks. Verify PRIVATE visibility before every signed push.
Computer Use is unused; continuation automation remains active.

## Durable background

[Project status/design](project-status.md), [inventory](context-kit-inventory.md),
[roadmap](roadmap.md), [model comparison](experiments-scenarios.md),
[token replay](qa-contrast-token-replay.md), [request preparation](qa-contrast-requests.md),
[oracle retrieval](longmemeval-oracle-retrieval-results.md),
[format compatibility](longmemeval-compatibility.md).
Earlier detail is preserved in Git history and
[the previous handoff archive](handoff-history-2026-10-14.md).
