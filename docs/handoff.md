# TraceMem handoff

Updated 2026-10-15 UTC. This is the latest delivery checkpoint; read
[active-cycle.json](../.local/active-cycle.json) for freshest owner, job,
session, receipts, and next action. Verify live state by 2026-10-15. The exact
previous handoff, including its full narrative, receipts, and links, is archived
in [handoff-history-2026-10-14.md](handoff-history-2026-10-14.md).

## Delivered checkpoint

Signed runtime baseline commit
[70fa402c654b0125870b73f254e8763f49a7c8f4](https://github.com/1337mus/tracemem/commit/70fa402c654b0125870b73f254e8763f49a7c8f4)
and documentation commit
[07ff6bf533b1ff7508346a95e771ee03a1305f6b](https://github.com/1337mus/tracemem/commit/07ff6bf533b1ff7508346a95e771ee03a1305f6b)
were delivered privately. Source CI [38657047378](https://github.com/1337mus/tracemem/actions/runs/38657047378)
and docs CI [38658597235](https://github.com/1337mus/tracemem/actions/runs/38658597235)
both succeeded. Root checks: 472 tests (one skip), Ruff on 102 files, mypy
on 32; installed focused checks: 69 tests and fictional evaluator/import
check. Hosted CI: 472 source tests (one skip), 220 installed tests (two skips).
The new wheel SHA-256 is
`0d005d093911a42ed850ab4972d79c3ef0daa64356556bc272438060d1674f60`.

The pinned 15,388,478-byte oracle replay passed all 500-question parsing,
normalization, evidence-tuple, scan, selected-five, and unchanged-input-file
checks. This verifies the narrow format compatibility described in
[the report](longmemeval-compatibility.md) and
[validation receipt](validation-longmemeval-integer-answer.json); it does not
establish QA or evaluator equivalence, data-use rights, or Gate A.

## Current bounded work

Latest private delivery: signed commit
[`d8d0847c058559aff791fae70a0ed757c7510dfa`](https://github.com/1337mus/tracemem/commit/d8d0847c058559aff791fae70a0ed757c7510dfa),
with exact [CI38668432855](https://github.com/1337mus/tracemem/actions/runs/38668432855)
success: 502 source tests (one skip), 250 installed tests (two skips).
Root verified the signature, owner/origin/PRIVATE visibility before pushing.
The fictional end-to-end demo was rerun successfully after delivery.

The [answering runner](qa-contrast-answer-runner.md) is implemented and root
reviewed across both source modules, all focused tests and installed-CI copies.
Twelve focused tests and full checks pass: 514 tests (one skip), Ruff on 113
files and mypy on 32 runtime files. Root independently compared all 46 prompts,
the complete generation protocol and resource envelope against frozen inputs.
Raw responses are flushed and fsynced before app teardown and validation.
Installed-wheel verification passed all 12 focused tests and exact offline CLI
replay, with all 37 runtime members unchanged. Root inspected that harness and
its evidence. The validation receipt records the actual failure chronology;
the log named `red-reconstructed` passed and is not RED evidence. A separate
bounded `gpt-6-luna/medium` worker, `/root/contrast_answer_analysis`, implements
the predeclared offline analysis; that follow-up remains unreviewed and is
excluded from runner delivery. Verify live ownership before resuming workers.
Private delivery and an actual launch are pending. No GPU job has launched
for this payload. Root freeze and predeclared
literal-analysis protocol are saved at `.local/qa-contrast-answer-root-freeze.json`
and `.local/qa-contrast-answer-scoring-freeze.json`. Verify live state by 2026-10-15.

Phase 0/1 is complete; Gate A is unmet, with zero human labels, no selected
model, and no training. The frozen first-five oracle retrieval diagnostic is
complete: BM25/hash/hybrid each recovered 0.80 macro explicit-label recall at
cutoff 5, with all labels found for 3/5 questions. This is five oracle questions,
not QA or an official score. Root independently reconstructed IDs, revisions,
labels, citations and metrics. See [results](longmemeval-oracle-retrieval-results.md)
and [validation](validation-oracle-retrieval.json). No model call occurred.
The runner wrote a final passed receipt; the outer memory-measurement wrapper
exited 1 on a denied macOS sysctl, so RSS is unavailable. Do not rerun it.

The new [development-only QA contrast preparation](qa-contrast-preparation.md)
is implemented and root reviewed: six fictional cases, 12 sessions, 60 turns,
399 unique prompts, 18 rankings and 72 byte-packing cells. Rankings differ for
6/6 cases; positive byte ceilings change retained membership within a fixed mode
for 6/6; explicit-label coverage differs for 3/4 answerable cases. Two unknowns
remain exclusions. Source/reference isolation and exact subset/packing behavior
are tested. Root independently reconstructed raw-source IDs, revisions, labels
and prompts. Full checks: 479 tests (one skip), Ruff 104 files, mypy 32 runtime
files. Seven installed tests and byte-identical artifact replay passed; all 37
installed runtime members and the historical catalog remain unchanged. See
[validation](validation-qa-contrast-preparation.json). References are provisional
agent-only; that preparation did not measure token counts or QA answers.

That preparation was privately delivered in signed commit
[7ada8c7841c74c24df9f2df0ced1edb50897a1c8](https://github.com/1337mus/tracemem/commit/7ada8c7841c74c24df9f2df0ced1edb50897a1c8).
Its [CI](https://github.com/1337mus/tracemem/actions/runs/38662273569) passed
479 source tests (one skip) and 227 installed tests (two skips).

A separately reviewed [CPU-only tokenizer counter](qa-contrast-token-counter.md)
then measured all 399 complete fictional prompts: 82–907 tokens, median 553.
Root checked exact identities/provenance and raw/validated equality; installed
replay accepted all counts without importing a tokenizer. The backend fingerprint
matches the historical counter. Full checks passed 487 tests (one skip), Ruff
106 files and mypy 32 runtime files; eight installed tests and exact payload
replay passed, with all 37 runtime members unchanged. The one Modal CPU app
stopped with zero tasks; no GPU, model inference or training ran. See
[validation](validation-qa-contrast-token-counter.json) and the fictional receipt.
The historical counter and receipts remain unchanged.

The counter was privately delivered in signed commit
[fdc45be01115fa68227c6ce959ef498322164ba3](https://github.com/1337mus/tracemem/commit/fdc45be01115fa68227c6ce959ef498322164ba3).
Its [CI](https://github.com/1337mus/tracemem/actions/runs/38664410629) passed
487 source tests (one skip) and 235 installed tests (two skips).

The [offline token-budget replay](qa-contrast-token-replay.md) is implemented
and root reviewed. All 72 cells match independently reconstructed subset-mask
packing, prompt identities/counts, citations and authored-label metrics. At
128/256/512/1,024 tokens the prompts retain 0/1/2/5 evidence lines. BM25 macro
label recall is 0/.25/.75/1; hash vectors 0/.25/.875/.875; hybrid 0/.125/.75/1.
Each aggregate has four answerable questions and two unknown exclusions.
All default cells are feasible; zero and mixed-feasibility ceilings are also
verified. These are provisional development labels, not QA accuracy.

Full checks passed 496 tests (one skip), Ruff 108 files and mypy 32 runtime
files. Nine installed tests passed; installed CLI output equals root exactly.
All 37 runtime members match the existing wheel, and historical inputs and
runners remain unchanged. No remote call or local model/tokenizer import ran.
See [compact cells](qa-contrast-token-replay-dev.json) and
[validation](validation-qa-contrast-token-replay.json). Read active-cycle for
the exact signed private delivery and CI state; verify it live before continuing.

The [source-only QA request preparer](qa-contrast-requests.md) is implemented
and root reviewed. All 46 unique prompts, 72 cell mappings and six references
match independently frozen expectations. A separate protocol freeze matches
the full Qwen revision, decoding and measured tokenizer settings. References
and scoring remain outside the payload. Totals are 23,533 unique input tokens
and at most 5,888 generated tokens; this is a bound, not measured cost.

Full checks passed 502 tests (one skip), Ruff 110 files and mypy 32 runtime
files. Six installed tests passed and all three JSON files reproduced
byte-for-byte, with all 37 runtime members unchanged. The request preparer
now binds its own source hash separately from the historical preparer hash.
A saved RED/GREEN proves that correction; the initial missing-API RED was
observed but its raw log was not saved. See [validation](validation-qa-contrast-requests.json).
No remote, GPU, local model or tokenizer call ran. Active-cycle records the
exact signed private delivery and CI once verified.

Next: finish installed verification, signed private delivery and exact CI for
the reviewed runner, then verify the personal workspace and zero active jobs
before one bounded synthetic-only GPU call. The runner rechecks exact counts,
template and backend before model loading, with no prompt truncation. Keep raw
responses and use the predeclared literal scoring protocol for the subsequent
development report. Gate A remains unmet; this does not select an extractor.

Before continuing, read `.local/active-cycle.json` and verify live ownership,
jobs, sessions, and next action; do not duplicate a remote job. Modal work is
limited to exact reviewed, bounded, synthetic-only single jobs; none is needed
for this slice. Original sources, Context Kit, and `~/.llm-context` stay
read-only. Keep private data and IDs local. Do not use local models or
tokenizers, train, or publish. Before each signed push, verify repository
owner, origin, and PRIVATE status.

## Durable references

- [Roadmap](roadmap.md) and [Context Kit inventory](context-kit-inventory.md).
- [Token-count results](token-count-results.md).
- [Synthetic QA results](synthetic-qa-qwen3-results.md) and
  [inference validation](validation-synthetic-qa-inference.json).
- [LongMemEval compatibility](longmemeval-compatibility.md) and
  [integer-answer validation](validation-longmemeval-integer-answer.json).
- [Frozen-five retrieval results](longmemeval-oracle-retrieval-results.md).
- [Development-only QA contrasts](qa-contrast-preparation.md).
- [Exact development prompt token counts](qa-contrast-token-counter.md).
- Full previous checkpoint and its detailed receipts:
  [handoff history](handoff-history-2026-10-14.md).
