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
agent-only; no token counts or QA answers were measured for this new fixture.

Next: prepare a separately reviewed source-free CPU-only tokenizer counter for
this exact 399-prompt catalog. Preserve the historical pinned runner. Review
payload/runtime hashes and verify the personal Modal workspace before any new
remote call; no GPU inference is needed yet. No worker or GPU call remains
active at this checkpoint; verify live state before continuing.

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
- Full previous checkpoint and its detailed receipts:
  [handoff history](handoff-history-2026-10-14.md).
