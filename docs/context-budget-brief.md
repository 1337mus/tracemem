# Fixed context budget diagnostic

Updated 2026-10-14. Prototype; synthetic fixtures only, no model/network calls.

Objective: measure relevant and sufficient source evidence after the existing
whole-passage packer enforces identical UTF-8 byte budgets. This is retrieval
and packing evidence, not QA accuracy, tokenizer counts or trained-model quality.

Keep `pack_context(hits, max_bytes)` output and validation exactly compatible.
Add `PackedContext` (immutable text and selected hits) and
`pack_context_with_manifest(hits, max_bytes)` in `retrieval.py`. Both APIs must
use one selection implementation, preserving rank order and existing behavior
that skips oversized passages and can include smaller later passages. Include
all metadata/citation bytes and newline separators in the budget. Selected-hit
manifest must exactly identify emitted whole passages, including Unicode and
repeated citations; do not parse packed raw text to recover selected IDs.

New `budget_benchmark.py` measures the existing authored synthetic retrieval
fixture at budgets 0,256,512,1024,2048,4096 bytes and fixed depth5. Retrieve once
per query/mode then pack the same hits at every budget. Keep BM25, signed-token-
hash vectors and RRF settings identical and explicit. Report per-query retrieved
and packed IDs/citations/bytes, recall of relevant messages, whether any authored
sufficient evidence set survives, and empty-pack status. Keep unknown queries
separate; empty pack is mechanical and is not calibrated semantic abstention.
Summarize answerable sufficiency/recall per mode/budget with null denominators.
Use existing benchmark metric helpers when suitable, but do not change them or
mislabel raw-hit metrics as packed evidence metrics. Validate budgets: nonempty,
distinct nonnegative integers (reject bool), bounded reasonable list; k1..100.
Default hash vectors only; no dependency or model downloads. Record fixture
SHA256, index/embedder versions, k, budgets, environment and deterministic config.

CLI `experiments/evaluate_context_budgets.py --output PATH` creates a new result
exclusively; no overwrite or stdout source text. Public receipt contains only
fictional IDs/citations and metrics. Root generates/reviews receipt after code.

Worker owns ONLY `src/tracemem/retrieval.py`,
`src/tracemem/budget_benchmark.py`, `tests/test_retrieval.py`,
`tests/test_context_budgets.py`, `experiments/evaluate_context_budgets.py`,
`docs/context-budget.md`. Root owns other docs, fixtures and delivery.
Apply TDD RED/GREEN per behavior; <=500 new production LOC total. Meaningful
tests: Unicode/metadata budget accounting, oversized-first/later-small selection,
manifest/text agreement, zero budget, unchanged existing packer behavior; real
SQLite fixture with sufficient-set lost/preserved and unknown-query denominators;
existing output unchanged. Run targeted tests + Ruff + source/scoped strict mypy.
Root reviews all source/tests and uses one final full-suite gate for both slices.
