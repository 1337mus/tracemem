# TraceMem handoff

Updated 2026-10-14. This checkpoint records the completed synthetic QA run and
the current offline follow-up. Verify live GitHub and worker state before any
operational action by 2026-10-15. No new inference is authorized by this handoff.

## Current state

Corrected-source commit `814c5ca09aa251f6d06c918adf0e621a35dd888a` is signed
and pushed to private `1337mus/tracemem`. CI [38648259693](https://github.com/1337mus/tracemem/actions/runs/38648259693)
passed: 447 tests in 42.572s (one optional Context Kit skip), 195 installed
tests in 16.149s (two skips), Ruff on 96 files, and source mypy on 31 files.
The earlier Python 3.11 image initialization failure remains documented with
its original receipt and hashes in
[`validation-synthetic-qa-inference.json`](validation-synthetic-qa-inference.json).

Exactly one corrected-source Qwen3-4B call completed successfully with 72
answers. The Modal app is stopped with zero tasks; the read-only live snapshot
reported zero active GPU jobs. Root independently replayed all 18 analysis
cells, all 72 literal outcomes, and token/empty/finish counts. An installed-wheel
reproduction outside the checkout, with `PYTHONPATH` and `PYTHONHOME` unset and
no model or network calls, reproduced the analysis byte-for-byte. The raw receipt,
full analysis, root replay, and installed replay are preserved under `.local/`;
their durable documentation copies and limitations are listed in
[`synthetic-qa-qwen3-results.md`](synthetic-qa-qwen3-results.md).

One offline token-budget packing worker is active under
`.local/token-packing-brief.md`; root review is pending. It is the next useful
step: implement generic token-budget packing and verify it offline. No GPU jobs
are active. Do not launch or retry inference based on this document.

Phase 0/1 remain complete; see [`roadmap.md`](roadmap.md) and
[`context-kit-inventory.md`](context-kit-inventory.md). Context Kit at
`/Users/raj/Projects/ctxkit`, original conversations, and `~/.llm-context` remain
read-only. Keep model weights, tokenizers, and inference off this Mac. Reviewed
code, docs, and fictional fixtures may be signed and pushed only to private
`1337mus/tracemem`; assert the repository is still PRIVATE immediately before
each push. No public publication, private-data upload, or training is authorized.

## Result boundary

All 18 cells report literal agreement 1/4 overall, 0/3 for answerable questions,
and 1/1 for unknown questions. Every zero-context cell returns `UNKNOWN` for
all four questions. All 72 outputs ended in EOS; none were empty, length-stopped,
truncated, or hit the generation limit. The 18,078 input tokens range from 78
to 384; outputs used 636 generated tokens including EOS. GPU compute took
54.679459s, model loading 76.35161s, remote execution 137.175618s, and peak
allocated GPU memory was 8,822,848,512 bytes.
Cleanup inventory `.local/synthetic-qa-python312-cleanup.json` (checked at
`2026-10-14T22:39:23.111435Z`) records both apps stopped with zero tasks and
zero active GPU jobs.

Root replay found that each question's nonzero rendered context was identical
across retrieval modes and 512/2048-byte ceilings. This run cannot compare
retrieval modes or context capacity. Literal matching does not normalize
punctuation or semantic equivalence; observed differences include `teal` vs
`Teal.`, `bassoon` vs a full-sentence answer, and `No` vs `UNKNOWN`. Do not
rescore, claim semantic accuracy, salvage test performance, or call this
training-ready. References are provisional agent-authored; Gate A and
independent human labels remain unmet. Byte ceilings are UTF-8 bytes, not
tokenizer-matched budgets. There is no pooled score or model ranking.

## Durable evidence

- Raw answer receipt copy: `examples/synthetic/qwen3-4b-qa-diagnostic-remote.json`
  (SHA256 `1da725c57bf858a6cf205bb300122616904520ad2966ff212f84605ca25d0738`).
- Full analyzer output: `docs/synthetic-qa-qwen3-analysis.json` (SHA256
  `0b3932af0f140f16fd98e7d01d60f94072162c01bba312173f32a6c4886af89f`).
- Split artifacts: `docs/synthetic-qa-qwen3-dev.json` and
  `docs/synthetic-qa-qwen3-test.json`; each contains only its split's nine
  cells and 36 outcomes.
- Analysis procedure and separate results: `docs/synthetic-qa-analysis.md` and
  `docs/synthetic-qa-qwen3-results.md`.
- Correction, original failure receipt/hash, prior source-validation hashes,
  corrected run, root replay, cleanup, and CI facts:
  `docs/validation-synthetic-qa-inference.json`.

The working tree may include the independent offline token-packing slice. Do not
change `.local/active-cycle.json`, receipts, or source as part of result
documentation. Keep private artifacts under ignored `.local` with restrictive
permissions.
