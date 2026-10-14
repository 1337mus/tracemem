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

The generic offline token-budget packing foundation is implemented and root-
reviewed locally. The full suite, static checks, independent oracle, and an
installed-wheel replay passed; exact evidence and limitations are recorded in
[`validation-token-budget-packing.json`](validation-token-budget-packing.json).
Root signed the exact reviewed source, test, documentation, and workflow change
as commit `717ff8c986238cbbf5b133238df9ad17729bc8bb`, delivered privately with
validation documentation at `d7bb4f8bec846295ecb2eb8a96e0e55aff20dc79`.
CI [38650751759](https://github.com/1337mus/tracemem/actions/runs/38650751759)
passed: 454 source tests (one skip), 202 installed tests (two skips), Ruff on
98 files, and mypy on 32 source files. Matched-token QA has not run. A bounded
offline prompt-count catalog is now implemented and root-reviewed in signed
commit `2f959a8347fd4ccbb97c4db100c1d1b587746650`. Root's full check passed 460
tests (one optional Context Kit skip), Ruff on 100 files, and mypy on 32 source
files; the independent oracle and isolated installed-wheel replay also passed.
Catalog-specific hosted CI is pending. The earlier token-packing CI run above
does not validate this catalog commit. Exact hashes, test chronology, the
preserved installed-harness failure, and remaining limitations are recorded in
[`validation-token-count-catalog.json`](validation-token-count-catalog.json).
The catalog prepares 28 synthetic prompt items from 84 subset instances and
replays caller-supplied counts; its invented test/oracle integers are not token
measurements, and it does not perform QA or tokenizer-attested counting.

The last personal Modal workspace inventory, at
`2026-10-14T22:56:18.872005+00:00`, found no active apps or tasks; refresh the
personal Modal workspace and no-active-job state before remote counting. Before
any push, separately verify that the GitHub repository is still PRIVATE. A
reviewed fixed-fictional count payload is staged locally at
`.local/token-count-payload-reviewed.json`
(payload SHA256 `125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88`);
it has not been sent, and no count runner hash or remote call exists. Under
existing bounded synthetic-only authority, after separate source/payload
review, the next bounded step is source-free remote tokenizer counting of that
payload, with recorded catalog, payload, and runner hashes. Counts for
overlapping prompts can be checked against existing QA `input_tokens` without
new QA inference. Do not load model weights, run new QA inference, or train.

Phase 0/1 remain complete; see [`roadmap.md`](roadmap.md) and
[`context-kit-inventory.md`](context-kit-inventory.md). Context Kit at
`/Users/raj/Projects/ctxkit`, original conversations, and `~/.llm-context` remain
read-only. Keep model weights, tokenizers, and inference off this Mac. Reviewed
code, docs, and fictional fixtures may be signed and pushed only to private
`1337mus/tracemem`; assert the repository is still PRIVATE immediately before
each push. No public publication, private-data upload, or training is authorized.

## Token-budget packing checkpoint

`tracemem.token_budget.pack_context_by_tokens` now packs whole ranked
citation/evidence lines using a caller-supplied counter for the complete final
prompt. It measures the empty prompt and each complete proposed context, skips
over-budget hits while trying later hits, and returns the exact selected hits.
It has no tokenizer integration, does not establish actual model token usage,
and makes no matched-token QA claim. The packer documentation describes the
counter contract and privacy boundary; the validation record preserves the
source hashes, behavioral RED/GREEN chronology, root checks, and installed
replay.

This is an offline implementation result only. Private source text stays local;
remote token counting remains limited to reviewed synthetic-only inputs under
current authorization. Matched-token QA has not run, and this checkpoint does
not authorize inference.

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

Root maintains [`.local/active-cycle.json`](../.local/active-cycle.json) with
exact live delivery, job, and next-action state. Keep private artifacts under
ignored `.local` with restrictive permissions; recheck live jobs before work.
