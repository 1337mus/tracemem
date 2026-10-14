# TraceMem handoff

Updated 2026-10-14. This checkpoint records the completed synthetic QA run and
the completed CPU-only synthetic token-count measurement. Verify live GitHub
and worker state before any operational action by 2026-10-15. No new inference
is authorized by this handoff.

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
Catalog source and validation docs were delivered privately at
`403cc47edc54ae462466cec0522842dd6aceda01`. Catalog-specific
CI [38652529551](https://github.com/1337mus/tracemem/actions/runs/38652529551)
passed: 460 source tests (one skip), 208 installed tests (two skips), Ruff on
100 files, and source mypy on 32 files. Exact hashes, test chronology, the
preserved installed-harness failure, and remaining limitations are recorded in
[`validation-token-count-catalog.json`](validation-token-count-catalog.json).
The catalog prepares 28 synthetic prompt items from 84 subset instances and
replays caller-supplied counts; its invented test/oracle integers are not
measurements. A separate source-free CPU-only Modal call has now measured all
28 complete fictional prompts with the pinned Qwen tokenizer. Root replay
confirmed 72 historical input-token cross-checks over 16 unique prompts with
zero mismatches, and checked 72 packing outcomes split 36/36 across
dev and test. Details, provenance, and boundaries are in
[`token-count-results.md`](token-count-results.md); the durable raw and split
receipts are listed under [Durable evidence](#durable-evidence).

The count source is signed in local commit
`efa9434052d5b927a101fc119044ec6dc5571471` and was delivered privately.
[Source CI](https://github.com/1337mus/tracemem/actions/runs/38655447870)
passed with 468 source tests (one skip) and 216 installed tests (two skips). Root's full check passed 468 tests with one skip in 15.848s,
Ruff on 102 files, source mypy on 32 files, and eight installed focused tests.
All 37 runtime files remained byte-identical. A real Modal serialization
roundtrip and isolated-body fake-tokenizer test passed without a remote call.
The one actual tokenizer count call used CPU only and no GPU, model weights,
generation, inference, or QA calls. App `ap-NiLyZ3rdVPfd9Nx4kkn3Eb` is stopped
with zero tasks.

The raw measurement receipt and its separately generated dev/test packing
reports are durable. Before any source push, separately verify that the GitHub
repository is still PRIVATE. The measured run checked only the reviewed
fictional payload; it did not perform QA scoring or inference. The generic
packer still accepts caller-supplied, unattested count flags. Its split report
metadata does not cryptographically attest those inputs; the separately
reviewed measurement receipt supplies the tokenizer provenance. Do not load
model weights, run new QA inference, or train.

Phase 0/1 remain complete; see [`roadmap.md`](roadmap.md) and
[`context-kit-inventory.md`](context-kit-inventory.md). Context Kit at
`/Users/raj/Projects/ctxkit`, original conversations, and `~/.llm-context` remain
read-only. Keep model weights, tokenizers, and inference off this Mac. Reviewed
code, docs, and fictional fixtures may be signed and pushed only to private
`1337mus/tracemem`; assert the repository is still PRIVATE immediately before
each push. No public publication, private-data upload, or training is authorized.

## Next bounded work

A bounded worker is probing compatibility of the existing adapter/scanner with
the pinned public LongMemEval oracle file. Its public dataset-card MIT
declaration does not resolve constituent provenance. Keep downloaded benchmark
content and any record identifiers local; this check performs no model call,
QA scoring, training or redistribution. Read `.local/active-cycle.json` for
current ownership and exact receipts before continuing. Verify live worker
state by 2026-10-15; do not duplicate the probe or any remote job.

## Token-budget packing checkpoint

`tracemem.token_budget.pack_context_by_tokens` now packs whole ranked
citation/evidence lines using a caller-supplied counter for the complete final
prompt. It measures the empty prompt and each complete proposed context, skips
over-budget hits while trying later hits, and returns the exact selected hits.
The packer itself has no tokenizer integration and does not establish actual
model token usage. A separate reviewed receipt now supplies measured counts
for the fixed synthetic fixture; see the [token-count results](token-count-results.md).
The packer's count fields remain caller-supplied and are not attested by the
generic packer.

The packing implementation was validated offline. Private source text stayed
local; the remote count used authored fictional prompts only. Matched-token QA
has not run, and this checkpoint does not authorize new inference.

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
- Measured token-count report: [`token-count-results.md`](token-count-results.md).
- Raw tokenizer receipt: [`examples/synthetic/qwen3-4b-token-count-remote.json`](../examples/synthetic/qwen3-4b-token-count-remote.json)
  (SHA256 `5f4b8cd5c5215b8a82c4cccd610f57e32bcedc20586379a10f73611d0124a0f8`).
- Split packing reports: [`token-count-dev.json`](token-count-dev.json) (SHA256
  `42f28d9127c1ed0d815dff48395d5adb2499e79db303d3d7b55180e8e93f8e20`) and
  [`token-count-test.json`](token-count-test.json) (SHA256
  `36377991c30df8e0183e29334be3fd1671584874b6466b7668d4dc92de44a412`).
- Measurement and replay validation: [`validation-modal-token-count.json`](validation-modal-token-count.json)
  and [`token-count-replay-validation.json`](token-count-replay-validation.json).
- Correction, original failure receipt/hash, prior source-validation hashes,
  corrected run, root replay, cleanup, and CI facts:
  `docs/validation-synthetic-qa-inference.json`.

Root maintains [`.local/active-cycle.json`](../.local/active-cycle.json) with
exact live delivery, job, and next-action state. Keep private artifacts under
ignored `.local` with restrictive permissions; recheck live jobs before work.
