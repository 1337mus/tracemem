# Offline QA controls answer contract

This is the next offline step after the [frozen QA controls](qa-interpretation-controls.md).
It binds a source-only request and describes an offline receipt validator; it
does not launch a model or score answers.

## Frozen request

The request covers 36 fictional development prompts from the reviewed controls
payload. Its canonical request SHA-256 is
`df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871`; the bound
prepared-payload SHA-256 is
`36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1`.
The model is Qwen3-4B with the same revision and generation settings as the
[legacy QA runner](qa-contrast-answer-runner.md). This request contains no
measured input-token counts.

Each of the 36 prompts is bounded at 1,024 input tokens and 128 generated
tokens. The aggregate ceilings are 36,864 input tokens and 4,608 generated
tokens; these are upper bounds only, not measured usage. A future launcher must
reject over-limit prompts without truncation.

## Offline receipt contract

The source API is `build_request`, `request_sha256`, `validate_request` and
`validate_result` in `experiments/qa_interpretation_answer_receipt.py`. The
offline contract accepts a complete receipt with input and generated token IDs
plus structural and runtime provenance. It checks the reported contract; it
cannot establish that a model actually ran, that tokenization or decoding was
correct, or that an answer is semantically right. All receipt fixtures and
contract tests are explicitly fake.

There is no launcher, answer scorer, GPU call or inference in this slice.
Model weights and tokenizers stay off the Mac. There are zero human labels;
Gate A remains unmet, and no model is selected or trained. Future launch work
requires root review of exact source and payload bytes, followed by a fresh
check of the active personal Modal workspace and inventory. Limit any later
authorized run to one GPU call.

Root reconstructed all 36 request items and accepted an independent fake
receipt. Three stop-boundary cases passed (EOS only, EOS at the cap, cap without
EOS); 61 invalid mutations were rejected with Modal, Torch, Transformers and
Tokenizers imports blocked. The final full check passed 556 tests (555 passed,
one skipped), Ruff on 123 files and mypy on 32. Installed replay passed 13 tests;
all 37 runtime members match the existing wheel. No network, wheel build or
inference occurred.

TDD chronology includes observed missing-module/API, request, deepcopy and
runtime-string failures. Many rejection rules existed before grouped negative
regression tests were added, so this does not claim a strict one-RED-per-behavior
cycle. Source review and local validation are complete; signed private delivery
and exact CI for this contract are still pending.

The preceding controls preparation was delivered privately with valid GitHub
signature and exact [CI run 38679061711](https://github.com/1337mus/tracemem/actions/runs/38679061711):
543 source tests attempted (542 passed, one skipped) and 291 installed tests
attempted (289 passed, two skipped). That validates only the earlier
preparation. New source/test/CI hashes and receipts are in the
[validation record](validation-qa-controls-answer-contract.json).
