# Source-free Modal QA controls answer runner

This runner prepares and, only after explicit hash approvals, can execute the
frozen 36-prompt fictional QA-controls request. It is a descriptive diagnostic,
not a selected model, scorer, or Gate A result. One reviewed, fictional call is
complete; its aggregate receipt and limits are in the
[dev-only results](qa-controls-answer-run-results.md).

## Request and limits

The source-only request binds the 36 prompt messages, order, and settings by
request SHA-256
`df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871` and prepared
payload SHA-256
`36a03f56b69500674feb219fc40f8479965982bc9041de7a74b4464cace5bed1`. The
request uses the same pinned Qwen3-4B revision and generation settings as the
[legacy QA runner](qa-contrast-answer-runner.md). It contains no answer key or
measured input-token counts.

Each prompt is limited to 1,024 input tokens and 128 generated tokens. The
36,864 input and 4,608 generated totals are upper bounds only. The remote body
tokenizes the complete messages before model load and rejects over-limit input;
it does not truncate. Generation is greedy and capped at 128 tokens. Receipt
validation accepts EOS-only, EOS-at-cap and cap-without-EOS stop boundaries.

## Source-free execution boundary

The entry point is `experiments.modal_qa_interpretation_answer`; the offline
contract lives in `experiments.qa_interpretation_answer_receipt`. By default it
prints offline request metadata and confirms `remote_started: false`:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_qa_interpretation_answer
```

`--show-request` prints the fixed request offline. Launch mode requires
`--launch`, exact `--approve-request-sha256` and `--approve-runner-sha256` values,
and a new `--output` path. It requires local Python 3.12 and verifies the active
personal Modal workspace before building the remote function. No automatic
retry is configured.

The remote function receives only the reviewed request and its request/runner
hashes. Modal source inclusion is disabled; the isolated function body has no
repository imports or free variables. Its envelope is one T4, two CPUs, 16 GiB
memory, at most one single-use container, no retries, a 300-second startup limit
and 600-second function timeout. These are resource limits, not a cost estimate.

The local receipt and `OUTPUT.remote.json` raw-return file are created
exclusively with mode 0600. The raw return is flushed and fsynced inside the app
context before teardown and validation. Existing paths are not overwritten; a
failure leaves the raw return where available and writes a bounded failure
envelope. Receipts include reported input/generated token IDs, counts, stop
reason and runtime provenance. Validation cannot prove remote execution,
correct tokenization or decoding, or semantic answer quality.

## Verification and limits

Root independently matched all 36 request items, rejected seven altered requests
before model imports, and checked four fake call modes: success, teardown,
validation failure and remote failure. A separate Modal 1.6.1 offline
serialization audit round-tripped the isolated function body; this does not
verify a live remote call. For the completed run, the local machine loaded no
model or tokenizer. Installed replay passed 13 tests with no failures, errors or
skips; all 37 runtime members match the existing wheel.

The runner is 530 physical lines, 480 nonblank lines and 475 production lines
excluding comments/docstrings; root accepted the production-code count under the
~500-line guidance. TDD logs record missing API/function REDs, but several
regression assertions followed implementation. Strict one-RED-per-behavior is
not claimed; environment/PYTHONPATH setup failures are not behavioral REDs.
Focused tests and the final source full check pass. The runner is delivered in
signed private commit [`7c5bf12`](https://github.com/1337mus/tracemem/commit/7c5bf12f08eee3594a4173c9cac65f3747d79a36);
exact [CI run 38683345935](https://github.com/1337mus/tracemem/actions/runs/38683345935)
passed 568 source tests (one skipped), 315 installed tests (two skipped), Ruff,
mypy and wheel build. The [validation snapshot](validation-qa-controls-answer-runner.json)
preserves pre-delivery checks and has not been rewritten to include live
inference. See the separate [run results](qa-controls-answer-run-results.md).

There are zero human labels; Gate A remains unmet. No model is selected or
trained. Weights and tokenizers stayed off the Mac. The completed run used one
remote GPU call, and the app stopped with zero tasks. Any further launch
requires review of the exact request and runner hashes plus a fresh personal
Modal workspace and inventory check; it is limited to one GPU call.

The offline [analysis](qa-controls-answer-run-results.md) uses the tracked
validated receipt and pinned fixture references. It performs literal matching
after casefolding and whitespace collapse only; it is not a semantic evaluator.
