# Bounded development QA answering runner

This runner implements the [frozen source-only payload](qa-contrast-requests.md).
It is an answering diagnostic over six fictional development scenarios. It
does not select an extractor, satisfy Gate A, or authorize training.

The bounded call answers 46 unique complete prompts. The local audit maps
those answers back to 72 mode/budget cells. Annotations and scoring metadata
stay outside the remote payload. The total measured input is 23,533 tokens;
the output limit is 128 tokens per prompt, or at most 5,888 generated tokens.
Repeated cells share an answer and are correlated observations.

The runner verifies exact payload/source hashes, pinned model revision,
chat template, tokenizer backend, package versions and all 46 measured token
counts. Tokenizer checks finish before loading the model or generating
answers. Complete prompts are never truncated. The model uses the frozen
float16/eager/greedy configuration with thinking disabled.

The resource envelope is one T4, two CPU cores and 16 GiB host memory, with
one container, no retries, a 300-second startup timeout and a 600-second
function timeout. These are limits, not a billed-cost estimate. The runner
verifies Raj's personal Modal workspace immediately before launch. Root
must independently confirm no active job or worker can duplicate the call.

The remote function receives only the reviewed fictional payload and binding
hashes. It runs with source inclusion disabled, with no local source mounts,
private corpus, labels, vectors, claims, checkpoints or credentials attached.
Model weights and inference remain remote. Historical runners and receipts
stay unchanged.

The complete raw response is saved in a new mode-0600 `OUTPUT.remote.json`,
flushed and fsynced before app teardown or result validation. A teardown or
validation failure preserves that response. `OUTPUT` contains the validated
receipt or a bounded, sanitized failure envelope. There is no automatic retry.

The offline contract lives in `experiments/qa_contrast_answer_receipt.py`; the
launcher pins its exact source bytes before delegating. The generation protocol
hash covers the canonical payload's model settings, rather than the model's
architecture configuration. The receipt records generated token IDs and
distinguishes EOS at token 128 from a non-EOS token limit.

Offline inspection requires no model downloads or Modal imports:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.modal_qa_contrast_answer
```

Launch is opt-in with `--launch`, both exact `--approve-payload-sha256` and
`--approve-runner-sha256` values, and a new `--output` path. Python 3.12 is
required. The root review must also verify the personal workspace and absence
of active jobs immediately before launching. Flags bind reviewed bytes; they
do not substitute for those operational checks.

Root reviewed both source modules, all focused tests and installed-CI copies.
Full checks passed 514 tests (one skip), Ruff on 113 files and mypy on 32
runtime files. All 12 installed tests pass outside the checkout; all 37 runtime
members match the existing wheel. Offline source and copied-launcher summaries
match with model and Modal imports blocked. Private delivery is pending; no
QA answer or resource result for this payload has been measured. See the
[validation receipt](validation-qa-contrast-answer-runner.json) for exact state.
