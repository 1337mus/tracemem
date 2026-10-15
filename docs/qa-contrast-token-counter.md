# Development contrast tokenizer counter

`experiments/modal_qa_contrast_token_count.py` is a separately frozen counter
for the six fictional development cases described in
[QA contrast preparation](qa-contrast-preparation.md). It preserves the
historical 28-prompt runner and receipts unchanged. All references remain
provisional agent-authored labels; there is no independent human evaluation.

The runner checks the fixture's raw hash before parsing, regenerates the
source-only catalog and requires exactly 399 complete system/user prompts.
References, authored evidence labels and audit metadata stay outside the
payload. Ordinary source passages can naturally contain answers.

| Frozen input | SHA-256 |
| --- | --- |
| Fictional fixture | `b4fb6ea6242c294574ca9ed15549f5b524a77d3154b4c87b9831d2f357beefed` |
| Canonical catalog | `2c18978bea31ba5c4e77d20fa6103a2a7a182bb93c7d94dd5c257b029e9662cf` |
| Canonical 534,629-byte payload | `7f3055ef98fec05f77676b3b995b27c6c6ebcfc2566fdd668177f065a3d9fcc2` |

The tokenizer is Qwen/Qwen3-4B revision
`1cfa9a7208912126459214e8b04321603b3df60c`, using Transformers 4.51.3,
tokenizers 0.21.1 and Jinja2 3.1.6. Its chat-template SHA-256 must be
`a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
Counting includes the complete messages and generation prompt, disables
thinking, and does not truncate. Counts apply to this exact configuration.

The default command is offline and prints reviewed payload and current runner
hashes. It does not import Modal or tokenizer packages:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_qa_contrast_token_count
```

`--show-payload` prints the fictional payload without launching. A launch
requires `--launch`, both exact reviewed SHA-256 approvals and a fresh
`--output` path. Local Python 3.12 is required for launching. The runner
reserves both the validated receipt and its `.remote.json` sibling with
exclusive creation, no-follow and mode `0600` before checking the personal
Modal workspace or constructing the function. A collision at the second
path can leave an empty first file; no remote call occurs and neither file
is removed or overwritten.

The isolated remote function has no repository globals or closure. It
validates the exact request hash, configuration and every prompt ID before
loading the tokenizer. Modal uses Python 3.12, one CPU container, no GPU,
no mounts, secrets or forwarded environment, no retries, no detached run,
and `include_source=False`. CPU allocation is 1–2 cores, memory 2–4 GiB,
startup timeout 300 seconds and function timeout 600 seconds. No model
weights, generation or inference are needed.

The raw JSON-serializable returned object is flushed and fsynced before
teardown and validation. This diagnostic artifact may contain nonstandard
JSON values such as NaN when a result is malformed; those values are rejected
by the validated receipt. Exceptions leave sanitized failure receipts and
are not retried. A transport failure can leave the reserved raw file empty.
Unserializable results cannot be retained as JSON. Validated counts require
exact ID coverage, nonnegative integers and the frozen tokenizer provenance.

Offline fake-tokenizer tests exercise transport, validation and file behavior;
their fixture integers are not measured token counts. A real counting call
requires separate root review of the exact runner and payload, a fresh
personal workspace and job check, and retention of both raw and validated
receipts. Token-budget evidence retention and QA answers remain separate
experiments; counting alone does not establish QA accuracy or Gate A.

## Measured run

One root-reviewed source-free CPU-only Modal call counted all 399 prompts.
Complete prompts ranged from **82 to 907 tokens**, with median **553**.
The remote tokenizer work took 4.971 seconds; this is not end-to-end query
latency or a billing measurement. No GPU or model inference ran. The app
stopped with zero tasks. Its backend tokenizer fingerprint matched the
historical pinned 28-prompt counter.

Root independently checked the exact prompt identities, literal provenance,
integer counts and raw/validated equality. Eight tests passed against the
installed wheel outside the checkout, its payload was byte-identical to the
root freeze, and an installed replay accepted all 399 measured counts without
importing a tokenizer. All 37 installed runtime members remain unchanged.
Full checks passed 487 tests (one optional skip), Ruff on 106 files and strict
mypy on 32 runtime files. Root caught and corrected raw-response loss on
teardown failure before the real call; its regression and the tokenizer
rejection controls pass.

The [fictional measured receipt](../examples/synthetic/qwen3-4b-qa-contrast-token-count-remote.json)
and [validation record](validation-qa-contrast-token-counter.json) preserve
the exact hashes and configuration. Counts cover correlated subsets of six
development cases, not 399 independent examples. They do not yet measure
evidence packing at the configured token ceilings, matched-budget QA, model
selection or training value. All original labels remain provisional, with
zero independent human labels and Gate A unmet.
