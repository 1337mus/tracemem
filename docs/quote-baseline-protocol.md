# Quote-only prompt baseline protocol

Status: planned dev-only prompt diagnosis before considering a fifth SFT.

## Comparison

Run `quote-v2` and `quote-coverage-v1` on the same 12 fictional quote-v4
development conversations, with identical source messages per pair. Each model
receives 24 prompts (12 cases × 2 variants); targets and review labels stay local.

| Model | Pinned revision |
| --- | --- |
| `Qwen/Qwen3-4B` | `1cfa9a7208912126459214e8b04321603b3df60c` |
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |

Use BF16 weights, greedy decoding, thinking disabled, a 1,024-token output cap, and a 4,096-token prompt limit. The source-free Modal call uses one A10G and
`serialized=True` to upload a serialized function without repository source.
It allows at most one container, disables retries, and has a 2,400-second
timeout. Make one model call at a time; confirm the prior call has stopped before starting the other.

## Preserve and review

Keep each raw receipt with exact item and case IDs, prompt version, raw output
text, full input/output token ID arrays and counts, final token ID, EOS and
finish status, model and input hashes, chat-template hash, tokenizer backend,
package versions, and GPU/model-load/remote/client timings.
Review structure, exact citation integrity, and provisional semantic support
separately. Compare actor and nested-reporter attribution, selected-fact
coverage, and abstention by model and prompt. Semantic labels are provisional;
this dev-only comparison has no human review and establishes no held-out
quality or generalization claim.

The quote-v4 test is excluded and must not guide prompt selection or other
tuning. Development results are adaptive. The historical SFT runs used 4-bit
weights, while this prompt comparison uses BF16; it cannot isolate prompt
effects as a controlled SFT ablation. Any later test must use fresh cases.

## Prepare and launch

Prepare each model locally. Preparation prints payload and runner hashes and
does not start a remote call:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_quote_baseline --model qwen3-4b
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_quote_baseline --model qwen3-8b
```

After local tests pass, copy the exact hashes from the matching preparation output
into the placeholders below. Use a new exclusive receipt path. Launch one model,
wait for its call to stop and verify zero active tasks, then prepare/launch the
other model with its own hashes and receipt path:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_quote_baseline \
  --model qwen3-4b --launch \
  --approve-payload-sha256 PAYLOAD_SHA256_FROM_PREPARATION \
  --approve-runner-sha256 RUNNER_SHA256_FROM_PREPARATION \
  --output .local/quote-baseline-run-20261016/qwen3-4b-v1.json
# Repeat only after the first call is stopped, using qwen3-8b's hashes.
PYTHONPATH=src:. .venv/bin/python -m experiments.modal_quote_baseline \
  --model qwen3-8b --launch \
  --approve-payload-sha256 PAYLOAD_SHA256_FROM_PREPARATION \
  --approve-runner-sha256 RUNNER_SHA256_FROM_PREPARATION \
  --output .local/quote-baseline-run-20261016/qwen3-8b-v1.json
```
