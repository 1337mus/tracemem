# Public Qwen3-4B cache

`experiments/prepare_public_model_cache.py` prepares a public-only cache for the pinned `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c`. It downloads only allowlisted config, tokenizer, and safetensors artifacts to the personal Modal volume `tracemem-public-model-cache-v1`. It accepts no dataset, private checkpoint, or user input and uploads no repository source. The Modal function uses one CPU, one container, a 600-second timeout, and zero retries.

By default the CLI is offline: it prints the fixed payload and exact source and payload SHA-256 values. Review both values and the current source before launching:

```sh
PYTHONPATH=src:. MODAL_PROFILE=reflex-personal .venv/bin/python -m experiments.prepare_public_model_cache
PYTHONPATH=src:. MODAL_PROFILE=reflex-personal .venv/bin/python -m experiments.prepare_public_model_cache \
  --launch \
  --approved-source-sha256 <reviewed-source-sha256> \
  --approved-payload-sha256 <reviewed-payload-sha256>
```

Launch requires Python 3.12, the active `reflex-personal` Modal workspace, and the existing fresh workspace and stopped-app checks. Each run writes an exclusive JSON receipt under `.local/public-model-cache/`; the directory is 0700 and receipts are 0600. A failed run keeps its receipt, and an existing receipt path is never overwritten. The remote result records file names, byte counts, SHA-256 values, and safe error types/statuses. It does not include exception messages or download URLs.

The focused offline contract tests run with:

```sh
PYTHONPATH=. python3 tests/test_public_model_cache.py
```
