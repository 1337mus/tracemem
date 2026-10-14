# Synthetic QA Modal runner

The runner rebuilds the reviewed 72-request fictional QA payload from the fixed
local fixture and preparer. It carries only the request IDs and system/user
messages needed for inference; answer references and scoring data stay local.
Its default mode is offline and does not inspect Modal credentials, build an app,
or import Torch or Transformers.

## Offline preparation

From the repository root, print a digest summary without showing the payload:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.modal_synthetic_qa
```

Use `--show-payload` only to inspect the fixed fictional request payload. The
payload SHA256 is pinned to
`80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac`; changes
to the fixture, preparer, prompt, request bundle, or canonical payload cause the
builder to refuse. The CLI reports the current runner-file SHA256 for review.

## Launch gate

No inference call was made while implementing or validating this runner. A
launch requires the explicit `--launch` flag, the exact reviewed payload hash,
the SHA256 of the exact runner file under review, and a path that does not
already exist. The runner verifies these inputs before checking the personal
Modal workspace or creating an app. The existing workspace verifier runs only
after both approvals pass. A successful call is made once, with no retry.

The flags are `--approve-payload-sha256`, `--approve-runner-sha256`, and
`--output NEW_PATH`. The runner hash is provenance for the reviewed local source;
it is not a remote attestation. Root retains responsibility for checking the
workspace, active jobs, exact payload/source hashes, and launch intent before
authorizing a call.

The remote body is registered by value with no source mount and validates its
wrapper, canonical payload hash, schema, generation config, exact request IDs,
and message format before importing inference libraries. It uses the pinned
Qwen3-4B revision with float16, greedy decoding, at most 2048 prompt tokens and
128 new tokens. All 72 prompts are tokenized before model loading; any over-limit
prompt aborts without truncation. Modal resources are bounded to one T4 call,
two CPUs, 16 GiB RAM, and a 600-second timeout, with retries disabled. These
limits bound runtime exposure and do not assert a provider-enforced dollar cap.

## Receipt

The output must be a new local path. It is created exclusively with mode `0600`,
so an existing file or symlink is refused rather than overwritten. The success
envelope contains:

```json
{
  "schema_version": 1,
  "payload_sha256": "…",
  "runner_sha256": "…",
  "client_elapsed_seconds": 0.0,
  "result": {}
}
```

`result` includes the raw output and input/generated token counts for each
request, EOS-versus-length stop information, model/config/template provenance,
runtime and GPU details, timing, peak allocated GPU memory, and the Modal call
ID. Failures are written in the same envelope with a sanitized `error` object
instead of `result`, then re-raised. The runner does not score answers.

## Validation record

Focused tests use the real fixed payload builder and mocked inference/launch
seams. They verify approval gates, fresh-output safety, by-value SDK
serialization, validation-before-model-load, all-request tokenization, token
limits, stop reasons, and sanitized failure receipts. No model, tokenizer, GPU,
network request, Modal app, or live launch was used.

The saved behavior logs under `.local/synthetic-qa-modal-worker-` preserve the
RED/GREEN sequence, with an important chronology limit: the full initial remote
body was implemented before its detailed per-behavior tests, so those remote
detail checks are postimplementation evidence rather than a blanket TDD claim.
The first payload-builder RED is a harness error: it ran
without `PYTHONPATH=src`, so the project package could not be imported. It is
preserved as history and is not evidence of a failing baseline; the later
`green-payload-builder` run is the valid result. Several early SDK/offline
attempts are similarly labeled `harness-invalid`. The `postimpl-remote-behaviors`
log caught early non-EOS termination being reported as a length stop; the
subsequent correction has failure/pass evidence in `postimpl-remote-behaviors`
and `postfix-remote-behaviors`. The final focused test,
Ruff check, Ruff format check, and strict mypy outputs are in the corresponding
`final-*` worker logs. Their commands used `PYTHONPATH=src` for unittest and
`MYPYPATH=src .venv/bin/mypy --follow-imports=silent` for strict checking of both
the runner and its tests. Imported experiment modules and optional inference
packages are outside that focused check.
