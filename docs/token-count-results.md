# Synthetic token-count results

Updated 2026-10-14. One source-free, CPU-only Modal call counted 28 complete
fictional system/user prompts with the pinned Qwen tokenizer. Counts ranged
from 78 to 384 tokens. The call used no GPU, model weights, generation,
inference, or QA requests. Root replay independently checked the returned
receipt and the downstream packing summaries.

## Measurement provenance

| Field | Measured value |
| --- | --- |
| Tokenizer | `Qwen/Qwen3-4B` revision `1cfa9a7208912126459214e8b04321603b3df60c` |
| Runtime | Python 3.12.10; Transformers 4.51.3; tokenizers 0.21.1; Jinja2 3.1.6 |
| Tokenizer mode | Fast tokenizer; `trust_remote_code=false` |
| Chat template SHA-256 | `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8` |
| Template options | `tokenize=true`, `add_generation_prompt=true`, `enable_thinking=false` |
| Backend tokenizer SHA-256 | `a63080b4b30dc861fe1b0e420c53a241304dc81aff3d9bb315e6e33e2a1ab794` |
| Catalog SHA-256 | `85682edbcdd728103e6540c86106de953f400e217a9341ffbd82420d0429e478` |
| Payload SHA-256 | `125476f3d93363244fe0f6c5c51c21203f409a7dbf24ba397b25be469435cc88` |
| Runner SHA-256 | `9d8acf587422248275cd71376378a84033c0332d3b08da789ec1b0d2cbabaa5a` |
| Remote body elapsed time | 3.6516 seconds, including tokenizer load/init and any download |
| Client elapsed time | 24.262924 seconds |

The raw receipt is [qwen3-4b-token-count-remote.json](../examples/synthetic/qwen3-4b-token-count-remote.json), SHA-256
`5f4b8cd5c5215b8a82c4cccd610f57e32bcedc20586379a10f73611d0124a0f8`.
The counter ID is `Qwen/Qwen3-4B@1cfa9a7208912126459214e8b04321603b3df60c+transformers-4.51.3+template-a55ee1b1`.

## Historical cross-checks

The measured receipt matched all 72 historical QA `input_tokens` entries that
overlap the counted prompts, covering 16 unique prompt IDs. Their input-token
sum was 18,078, with zero mismatches. This is a consistency check against prior
receipts; it made no new QA or model calls.

## Development packing replay

The development split contains 36 packing cells, 12 at each complete prompt
ceiling.

| Complete prompt ceiling | Cells | Prompt-token range | Prompt-token sum | Selected citations | Empty-context cells |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12 | 78–93 | 1,008 | 0 | 12 |
| 256 | 12 | 221–244 | 2,772 | 12 | 0 |
| 512 | 12 | 226–377 | 4,020 | 21 | 0 |

The durable split report is [token-count-dev.json](token-count-dev.json).

## Test packing replay

The test split contains 36 packing cells, 12 at each complete prompt ceiling.

| Complete prompt ceiling | Cells | Prompt-token range | Prompt-token sum | Selected citations | Empty-context cells |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 12 | 79–85 | 978 | 0 | 12 |
| 256 | 12 | 222–241 | 2,745 | 12 | 0 |
| 512 | 12 | 227–384 | 4,026 | 21 | 0 |

The durable split report is [token-count-test.json](token-count-test.json).
An installed-package replay reproduced both split files byte-for-byte and
rechecked all 72 historical counts without network or model calls.

These packing reports use the generic packer interface, whose count origin is
caller-supplied and whose `tokenizer_attested` field is false. The separate
root-reviewed measurement receipt records the tokenizer provenance above; it
does not make the generic packer itself an attestation mechanism.

## Offline reproduction

This replays the saved measurement without a tokenizer download or remote call:

```sh
PYTHONPATH=src:. .venv/bin/python - <<'PYTHON'
import json
from pathlib import Path
from experiments.token_catalog import pack_from_counts

raw = json.loads(Path("examples/synthetic/qwen3-4b-token-count-remote.json").read_text())
packed = pack_from_counts(raw["result"]["count_receipt"], (128, 256, 512))
print({split: len(rows) for split, rows in packed["packs_by_split"].items()})
PYTHON
```

Expected output is `{'dev': 36, 'test': 36}`. The generic replay API validates
catalog binding and count coverage; it does not verify tokenizer provenance.

## Validation and limits

The signed source commit is `efa9434052d5b927a101fc119044ec6dc5571471`.
Root's full check passed 468 tests with one skip in 15.848 seconds, Ruff on 102
files, source mypy on 32 files, and all eight installed focused tests. The 37
installed runtime files remained byte-identical. A real Modal serializer
roundtrip and isolated-body fake-tokenizer test passed; neither test made a
remote call. The Modal app `ap-NiLyZ3rdVPfd9Nx4kkn3Eb` is stopped with zero
tasks. See [runner validation](validation-modal-token-count.json) and the
[root replay validation](token-count-replay-validation.json). The count scope
and acceptance checks are in [token-count-protocol.json](token-count-protocol.json).

The signed source was delivered to private `1337mus/tracemem`; [source CI](https://github.com/1337mus/tracemem/actions/runs/38655447870) passed with 468 source tests (one skip) and 216 installed tests (two skips). The 72 packing
outcomes repeat the same 12 questions across retrieval modes and ceilings, so
they are correlated fixture outcomes, not independent statistical observations.
The fixture establishes token counts for these prompts, not semantic accuracy,
retrieval quality, model selection, or training value. Matched-token QA and
independent human labels remain absent, and Gate A remains unmet. No QA
rescoring, matched-token QA, model-selection, or training claim follows from
this replay.
