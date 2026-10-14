# Qwen3-4B synthetic QA diagnostic results

This is the completed provisional agent-only run over the eight fictional QA
questions. It is descriptive diagnostic evidence, not a model evaluation. The
receipt, complete analysis, and split-only JSON artifacts are preserved at
[`examples/synthetic/qwen3-4b-qa-diagnostic-remote.json`](../examples/synthetic/qwen3-4b-qa-diagnostic-remote.json),
[`synthetic-qa-qwen3-analysis.json`](synthetic-qa-qwen3-analysis.json),
[`synthetic-qa-qwen3-dev.json`](synthetic-qa-qwen3-dev.json), and
[`synthetic-qa-qwen3-test.json`](synthetic-qa-qwen3-test.json).

## Dev split

Each row contains the same four dev questions under one retrieval-mode and
UTF-8 byte-ceiling cell. Agreement is literal agreement after the analyzer's
whitespace/casefold normalization: all questions, answerable questions, and the
single unknown question. Input token counts show per-cell min–max and sum;
generated token sums include EOS.

| Mode | Byte ceiling | Literal agreement all / answerable / unknown | Input tokens (min–max; sum) | Generated tokens (sum) | EOS / 4 | Zero-context UNKNOWN / 4 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| BM25 | 0 | 1/4 · 0/3 · 1/1 | 78–93; 336 | 8 | 4 | 4 |
| BM25 | 512 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |
| BM25 | 2048 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |
| Vector | 0 | 1/4 · 0/3 · 1/1 | 78–93; 336 | 8 | 4 | 4 |
| Vector | 512 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |
| Vector | 2048 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |
| Hybrid | 0 | 1/4 · 0/3 · 1/1 | 78–93; 336 | 8 | 4 | 4 |
| Hybrid | 512 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |
| Hybrid | 2048 | 1/4 · 0/3 · 1/1 | 226–377; 1,340 | 35 | 4 | — |

## Test split

| Mode | Byte ceiling | Literal agreement all / answerable / unknown | Input tokens (min–max; sum) | Generated tokens (sum) | EOS / 4 | Zero-context UNKNOWN / 4 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| BM25 | 0 | 1/4 · 0/3 · 1/1 | 79–85; 326 | 8 | 4 | 4 |
| BM25 | 512 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |
| BM25 | 2048 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |
| Vector | 0 | 1/4 · 0/3 · 1/1 | 79–85; 326 | 8 | 4 | 4 |
| Vector | 512 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |
| Vector | 2048 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |
| Hybrid | 0 | 1/4 · 0/3 · 1/1 | 79–85; 326 | 8 | 4 | 4 |
| Hybrid | 512 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |
| Hybrid | 2048 | 1/4 · 0/3 · 1/1 | 227–384; 1,342 | 63 | 4 | — |

The dash means zero-context UNKNOWN is not scored in that nonzero-context cell.
Every cell has four EOS finishes, zero length stops, zero other finishes, zero
empty outputs, zero truncations, and zero outputs at the `max_new_tokens` limit.
There is no pooled dev/test score or ranking.

## Run, resources, and provenance

The single corrected-source call used `Qwen/Qwen3-4B` revision
`1cfa9a7208912126459214e8b04321603b3df60c`, Python 3.12.10, PyTorch 2.7.1,
Transformers 4.51.3, Accelerate 1.6.0, CUDA 12.6, and a Tesla T4 (compute
capability 7.5). It processed 72 requests: 18,078 input tokens (range 78–384)
and 636 generated tokens including EOS. GPU compute was 54.679459s, model load
was 76.35161s, remote function elapsed time was 137.175618s, and peak allocated
GPU memory was 8,822,848,512 bytes. The Modal app stopped with zero tasks; no
GPU jobs remain active.

- Source commit: `814c5ca09aa251f6d06c918adf0e621a35dd888a`; CI
  [38648259693](https://github.com/1337mus/tracemem/actions/runs/38648259693)
  passed (447 tests, one optional skip; 195 installed tests, two skips; Ruff 96
  files; source mypy 31 files).
- Raw receipt file SHA256:
  `1da725c57bf858a6cf205bb300122616904520ad2966ff212f84605ca25d0738`.
- Full analysis SHA256:
  `0b3932af0f140f16fd98e7d01d60f94072162c01bba312173f32a6c4886af89f`.
- Payload SHA256:
  `80e058c629de8f8a06ca4caa8f1093a8308a299c976b018ab7e8c5c62493b2ac`.
- Model config SHA256:
  `bab03d3f86877737d8c4a04eb4fd5fddb7b480a449527d80147eb4739d9f3f15`.
- Tokenizer template SHA256:
  `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`.
- Analyzer source SHA256:
  `273c2c38000afe8186ba9e1f38148ad481c049ae9d72a8532108bf3b55fdabe3`.
- Request bundle SHA256:
  `85d1334d0401ef2bdc920bf316f066bc3e74e9423ee5bec0aa63f4a559be2084`;
  reference bundle SHA256:
  `70769ec36ac4c1e6a226d7432ea88241225e35d91d803cdc0a678fe2589226d2`.

Root replay at `2026-10-14T22:35:31.834181Z` independently checked the 18
cells, all 72 literal outcomes, and token/empty/finish counts. It also found
that each question's nonzero rendered context was identical across retrieval
modes and the 512/2048-byte ceilings. The replay receipt is
`.local/synthetic-qa-real-run-root-replay.json` (SHA256
`56f8339d4d982f92b4c802adc22013feebcee6f74ecd7bdd8db15592c1414c53`). A second
reproduction used the existing installed wheel outside the checkout, with
`PYTHONPATH` and `PYTHONHOME` unset and no SDK, model, or network calls; it
reproduced the exact analysis SHA256 above. Its receipt is
`.local/synthetic-qa-real-run-installed-replay.json` (SHA256
`c1ac68c65be24edb6cfd2b47311feaaa2d0c07837848aea8cecd7dc259cc7f93`).

## Offline reproduction

From the repository root, run the frozen analyzer against the copied receipt.
This makes no model or network calls. The output path must not already exist.

```sh
PYTHONPATH=src .venv/bin/python -m experiments.analyze_synthetic_qa \
  --receipt examples/synthetic/qwen3-4b-qa-diagnostic-remote.json \
  --approve-runner-sha256 1fa6881e03b8f501cc127de97b29576bc70561b777b69f5359d381b2a1ca968c \
  --approve-template-sha256 a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8 \
  --output .local/synthetic-qa-qwen3-reproduced.json
```

## Limitations

- Literal matching only: the analyzer casefolds and collapses whitespace, but
  does not normalize punctuation, aliases, answer spans, or semantic
  equivalence. Examples include `teal` vs `Teal.`, `bassoon` vs a full-sentence
  answer, and reference `No` vs output `UNKNOWN`.
- Root replay found identical nonzero rendered context for each question across
  all retrieval modes and both nonzero byte ceilings. This run cannot compare
  retrieval modes or context capacity.
- The byte ceilings are UTF-8 byte limits, not matched tokenizer budgets.
- Each split has four questions, three answerable and one unknown. The same
  questions repeat across retrieval modes and ceilings; the cells are not
  independent scenarios. Do not combine the splits into a pooled metric.
- References are provisional agent-authored and were not independently reviewed
  by human annotators. Gate A remains unmet.
- Do not rescore, claim semantic accuracy, salvage test performance, infer
  model superiority/calibration/cost per correct answer, call this
  training-ready, or treat it as evidence of benchmark compatibility.
- This completed run authorizes no further inference. The next useful work is
  generic offline token-budget packing.
