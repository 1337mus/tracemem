# QA controls answer run results

This is the dev-only record for one reviewed call over the frozen fictional
36-prompt QA-controls request. The six fixture groups are correlated. This run
does not establish semantic truth, answer quality, held-out performance, or
progress through Gate A.

## Launch and artifacts

The runner was delivered in signed private commit
[`7c5bf12`](https://github.com/1337mus/tracemem/commit/7c5bf12f08eee3594a4173c9cac65f3747d79a36)
and exact [CI run 38683345935](https://github.com/1337mus/tracemem/actions/runs/38683345935)
passed. The reviewed request SHA-256 is
`df91a152242ed6b2942802b5957dc7531a58bad3bffce67a284ce4526f90b871`; the
runner SHA-256 is
`9ee4f7fbb440d166eff415c978239ca71da69a6e607fe63f98cc47fcf9a8568c`.

An initial local launcher invocation with `PYTHONPATH=src` failed to import
`experiments` before Modal workspace checks and made no remote call. The
corrected command below was executed once:

```sh
PYTHONPATH=src:. .venv/bin/python .local/qa-controls-producer-launch.py
```

There was one remote call and no automatic retry. It sent only the reviewed
fictional request and hash metadata; private source data and references were not
included. The isolated function ran on one Tesla T4. The client call elapsed
98.164 seconds; the remote function reported 91.041 seconds. The final Modal
inventory at 2026-10-15 04:58:40 UTC showed the app stopped with zero tasks.

The ignored mode-0600 local artifacts were copied byte-for-byte into tracked
synthetic fixture paths after root verified their lineage:

- Raw remote return: `examples/synthetic/qwen3-4b-qa-controls-answer-raw.json`
  (file-byte SHA-256 `3112fb466073e6e7ae42f916b957e7d98db7a9d170be293581cac6518b690504`).
- Validated receipt wrapper: `examples/synthetic/qwen3-4b-qa-controls-answer-receipt.json`
  (file-byte SHA-256 `5ee39bc640d5c767858f7a2dfa631cafc871c703ccb3f844a8acae3347ec4fe7`).
- Independent aggregate audit: `.local/qa-controls-producer-inference-root-audit.json`.

## Observed output

The return contained all 36 ordered outputs; none were empty. The structurally
audited producer-reported counts were 6,993 input tokens total (80–365 per
prompt) and 196 generated tokens total. All 36 outputs reported EOS as their
stop reason; none reached the 128-token generation cap. The raw return and
validated result matched. No model or tokenizer was loaded on the local Mac;
the remote call performed inference on the T4.

These token counts come from the remote producer's reported token IDs and
counts, which the receipt validator checked structurally. No independent local
tokenization or decoding check was performed. The counts are observations for
this run, not an answer-quality metric.

## Interpretation limits

The literal diagnostic below is not a semantic answer-quality score; there are
no human labels. The 36 outputs are
not independent observations because prompts share six correlated fixture
groups. The run is a dev-only descriptive diagnostic, not held-out evidence,
human ground truth, model selection, or Gate A evidence. Gate A remains unmet.
Any separate answer analysis must preserve that boundary and report its scoring
method and denominators independently.

## Offline literal analysis

The tracked report is [qa-controls-answer-dev.json](qa-controls-answer-dev.json),
SHA-256 `fd6fc6933b0c83b5905181109393965f1dd6706ae25f85fc1d6ffd0421112665`.
To reproduce it without a remote call, use the pinned validated receipt and a
fresh, nonexistent output path (the writer uses exclusive creation):

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.analyze_qa_interpretation_answers \
  --receipt examples/synthetic/qwen3-4b-qa-controls-answer-receipt.json \
  --output /tmp/qa-controls-answer-dev-replay.json
```

The analyzer binds the request, runner, receipt, result and prepared reference
hashes. Its `receipt_sha256` and `result_sha256` are canonical-JSON hashes, not
the file-byte hashes listed above: `receipt_sha256` is
`19d6d08574dd5d87a449cdfff3acc14677e47257166bdbb191169f91eac0fa37`, and
`result_sha256` is
`127882535b04a8cf7a12a1ee12dd43dc894dde9cb92623ef9355895fe0a3654f`.
Normalization is casefolding followed by
whitespace collapse only; punctuation and added wording remain significant.
The exact-`UNKNOWN` column counts normalized outputs equal to `unknown`.

| Prompt variant | Prompt-label literal match | Full-source-label literal match | Exact `UNKNOWN` outputs |
| --- | ---: | ---: | ---: |
| `full_all_message_ids` | 2/12 | 2/12 | 3/12 |
| `support_omitted` | 12/12 | 2/12 | 12/12 |
| `empty` | 12/12 | 2/12 | 12/12 |
| **Total** | **26/36** | **6/36** | **27/36** |

The deterministic always-`UNKNOWN` baseline also scores 26/36 on prompt labels
and 6/36 on full-source labels. Equal literal totals do not imply equal semantic
quality. Root's review noted that nine full-context strings contain expected
information but fail exact comparison because of punctuation or added wording;
one answerable conditional-item question returned `UNKNOWN`. No semantic
accuracy is assigned to either observation. The report remains dev-only across
six correlated groups, with zero human labels and an empty test split.

Root independently matched all 36 rows and three variants to the frozen
expectations and rejected 16 malformed receipt/reference cases; the historical
literal report is unchanged. Seven focused tests passed. The final local source
check passed 575/576 tests (one optional Context Kit skip), Ruff on 127 files and
mypy on 32 runtime files. The analyzer is 369 physical lines, 334 nonblank and
330 production lines excluding comments/docstrings, within the reviewed 400-line
cap. Installed replay passed all seven focused tests without failures, errors or
skips; all 37 runtime members match the pinned wheel and source. The copied
report reproduced these exact 30,752 bytes outside the checkout, with real model
imports blocked and input receipts preserved. See the [analysis validation
snapshot](validation-qa-controls-answer-analysis.json). Signed private delivery
and exact-commit CI are recorded separately in the local active-cycle file.
