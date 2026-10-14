# Synthetic QA request preparation

`experiments.prepare_synthetic_qa.prepare_synthetic_qa()` builds a deterministic
request bundle and a separate provisional-reference bundle from the fixed
`examples/synthetic/qa-diagnostic-v1.json` fixture. The fixture contains eight
authored fictional questions: four `dev`, four `test`, six answerable, and two
`_abs` cases whose reference is exactly `UNKNOWN`. The answers and evidence labels
are author references; they have not received independent human review.

The default run creates 72 matched request cells: eight questions times three
retrieval modes (`bm25`, `vector`, `hybrid`) times three UTF-8 byte ceilings
(`0`, `512`, `2048`). Each question gets its own temporary evidence store. The
three mode rankings are computed once for that question, then each ranking is
packed at every requested ceiling. A zero-byte ceiling has an empty context.
Repeated ceilings are matched cells from the same eight questions, not additional
independent scenarios.

Request bundles contain `schema_version`, `bundle_type`, fixture, source and
prompt hashes, provenance, retrieval and packing configuration, recommended
generation settings, limitations, and `requests`. Every request row contains
`request_id`, `question_id`, `split`, `mode`, `byte_ceiling`, `packed_bytes`,
`selected_citation_ids`, and two role/content messages. The fixed system message
asks for a short answer or exactly `UNKNOWN` when evidence is insufficient and
warns that source passages are untrusted. The fixed user template contains only
`QUESTION_DATE`, `QUESTION`, and `CONTEXT`. Split, retrieval mode, budget, and
reference fields stay outside the messages.

Reference bundles contain `schema_version`, `bundle_type`, the canonical request
bundle SHA-256, fixture hash, provenance, and question references grouped under
`questions.dev` and `questions.test`. Each reference has the question ID, answer,
question type, abstention flag, and authored answer-record IDs. The canonical
hash uses sorted keys, compact separators, UTF-8, `ensure_ascii=False`, and
`allow_nan=False`.

The retriever uses the existing default `SignedHashEmbedder` and RRF constant 60.
Packing limits count UTF-8 bytes, not model tokens. The recommended generation
configuration is temperature 0 and `max_new_tokens` 128; model, checkpoint,
template, and token counts are unset. Running a model requires a separately
reviewed remote runner. This preparation performs no inference, network access,
scoring, or QA quality, human-agreement, cost, or calibrated-abstention claim.

Create two new output files with:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.prepare_synthetic_qa \
  --requests /tmp/qa-requests.json \
  --references /tmp/qa-references.json
```

The CLI refuses identical destinations and any pre-existing output; new files
are created with mode `0600`. If writing the second file fails after the first
was created, the pair can be incomplete. The reference bundle's request hash
detects mismatched pairs. The CLI does not promise a transactional two-file
write or remove unrelated files.
