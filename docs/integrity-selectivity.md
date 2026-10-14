# Integrity selectivity experiment

This offline experiment compares evidence-integrity checks with reviewer labels
for model-generated claims in the fictional scenario set. It measures three
retrospective filters over schema-valid claim origins. It does not choose a
winner or tune a threshold.

## Run it

From the repository root, pass one or more immutable raw scenario receipt files
and zero or more complete reviewer response documents:

```sh
python -m experiments.evaluate_integrity_selectivity \
  --receipts .local/receipt-a.json .local/receipt-b.json \
  --responses .local/reviewer-a.json .local/reviewer-b.json \
  --seed 1729 \
  --output .local/integrity-selectivity.json
```

The command validates every receipt and response before writing. It uses the
existing exclusive-output helper, so it refuses to replace an existing output
or write over an input. The output directory must already exist. The experiment
does not call a model, use a GPU, or access a network service. The Python API is
`evaluate_integrity_selectivity(receipt_bytes_sequence, response_documents,
seed=...)` and returns the report without writing a file.

## What the filters mean

The report maps each original model, case, and claim index to its review task
through regenerated provenance. Multiple origins can share one deduplicated
task. Each origin remains a separate count, including when the raw versions of
those origins have different integrity results. Reviewer labels come from the
anchored task because anchoring is checked to change offsets only. Candidate
count and order must match, and each anchored claim must equal the result of
applying the existing unique-quote anchoring function to its raw claim.

The report compares exactly these filters:

- `all_schema_valid` passes every schema-valid origin that entered the blinded
  review bundle.
- `raw_integrity` passes an origin when the existing verifier returns `REVIEW`
  for its original evidence offsets.
- `anchored_integrity` applies unique exact-quote anchoring, then passes the
  origin when the verifier returns `REVIEW`.

`REVIEW` means the integrity checks passed. It does not mean that a claim is
supported or true. Anchoring repairs only uniquely located exact-quote offsets;
it does not establish entailment.

Each model has separate `dev` and `test` cells. Each filter reports passed and
rejected origins and coverage, calculated as passed origins divided by
schema-valid origins. Reviewer rows keep reviewer IDs and self-reported kinds
separate. They report passed and rejected counts for `supported`,
`unsupported`, and `uncertain` labels. Uncertain labels remain uncertain.

Reviewer metrics use these denominators:

- `supported_rate_all_passed_origins` is passed supported origins divided by
  all passed origins.
- `supported_rate_decisive_passed_origins` is passed supported origins divided
  by passed supported plus passed unsupported origins. Uncertain labels are
  excluded from this denominator.
- `supported_retention_rate` is passed supported origins divided by all
  reviewed supported origins.

Any metric with a zero denominator is `null`. A cell with no reviewers still
reports integrity counts and has an empty reviewer list. Parse failures,
schema exclusions, and valid empty outputs remain distinct in the existing
summary under `accounting`.

## Scope and limits

The report includes input and regenerated provenance hashes, the existing full
review accounting, per-model filter counts, verifier error-code counts, and an
origin audit containing model, case, claim index, task ID, pass flags, and error
codes. It does not include transcript or claim text, reviewer rationales, or
raw response bodies. Hashes and identifiers can still be sensitive; keep the
report local and access-controlled.

The filters apply retrospectively only to schema-valid candidate origins
selected for review. Reviewer labels are self-attested. Reviewer identity and
independence are not verified. Candidate-denominator coverage is not whole
output success or recall. These measurements provide no calibration or
precision guarantee, Gate A result, prospective test, runtime acceptance
policy, or model-superiority conclusion. No result from this experiment should
be interpreted as semantic truth.

## Recorded fictional replay

Root executed the report on 2026-10-14 using the frozen Qwen and Phi receipts
and the existing blind-agent and separate nonblind-root response documents.
The [machine-readable report](scenarios-integrity-selectivity.json) records the
exact raw receipt hashes, canonical response hashes and regenerated task mapping.
No prompt, label, threshold or runtime acceptance policy was changed.

| Model / split | Schema-valid origins | Raw integrity passes | Anchored integrity passes | Excluded parse / schema outputs |
| --- | ---: | ---: | ---: | ---: |
| qwen3-4b / dev | 41 | 1 | 41 | 0 / 1 |
| qwen3-4b / test | 42 | 1 | 40 | 0 / 2 |
| phi-3.5-mini / dev | 0 | 0 | 0 | 1 / 23 |
| phi-3.5-mini / test | 1 | 0 | 1 | 0 / 23 |

Labels among anchored-integrity passes remain separated by reviewer. Reviewer
IDs in the report are `luna-blind-support-20261014` and
`root-adjudication-20261014`; both are agents, with zero independently verified
human reviewers. Root was nonblind and is not an independent consensus label.

| Model / split | Reviewer | Passed supported | Passed unsupported | Passed uncertain |
| --- | --- | ---: | ---: | ---: |
| qwen3-4b / dev | blind agent | 26 | 14 | 1 |
| qwen3-4b / dev | nonblind root | 20 | 16 | 5 |
| qwen3-4b / test | blind agent | 24 | 14 | 2 |
| qwen3-4b / test | nonblind root | 21 | 14 | 5 |
| phi-3.5-mini / dev | blind agent | 0 | 0 | 0 |
| phi-3.5-mini / dev | nonblind root | 0 | 0 | 0 |
| phi-3.5-mini / test | blind agent | 0 | 1 | 0 |
| phi-3.5-mini / test | nonblind root | 0 | 1 | 0 |

On this replay, quote-offset repair retains many unsupported-labeled claims.
This demonstrates why offset integrity cannot serve as a semantic acceptance
rule. It does not establish human-grounded precision or a trained-model benefit.
A cell with no passed candidates has undefined support fractions; zero counts
in that cell do not imply perfect rejection. Verifier error-code counts count
each returned occurrence, including repeated errors within one claim.

Independent root checks replayed all 84 origin mappings/pass flags, 24
model/split/filter/reviewer cells, 72 reviewer metrics, 168 origin error lists
and eight error aggregates. A copied experiment run outside the checkout used
the approved installed runtime and reproduced the entire report exactly.
