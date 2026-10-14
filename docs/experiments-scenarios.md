# Frozen multi-session extraction comparison

2026-10-14. Both frozen scenario inference calls are complete. The Qwen app
stopped with zero tasks at 10:53:34 Pacific; Phi stopped with zero tasks at
11:11:16 Pacific. See the [compute record](modal-compute.md).

The fixture contains 48 newly authored fictional cases with 3–5 messages across
at least two sessions each, 24 dev and 24 test cases, eight no-memory cases and
43 provisional target claims. Root audited every message and target before launch.
The [data card](scenarios-data-card.md), [curation notes](scenario-curation-notes.md)
and [readiness audit](scenarios-readiness.json) describe its limits. Targets are
agent authored and may omit other supported memories. Dates correlate with split
order; declared entity slugs do not prove template or entity independence.

The dataset SHA-256 is
`5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214`.
The [comparison plan](../configs/scenario-comparison-plan.json) freezes the
explicit-v2 prompt, greedy FP16 inference, 4096 input/1024 generated tokens,
separate dev/test reporting and no post-test prompt tuning. Neither a training
split nor independent human annotations exist. Gate A is not met.

Qwen3-4B and Phi-3.5-mini use pinned revisions from different model families,
with the same rendered source messages and schema. Token counts differ by
tokenizer; the common limits are ceilings, not equal realized token costs.
Each call uses a single source-free T4 container with no retries, local source
mount or detached execution. The $2 operational target is not a billing cap.
Model weights and inference stay remote; payloads contain fictional text only.

Prepared payload SHA-256 values:

- Qwen: `c8470251500db75c83d680ab9c203fde975863a2eabf35853e34d839c69eb100`.
- Phi: `0165755adad0cf55eef58eab6899b16b62d0b810a64dea3300a568ad91a96f4c`.

`experiments.evaluate_scenarios` validates complete receipts and reports raw
and unique-exact-quote-anchored results separately. Parse/schema failures remain
visible. Anchoring repairs only offsets; it cannot prove support, correct
attribution, current truth, modality or semantic equivalence. Exact structured
F1 measures agreement with provisional authored targets and is not semantic
precision. Keep raw receipts immutable and write separate evaluation outputs.

`experiments.prepare_scenario_reviews` can build a shuffled generated-candidate
bundle with model/case keys in separate provenance. Generated candidates are
unlabeled. It excludes parse/schema failures explicitly, retains duplicate model
origins and never treats unmatched predictions as unsupported. See the
[annotation workflow](annotation-workflow.md). Human and agent judgments must
remain separate. One blind agent review supported all 43 authored targets in
the reference set; that review had no negative controls or human reviewers and
does not establish reviewer discrimination or exhaustive labels. A separate
16-task dev-derived control check had 8 positive and 8 negative examples, with
16/16 agreement from one agent reviewer and no human reviewers. It indicates
discrimination on those simple controls only; it does not establish semantic
calibration, independent truth or held-out performance. See the
[reference-review receipt](scenarios-reference-review.json) and
[reviewer-control receipt](reviewer-controls.json).

## Qwen result

The [raw receipt](../examples/synthetic/qwen3-4b-scenarios-remote.json) has SHA-256
`2035de5bfeceaf3d4be4117dd47a515ef67df1cb7899a56d5a2eacefa349410f`.
The [computed report](scenarios-qwen3-4b.json) preserves all outputs and bootstrap
intervals. All 48 outputs parsed and none hit the generation limit. Three outputs
failed strict schema validation, leaving 83 schema-valid candidate claims.

| Qwen diagnostic | Dev (24 cases) | Test (24 cases) |
| --- | ---: | ---: |
| Parse/schema failures | 0 / 1 | 0 / 2 |
| Schema-valid claims | 41 | 42 |
| Raw citation/role integrity | 1/41 | 1/42 |
| Exact-quote-anchored citation/role integrity | 41/41 | 40/42 |
| No-memory abstentions | 3/4 | 1/4 |
| Exact structured reference F1, raw and anchored | 0 | 0 |

Root inspected all three schema-failing outputs and both residual role/integrity
failures. The schema failures emit message timestamps into date-only validity
fields; the route-timing case also confuses an explicitly dated past routine with
a plan and message time with fact validity. In reading-corrections, a proposed
assistant behavior becomes a direct user plan, and another assistant statement
uses direct-user attribution. Unique quote matching cannot repair these errors.
The failed outputs remain excluded from candidate-level semantic review, not
silently reinterpreted as successful empty outputs. These exclusions also mean
any reviewed-candidate support fraction does not describe all generated text.

Exact F1 zero includes alternative wording, extra supported statements, schema,
span and label differences. It is not evidence that all 83 claims are false.
No prompt changes follow these test results. Phi uses the same frozen prompt.

Qwen used 57,947 input and 14,865 generated tokens, 919.587205 GPU seconds,
75.284621 model-load seconds and 1,069.663869 client-wall seconds. Peak allocated
GPU memory was 8,822,848,512 bytes. App `ap-2Hi9ACD33k9ynLHRyl7Coe` stopped at
2026-10-14 10:53:34 Pacific with zero tasks.

## Phi result

The [raw receipt](../examples/synthetic/phi-3.5-mini-scenarios-remote.json) has
SHA-256 `d5eff2e56a827a76a7426b47ccd26e88601bd85f2f57f178c4e0f62725a2648d`.
The [computed report](scenarios-phi-3.5-mini.json) records all 48 outputs; none
reached the 1,024-token generation limit. The pinned revision is
`2fe192450127e6a83f7441aef6e3ca586c338b77`. Phi used the same frozen explicit-v2
prompt and runner as Qwen; its prompt, runner and payload hashes are recorded in
the report and raw receipt.

| Phi diagnostic | Dev (24 cases) | Test (24 cases) |
| --- | ---: | ---: |
| Parse/schema failures | 1 / 23 | 0 / 23 |
| Schema-valid candidate claims | 0 | 1 |
| Raw citation integrity | — | 0/1 |
| Unique-quote-anchored citation integrity | — | 1/1 |
| No-memory abstentions | 0/4 | 0/4 |
| Exact structured reference F1, raw and anchored | 0 | 0 |

The [failure audit](scenarios-phi-audit.json) records first schema exceptions:
44 outputs failed because `valid_from` was too long and two used invalid
predicate labels. Those counts do not enumerate every defect in an output. The
sole schema-valid test claim had a raw quote mismatch; unique exact-quote
anchoring repaired its span, while the claim still invented an end date as
`valid_to`. Its citation repair is not semantic support. The one parse failure
and all schema failures remain visible; no parse/schema failure was silently repaired. No-
memory abstention was zero in both splits. Exact structured F1 zero is agreement
with provisional targets, not semantic precision.

Phi used 70,158 input and 18,698 generated tokens, 899.763988 GPU seconds,
56.772880 model-load seconds, 956.932029 remote-function seconds and
968.419187 client-wall seconds. Peak allocated GPU memory was 9,715,433,984
bytes. Its app `ap-zZWCi66Vv372G5Q7x93Iui` stopped at 2026-10-14 11:11:16
Pacific with zero tasks. EOS token IDs were `[32007, 32001, 32000]`.

## Generated-candidate review

The agent review covered all 84 schema-valid generated candidates. Its original
blind responses remain unchanged: 50 supported, 31 unsupported and 3 uncertain.
Root then read the full conversations, candidates and rationales in a separate
nonblind adjudication and changed nine labels: two to unsupported and seven to
uncertain. Root's resulting counts are 41 supported, 33 unsupported and 10
uncertain. This is not an independent reviewer or consensus result; there were
no human reviewers and no automatic acceptance.

The table gives the exact model/split accounting from the
[two-reviewer report](scenarios-reviewed-support.json). `Excluded parse/schema`
counts preserve outputs omitted from candidate review; schema-valid empty
outputs are also shown separately. Anchoring exclusions were zero in every row.

| Model / split | Raw outputs | Excluded parse / schema | Valid empty outputs | Reviewed candidates | Blind agent supported / unsupported / uncertain | Root nonblind supported / unsupported / uncertain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B / dev | 24 | 0 / 1 | 3 | 41 | 26 / 14 / 1 | 20 / 16 / 5 |
| Qwen3-4B / test | 24 | 0 / 2 | 1 | 42 | 24 / 16 / 2 | 21 / 16 / 5 |
| Phi-3.5-mini / dev | 24 | 1 / 23 | 0 | 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Phi-3.5-mini / test | 24 | 0 / 23 | 0 | 1 | 0 / 1 / 0 | 0 / 1 / 0 |

The single Phi test candidate has invalid raw citation integrity and becomes
quote-valid only after unique exact-quote anchoring; its support judgment remains
unsupported. The review covers valid candidates only, not all generated text.
These descriptive judgments and their fractions are not recall, independent
human truth, precision calibration, semantic acceptance, or Gate A. See the
[root adjudication receipt](scenarios-root-adjudication.json) for its scope and
limits. The fictional-only source bundle, key, provenance and separate responses
are preserved under [synthetic review artifacts](../examples/synthetic/reviews/README.md).
No model is selected from these results.

The [retrospective integrity-versus-support replay](integrity-selectivity.md)
now records raw and uniquely anchored integrity filters against both existing
reviewers, with all model/split denominators and exclusions preserved. Quote
repair retains unsupported-labeled claims in both splits; it is not a semantic
acceptance rule. This replay makes no inference calls and changes no labels,
prompts or runtime behavior.
