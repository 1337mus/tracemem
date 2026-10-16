# Model candidates and diagnostics

This page records model metadata and historical inference diagnostics. Metadata
was checked through the public Hugging Face model API on 2026-10-13; the pinned
NLI model card was reviewed on 2026-10-14. The bounded candidate comparisons
used remote containers and authored synthetic inputs only; those runs did not
train models, and no weights were downloaded to this Mac. Separately, four
synthetic SFT runs have completed. No model is selected for product use. See
[current project status](project-status.md) and the [quote-v4 results](sft-quote-v4-results.md).

| Candidate | Parameters | Model-card license | Role | Status |
| --- | ---: | --- | --- | --- |
| [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4.02B | Apache-2.0 | Extraction diagnostic | Historical V1 and explicit-v2 prompt runs used the same 24 diagnostic cases. V1 had 4 schema failures, 0/24 integrity-valid claims, and exact-match F1 0; anchoring raised span integrity to 21/24 with three role conflicts. Explicit-v2 had one parse and one schema failure, 0/21 raw claims with valid evidence integrity, and exact-match F1 0; anchoring raised span integrity to 21/21, with F1 unchanged. The separate frozen 48-case comparison is complete; see the [scenario report](experiments-scenarios.md). |
| [cross-encoder/nli-deberta-v3-base](https://huggingface.co/cross-encoder/nli-deberta-v3-base) | Not recorded | Apache-2.0 card | Support-verifier diagnostic | Measured on 48 agent-reviewed synthetic pairs across 24 scenarios. At entailment thresholds 0.50/0.90/0.99, supported precision was 0.8276/0.8214/0.8750 and coverage was 29/48, 28/48, and 24/48. Labels are not independent human ground truth; no verifier policy or model selection follows. See the [support audit](support-audit.md). |
| [Phi-3.5-mini-instruct](https://huggingface.co/microsoft/Phi-3.5-mini-instruct) | 3.82B | MIT (metadata) | Different model family | The frozen 48-case comparison is complete. One output passed schema validation; its raw citation was invalid and only its quote span was repaired by anchoring. See the [scenario report](experiments-scenarios.md). |
| [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | 8.19B | Apache-2.0 | Larger comparator | Metadata only. Not run. |

## Frozen scenario comparison

The current extraction comparison uses a new fixture distinct from the historical
24-case diagnostic challenge: 48 fictional multi-session cases split 24 dev and
24 test, with eight no-memory cases and 43 provisional agent-authored target
claims. Its SHA-256 is
`5a12c98d8cc404d8bf229dab9a5d709b7ef106f4d5ffe7ebe3daafb9ca434214`. One blind
agent review supported all 43 authored targets. There were zero human reviewers
and zero negative controls in that reference-only review; this positive-only
agreement does not establish reviewer discrimination, exhaustive labels,
semantic precision, or Gate A. A separate 16-task dev-derived control check
included eight positive and eight simple negative controls; one blind agent
agreed with all 16. This shows discrimination on those controls only, not
semantic calibration, independent truth or held-out performance. See the
[control receipt](reviewer-controls.json).

The two inference runs in this historical comparison are complete and their
Modal apps stopped with zero tasks. These statements apply to those comparison
runs; quote-v4 training was a separate completed run. No model has been
selected, and these runs did not upload private text. Frozen settings,
payload details, results and limitations are in the
[comparison report](experiments-scenarios.md), [data card](scenarios-data-card.md),
and [reference-review receipt](scenarios-reference-review.json).

The Qwen results on the older 24-case fixture and the current 48-case run are
separate studies; the former is not a baseline row for the new split. Exact
structured F1 compares outputs with provisional authored representations and
can penalize alternative wording, schema, citation, span or label differences.
It does not measure semantic falsehood by itself. Quote anchoring can repair
offsets but does not prove support, correct attribution, current truth, modality
or semantic equivalence. An agent reviewed all 84 schema-valid generated
candidates: its original blind responses were 50 supported, 31 unsupported and
3 uncertain. A separate nonblind root adjudication changed nine judgments (two
to unsupported and seven to uncertain), resulting in 41 supported, 33
unsupported and 10 uncertain. Root's labels are not an independent reviewer,
human truth, consensus or automatic acceptance. See the
[per-model/split report](scenarios-reviewed-support.json) and
[root adjudication receipt](scenarios-root-adjudication.json). The report keeps
parse/schema exclusions explicit; invalid outputs are not silently repaired or
counted as empty responses.

[Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) is a candidate, not a selected
model or a claim that it is the best checkpoint. The 24-case prompt runs share
their cases and are not independent evaluations. The pinned Qwen LICENSE text
and MiniLM README license card were inspected; see [data governance](data-governance.md)
for source pins, hashes and remaining rights limits. This is not a complete
rights review. Dependencies and any quantization artifact terms must be checked
before downloading or releasing derived weights. The registry's `status` field
accurately says no model is selected; current execution status appears in this
report.

## Retrieval embedding diagnostic

[all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, ran as a separate remote
embedding baseline on 40 synthetic records and 17 queries. Vector mode retrieved
sufficient evidence for 14/14 answerable queries, with recall@5 1.0. It did not
abstain on any of the 3 unknown queries. Hybrid recall@5 was 0.9167, also with
0/3 unknown abstentions. See the [computed retrieval report](baseline-minilm-retrieval.json)
and [Phase 2 measurements](experiments-phase2.md). This small fixture is
diagnostic, not held-out or calibrated evidence.

The inspected machine reports Apple M5 Max, 64 GB unified memory and 18 CPU
cores. The runnable baseline uses CPU-only Python and SQLite. Modal is the
preferred remote backend. Keep weights and inference off this Mac by default.
The recorded diagnostic inventory contains four earlier synthetic-only Modal
calls and two calls for the scenario comparison (six total; both scenario apps
were stopped at that time). It predates quote-v4 training; see [compute
notes](modal-compute.md), [current project status](project-status.md), and the
[quote-v4 results](sft-quote-v4-results.md). Selection claims still need
applicable rights review and independent human evidence; private-source
uploads, public release, or materially different costly scope require separate
approval.

## Comparison boundary and next steps

The earlier broader comparison plan called for at least 100 distinct,
license-clean scenarios. The current 48-case set is an intermediate frozen
comparison and does not meet that planned scale or Gate A. The two model runs,
generated-candidate agent review, and separate root adjudication are complete.
Before model or quality claims, obtain independent human labels and audit a
sufficiently broad held-out set. Any later comparison should keep the prompt,
schema, pinned model revisions and decoding settings fixed across candidates,
report split-specific results and uncertainty, and measure resource use. Exact
quote matches alone do not prove entailment.

The present retrieval fixture cannot justify training or model-selection
conclusions.
