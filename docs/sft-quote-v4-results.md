# Fourth SFT run: quote-v4

Updated: 2026-10-16 UTC

The fourth synthetic-only QLoRA run is complete. Its tuned adapter failed the
predeclared test threshold: it completely retained fewer selected facts than
the prior quote-v3 adapter, and adverse responses increased. The run does not
support a model-quality or production claim. Review was agent/root only, with
zero human reviewers; Gate A remains unmet and no model was selected.

## Experiment and training

The frozen comparison used 12 development and 24 test conversations. These
contain 9 nonempty/3 empty development cases with 20 selected facts, and 18
nonempty/6 empty test cases with 53 selected facts. Three arms read the same
source: the pinned Qwen3-4B base, the unchanged quote-v3 adapter, and a new
adapter trained on 768 rows from 72 families, with target corrections in
seven families (64 rows). The old quote-v3 test was spent and excluded. The
`V4T-` training-ID prefix also changes prompt and target tokens, so this is not
a pure target-text ablation.

Training used seed 17, 768 rows from 72 correlated families, 696 claims, 140
empty rows, 192 optimizer steps, and a 2,048-token training limit. The seven
corrected families contain 64 rows. Training took 1,147.795 seconds; worker
elapsed time was 2,945.737 seconds and client elapsed time was 2,963.787
seconds. The request specified A10G; the worker reported NVIDIA A10. All 192
losses were finite and the adapter changed from its verified initial state.
The prior adapter matched its recorded hashes before and after comparison,
was unloaded without merge, and left no prior LoRA parameters loaded; RNG state
was restored. The new adapter's `adapter_model.safetensors` SHA-256 is
`7e48da34440cfa75f5ae13a95de4a9a5394c81c6dad64996fa2588d20c96f0e0`. It is
stored in the personal Modal volume `tracemem-sft-quote-v4` at
`adapters/quote-v4-20261016-01/adapter`. No weights were downloaded here.

All arms used greedy decoding with a 1,024-token output cap and a 4,096-token
evaluation context. The cap is higher than v3's historical 512-token cap, so
raw counts should be compared within v4 only. No output was truncated.

## Strict structural scoring

The evaluator scored original response text without fence stripping or repair.
Citation checks validate exact source, quote, and explicit-date integrity; they
do not establish semantic support. Reference-string matches are literal
wording matches. All parseable responses had zero schema failures, and all
three arms had zero truncations.

Development results:

| Arm | Parse failures / 12 | Citation-valid / checked claims | Literal matches / 20 | Correct empty abstentions / 3 |
| --- | ---: | ---: | ---: | ---: |
| Base | 9 | 3/7 | 0 | 0 |
| Prior quote-v3 | 0 | 28/32 | 0 | 3 |
| Tuned quote-v4 | 0 | 15/17 | 0 | 3 |

Test results:

| Arm | Parse failures / 24 | Citation-valid / checked claims | Literal matches / 53 | Correct empty abstentions / 6 |
| --- | ---: | ---: | ---: | ---: |
| Base | 21 | 8/11 | 0 | 0 |
| Prior quote-v3 | 0 | 62/68 | 0 | 6 |
| Tuned quote-v4 | 0 | 55/58 | 0 | 6 |

The base arm's strict parse failures were caused by outer JSON fences. A
separate diagnostic removed only an exact outer fence; this reduced base parse
failures to 0/12 in development and 0/24 in test. The fence-only citations
were 38/43 in development and 104/114 in test, with no correctly empty
abstentions. This diagnostic changes presentation parsing only and does not
replace the strict results.

## Semantic review and selected-fact coverage

One fresh-context Luna reviewer assessed 108 shuffled tasks and 332 claims
across every development and test case and all three arms. Root independently
completed a first pass, then adjudicated with arm identities visible. The
counts below use the final root labels; changes from the blind review are
preserved in the separate adjudication artifact. The reviewer and root used
the same model family. These provisional labels are not independent truth or
human agreement.

An adverse response contains at least one unsupported or uncertain claim.
Coverage counts only whether all selected facts were retained in a nonempty
response with the right actor, modality, and time scope. The selected facts are
not exhaustive; an incomplete response may still preserve useful facts, and an
adverse label applies at the whole compound-claim level.

Development results:

| Arm | Complete nonempty responses | Selected facts retained | Adverse responses |
| --- | ---: | ---: | ---: |
| Base | 3/9 | 12/20 | 5/12 |
| Prior quote-v3 | 6/9 | 15/20 | 5/12 |
| Tuned quote-v4 | 2/9 | 11/20 | 3/12 |

Test results:

| Arm | Complete nonempty responses | Selected facts retained | Adverse responses |
| --- | ---: | ---: | ---: |
| Base | 11/18 | 45/53 | 5/24 |
| Prior quote-v3 | 9/18 | 41/53 | 3/24 |
| Tuned quote-v4 | 8/18 | 40/53 | 6/24 |

The paired test coverage comparison had three gains (V4E-12, V4E-13, V4E-17)
and four losses (V4E-03, V4E-08, V4E-15, V4E-18); five cases were complete in
both arms and six in neither. Paired adverse-response accounting was four
tuned-only, one prior-only, two adverse in both, and 17 clean in both. The
predeclared threshold required at least three additional complete nonempty
test responses versus quote-v3, with no increase in adverse responses or
invalid citations. Completeness fell from 9/18 to 8/18 and adverse responses
rose from 3/24 to 6/24. Invalid claims did fall from six to three, but that
does not offset the failed coverage and adverse-response requirements.

## What the errors show, and what they do not

The tuned output still reassigned a pet's breakfast to the user (V4E-16),
changed “choosing” into “chose” (V4E-13), and changed a future plan into
“now” (V4E-03). The blind reviewer misread a literal base-arm Lumen quotation
as an actor swap and missed an adapter quilt actor swap; the adjudication
preserves these as limits on blind-label reliability. V4E-04 (official
reporting), V4E-08 (editorial policy), and V4E-24 (a fictional contradiction)
are sensitive boundary cases. Their labels should not be treated as exact
precision estimates.

The test is spent and was not used to tune prompts or targets. A valid quote
does not prove the claim is semantically supported, and fact coverage measures
only the selected checklist. This small synthetic comparison does not show
generalization, downstream QA benefit, token or cost savings, or Gate A
readiness. Billing was not measured. No personal conversations or private
derived data entered training or inference.

The planned next comparison is dev-only, with two prompt variants on matched
BF16 Qwen3-4B and Qwen3-8B runs. The earlier SFT runs used 4-bit weights, so
this will not be a controlled SFT ablation that isolates the prompt change.
Keep prompt selection on development data; do not tune on test outputs.

## Artifacts and replay

The [frozen protocol](sft-quote-v4-protocol.md) and [training-target audit](training-target-review.md)
define the setup. The machine-readable [training job](../examples/synthetic/sft-quote-v4-job-v1.json)
and [remote receipt](../examples/synthetic/sft-quote-v4-remote-v1.json) preserve
the submitted inputs and completion metadata. The [structural report](sft-quote-v4-results.json),
[semantic summary](sft-quote-v4-semantic-summary.json), [selected-fact
coverage](sft-quote-v4-coverage.json), [adjudicated review](sft-quote-v4-adjudicated-review.json),
and [final adjudication](sft-quote-v4-adjudication.json) retain case-level
evidence. The [validation record](validation-sft-quote-v4.json) binds copies,
hashes, independent replay, and source-delivery checks. Its source CI run
predates these result-document changes and does not verify the current docs.

The structural JSON's `semantic_support` field remains `not yet evaluated`.
Semantic support and selected-fact judgments are separate final review,
coverage, and adjudication artifacts; the structural score does not contain
those labels.

Offline structural replay uses saved outputs and makes no model or GPU call.
Choose a new output path because the evaluator creates the file exclusively:

```sh
PYTHONPATH=src:. .venv/bin/python -m training.evaluate_quote_v4 \
  --job examples/synthetic/sft-quote-v4-job-v1.json \
  --receipt examples/synthetic/sft-quote-v4-remote-v1.json \
  --dev-fixture datasets/sft-quote-v4/dev.json \
  --test-fixture datasets/sft-quote-v4/test.json \
  --output .local/replay-v4/quote-v4-report.json
```

Offline semantic replay validates the saved packet, curator map, final
adjudicated review, and coverage, then compares the computed summary with the
saved summary. It reads these files and writes nothing:

```sh
PYTHONPATH=src:. .venv/bin/python - <<'PY'
import json
from pathlib import Path
from experiments.quote_review_summary import summarize

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

result = summarize(
    read("examples/synthetic/sft-quote-v4-review-packet-v1.json"),
    read("examples/synthetic/sft-quote-v4-review-map-v1.json"),
    read("docs/sft-quote-v4-adjudicated-review.json"),
    read("docs/sft-quote-v4-coverage.json"),
)
assert result == read("docs/sft-quote-v4-semantic-summary.json")
PY
```
