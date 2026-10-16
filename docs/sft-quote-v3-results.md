# SFT quote-v3 results

Updated: 2026-10-16 UTC

The third synthetic-only QLoRA run and its frozen three-arm evaluation are complete. The final semantic decision is **mixed**: the tuned arm had fewer responses with an unsupported or uncertain claim in the targeted test group, but required-fact coverage fell from 21/36 to 16/36. That fails the predeclared promising criteria. The run does not establish a general quality improvement, support Gate A, or select a production checkpoint.

Next research should target omitted required facts and explicit actor/report
attribution, with stronger annotations and longer independent conversations.
This frozen test is spent and must not be retuned against.

## Run and data

The run fine-tuned Qwen/Qwen3-4B at revision
1cfa9a7208912126459214e8b04321603b3df60c for 192 optimizer steps over 768
fictional rows from 72 families, including 24 new families. Training took
811.068 seconds; the remote run took 2,817.255 seconds. The NVIDIA A10G
reported peak allocated/reserved memory of 5.97/8.52 GB. The final adapter
model file SHA-256 is
d7d9e25c2c60e2348b646db92188f5aed5911f126a9eaf891e1e09fad55021fb.
The adapter remains in the personal Modal volume at
/mnt/tracemem-sft-quote-v3/adapters/quote-v3-20261015-01/adapter; the
quote-v2 prior adapter was read from
/mnt/tracemem-sft-quote-v2/adapters/quote-v2-20261015-01/adapter. The initial
adapter parameter hash 830c0b37e6faa70e2f95152cfedd55a48e25e68c15d13dcf7b41005e7829aa03
matched the second run's pre-training parameter hash. The client call took 2,845.301 seconds. No model
weights were downloaded to this Mac.

The frozen comparison used the same prompts and greedy 512-token generation
for base, quote-v2 prior, and quote-v3 tuned arms. It scored a 24-case adaptive
v2 regression cohort, 24 new development cases, and a separate 48-case test.
Generating each 96-case arm took 690.133 seconds for base, 627.905 for prior,
and 598.617 for tuned.
The job contained no private conversations or evaluation labels. The exact job
and remote receipt are preserved as [synthetic artifacts](../examples/synthetic/sft-quote-v3-job-v1.json)
and [receipt](../examples/synthetic/sft-quote-v3-remote-v1.json).

The prior adapter was verified against its recorded v2 hashes. The run audit
records that it was unloaded without merge, RNG state was restored, and no
prior LoRA parameters remained. The prior arm reproduced the historical v2
outputs on all 24 regression cases, including exact response text and
completion token IDs. The recorded post-run inventory showed the experiment
app stopped with zero tasks; verify current live state before any future
remote call.

## Strict raw scoring

The evaluator scores the original response text without fence removal or
repair. Schema failures were zero among parseable responses; strict parse
failures have no schema result. Citation checks validate exact
source, quote, and explicit-date integrity only; they do not establish semantic
support. Reference-string matches are literal wording matches, not accuracy or
recall.

| Cohort | Arm | Parse failures | Citation-valid claims / checked | Wording matches / reference strings | Correct empty abstentions |
| --- | --- | ---: | ---: | ---: | ---: |
| Regression (24) | Base | 23/24 | 1/1 | 0/19 | 0/6 |
| Regression (24) | Prior | 0/24 | 19/19 | 2/19 | 6/6 |
| Regression (24) | Tuned | 0/24 | 20/20 | 2/19 | 6/6 |
| Dev (24) | Base | 24/24 | 0/0 | 0/19 | 0/6 |
| Dev (24) | Prior | 0/24 | 24/24 | 0/19 | 6/6 |
| Dev (24) | Tuned | 0/24 | 19/19 | 0/19 | 6/6 |
| Test (48) | Base | 47/48 | 2/2 | 0/37 | 0/12 |
| Test (48) | Prior | 0/48 | 39/40 | 0/37 | 12/12 |
| Test (48) | Tuned | 0/48 | 37/37 | 0/37 | 12/12 |

The base arm's strict parse failures came from outer JSON fences. A separate
diagnostic removed only an exact outer JSON fence; it did not alter response
content or replace strict results.

| Base arm, fence-only diagnostic | Parse failures after stripping | Citation-valid / checked | Correct empty abstentions |
| --- | ---: | ---: | ---: |
| Regression (24) | 0/24 | 24/26 | 1/6 |
| Dev (24) | 0/24 | 31/32 | 1/6 |
| Test (48) | 0/48 | 69/73 | 1/12 |

## Blinded test review and root adjudication

A fresh-context agent reviewed both prior and tuned responses for all 48
frozen test cases: 96 blinded tasks and 77 claims. Root then reviewed all 96
tasks, the source messages, and all 77 claims, and changed one prior-arm label
on TEST48 from unsupported to uncertain because historical versus continuing
custody was ambiguous; the adverse-response count did not change. There were
zero human reviewers. Semantic judgments are provisional agent labels, not
independent truth.

An adverse response here contains at least one unsupported or uncertain claim.

| Test group | Prior adverse responses | Tuned adverse responses |
| --- | ---: | ---: |
| Targeted chronology and subject cases (32) | 5/32 | 2/32 |
| Chronology subgroup (16) | 2/16 | 2/16 |
| Subject-attachment subgroup (16) | 3/16 | 0/16 |
| Retention/general controls (16) | 2/16 | 1/16 |
| All test cases (48) | 7/48 | 3/48 |

Required-fact coverage counts non-empty responses that retained every
predeclared outcome-critical point. It is a bounded checklist, not exhaustive
recall; a valid quote field does not compensate for omitted information in the
claim text.

| Test group | Prior fully covered | Tuned fully covered |
| --- | ---: | ---: |
| All non-empty responses (36) | 21/36 | 16/36 |
| Targeted chronology and subject cases (24) | 13/24 | 9/24 |
| Chronology subgroup (12) | 6/12 | 1/12 |
| Subject-attachment subgroup (12) | 7/12 | 8/12 |
| Retention/general controls (12) | 8/12 | 7/12 |

Paired accounting shows the targeted adverse-response result as 4 improved,
1 regressed, 1 adverse in both arms, and 26 clean in both. Across all 48
cases, those counts are 5, 1, 2, and 40. For full required-fact coverage,
the targeted cases had 2 gains, 6 losses, 7 full in both, and 9 incomplete in
both; across all non-empty cases the counts were 3, 8, 13, and 12. The
coverage losses drive the mixed decision despite lower adverse-response
counts. See the [paired accounting](sft-quote-v3-paired-case-accounting.json)
and [root adjudication](sft-quote-v3-adjudication.json).

Concrete test examples include the tuned response omitting the actual call
date in TEST11 and correcting reported ownership in TEST23. This small,
fictional sample does not justify a broad effectiveness claim.

## Limits and remaining failures

Semantic review covered prior and tuned outputs on the frozen test only. The
base arm, development cohort, and adaptive regression cohort did not receive
full semantic review. Root separately spot-checked two original v2 development
failures, not the full regression cohort: D14 still turns a correction of an
earlier statement into an invented sequence of earlier consideration and
later choice, and D21 still leaves the keeper/owner attachment ambiguous. See the
[original-failure spot-check](sft-quote-v3-original-failures-spotcheck.json).

The review also records limits in the frozen cases: short and leading contexts,
structural similarities between V3DEV19/V3TEST36 and V3DEV21/V3TEST42, a
prospective TEST40 collection date that precedes its 2026 recorded-at
timestamp, unresolved valid-to scope in TEST37, and date versus earlier-booking
ambiguity in TEST07.
The TEST40 inconsistency remains unresolved. These fixtures were preserved;
the review does not turn them into evidence of human truth or generalization.

The [evaluator report JSON](sft-quote-v3-results.json) retains raw structural
scoring and its semantic-support field remains “not yet evaluated”; semantic
labels live separately in the [blind review](sft-quote-v3-blind-review.json),
[required-fact coverage](sft-quote-v3-coverage.json), and
[adjudication](sft-quote-v3-adjudication.json). The
[validation record](validation-sft-quote-v3.json) binds these artifacts,
copy hashes, independent replay, control audit, and source CI. Its successful
hosted CI run tested source commit a396d48da65b059bde55d0088e5996ec689cf300;
it did not test this later documentation working-tree tip.

## Offline replay

The evaluator replays the saved outputs and makes no model or GPU call. It
creates the report exclusively, so choose a new output path that does not
already exist:

    PYTHONPATH=src:. .venv/bin/python -m training.evaluate_quote_v3 \
      --job examples/synthetic/sft-quote-v3-job-v1.json \
      --receipt examples/synthetic/sft-quote-v3-remote-v1.json \
      --regression-fixture datasets/sft-quote-v2/dev.json \
      --dev-fixture datasets/sft-quote-v3/dev.json \
      --test-fixture datasets/sft-quote-v3/test.json \
      --output .local/replay-v3/quote-v3-report.json
