# Quote-only prompt baseline results

Updated 2026-10-16 UTC. This is a development-only comparison of `quote-v2`
and `quote-coverage-v1` on 12 fictional quote-v4 development conversations.
Each pinned model saw the same two prompts for every conversation. The input
token arrays matched for all 24 paired items. The cases are adaptive; this
comparison does not establish held-out quality, exhaustive recall, QA benefit,
competitive performance, or an SFT improvement.

## Selected-fact and structural results

| Model | Prompt | Strict JSON failures | Valid citations / checked | Selected facts retained | Complete selected-fact cases | Correct empty abstentions |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-4B | `quote-v2` | 1/12 | 39/40 | 17/20 | 6/9 | 0/3 |
| Qwen3-4B | `quote-coverage-v1` | 1/12 | 56/63 | 17/20 | 7/9 | 0/3 |
| Qwen3-8B | `quote-v2` | 0/12 | 37/39 | 16/20 | 6/9 | 0/3 |
| Qwen3-8B | `quote-coverage-v1` | 0/12 | 52/58 | 16/20 | 6/9 | 0/3 |

Citation checks require exact source, quote, and explicit-date integrity. They
do not establish semantic support. A separate outer-fence-only diagnostic for
4B `quote-v2` changes its citation denominator from 40 to 43 and removes its
single strict JSON failure; it leaves 39 valid citations and finds four
invalid. The other arms are unchanged by that diagnostic. The 4B coverage
response that hit the output limit was truncated and malformed, so it remains
unreviewable rather than being repaired or counted as complete. Across the
four arms there were 48 responses, 47 reviewable responses, and 203 reviewed
claims.

The larger model and the broader coverage prompt did not improve selected-fact
retention on these cases. Both 4B prompts retained 17/20 selected facts, while
both 8B prompts retained 16/20. The coverage prompt produced one more complete
4B case, but did not improve 8B completeness. All four arms failed to abstain
correctly on the three empty-target cases. These selected facts are a small
curated target, not a measure of exhaustive recall.

The following replay uses existing structural-scoring helpers on the copied
fictional receipts. It prints parse, citation, abstention, and truncation
counts only; it does not replay semantic judgments.

```python
import json
from pathlib import Path

from experiments.quote_baseline_payload import _load_dev_cases
from training.evaluate_quote_sft import _summarize_arm
from training.evaluate_quote_v3 import _fence_diagnostic

cases, _ = _load_dev_cases()

def counts(summary):
    return {
        "parse_failures": summary["parse_failures"],
        "citation_integrity": {
            key: summary["citation_integrity"][key]
            for key in ("claims_checked", "valid_claims", "invalid_claims")
        },
        "abstention": {
            key: summary["abstention"][key]
            for key in (
                "empty_target_cases",
                "valid_empty_target_abstentions",
                "missed_nonempty_target_cases",
            )
        },
        "truncated_cases": len(summary["truncated_case_ids"]),
    }

for model in ("qwen3-4b", "qwen3-8b"):
    receipt = json.loads(
        Path(f"examples/synthetic/quote-baseline-{model}-remote-v1.json").read_text()
    )
    for prompt in ("quote-v2", "quote-coverage-v1"):
        outputs = {
            row["case_id"]: {
                "raw_text": row["raw_output"],
                "response_text": row["raw_output"],
                "generated_tokens": row["output_tokens"],
                "reached_generation_limit": row["hit_max_new_tokens"]
                and not row["ended_with_eos"],
            }
            for row in receipt["generated_outputs"]
            if row["prompt_version"] == prompt
        }
        strict = _summarize_arm(cases, outputs)
        fence = _fence_diagnostic(cases, outputs)["summary"]
        print(json.dumps({
            "model": model,
            "prompt": prompt,
            "strict": counts(strict),
            "outer_fence_diagnostic": counts(fence),
        }, sort_keys=True))
```

Semantic support, uncertainty, eligibility, and adverse-response annotations
remain review judgments, not automatic ground truth.

## Provisional semantic review

| Model | Prompt | Unsupported / uncertain claims | Ineligible claims | Responses with unsupported / uncertain claims | Responses with ineligible claims |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3-4B | `quote-v2` | 4 / 0 | 6 | 2/12 | 3/12 |
| Qwen3-4B | `quote-coverage-v1` | 5 / 1 | 10 | 4/11 | 3/11 |
| Qwen3-8B | `quote-v2` | 3 / 0 | 4 | 3/12 | 3/12 |
| Qwen3-8B | `quote-coverage-v1` | 2 / 1 | 6 | 3/12 | 3/12 |

The response counts use reviewable-response denominators (12, 11, 12, and 12).
Unsupported or uncertain claims and claims judged ineligible are separate
signals. Eligibility does not mean a statement is false: the review corrected
two blind 8B labels that had treated real supper and napkin episodes as
ineligible; salience is a separate question. Semantic labels are provisional,
not human-validated judgments.

Two fresh-context Luna reviewers performed blind first passes. Root recorded
an independent first pass before seeing their labels, then adjudicated after
reviewing them. All reviewers were from the same model family; there were zero
human reviewers. The review is useful for diagnosing these cases, not for
establishing independent truth. For example,
both coverage arms left the orchard-layout report scope uncertain. The 4B
`quote-v2` conditional phrasing was accepted, while applying a stricter rule
would add one adverse response. The 8B `quote-v2` phrase “It says” was accepted
because the booking page supplied its antecedent and the output retained the
reporting operator.

The 8B `quote-v2` output also omitted the volunteer source from an archive's
current schedule, dropped a grant-approval branch, and turned Marta's reported
move into an unqualified assertion. Both models retained fictional exercise
metadata. The expanded prompt still omitted selected facts. These examples
show why semantic support, eligibility, citation validity, parsing, and
selected-fact coverage remain separate checks.

## Interpretation and next step

Use 4B `quote-v2` only as the research generator reference for controlled,
development-only verification of candidates. It is not a production model
selection. The verifier contract is implemented and locally tested; it has
not been integrated into runtime or used for a verifier-gated candidate
inference, and it provides no evidence of model improvement. Keep the longer candidate
unread and quarantined until the full development decision is frozen and a
versioned 12-message contract is reviewed. Do not run a fifth target-tweak
SFT. Gate A remains unmet.

The historical SFT used 4-bit weights; this prompt comparison used BF16, so it
does not provide a controlled cross-run SFT comparison. The 4B and 8B worker
times were 410.072s and 454.199s on NVIDIA A10 hardware (A10G requested).
Billing was not measured. The post-8B inventory showed all apps stopped with
zero tasks, and no further GPU run was launched.

## Artifacts and validation

The protocol is in [quote-baseline-protocol.md](quote-baseline-protocol.md).
The copied fictional receipts, structural summaries, semantic summary, blind
reviews, adjudicated reviews, adjudications, packets, and curator maps are
linked here:

- [Qwen3-4B prepared payload](../examples/synthetic/quote-baseline-qwen3-4b-prepared-v1.json)
  and [remote receipt](../examples/synthetic/quote-baseline-qwen3-4b-remote-v1.json);
  [structural summary](quote-baseline-qwen3-4b-structural.json).
- [Qwen3-8B prepared payload](../examples/synthetic/quote-baseline-qwen3-8b-prepared-v1.json)
  and [remote receipt](../examples/synthetic/quote-baseline-qwen3-8b-remote-v1.json);
  [structural summary](quote-baseline-qwen3-8b-structural.json).
- [Semantic summary](quote-baseline-semantic-summary.json), 4B [blind review](quote-baseline-4b-blind-review.json),
  [adjudicated review](quote-baseline-4b-adjudicated-review.json), and
  [adjudication](quote-baseline-4b-adjudication.json); 8B [blind review](quote-baseline-8b-blind-review.json),
  [adjudicated review](quote-baseline-8b-adjudicated-review.json), and
  [adjudication](quote-baseline-8b-adjudication.json).
- Fictional [4B review packet](../examples/synthetic/quote-baseline-4b-packet-v1.json)
  and [curator map](../examples/synthetic/quote-baseline-4b-curator-map-v1.json);
  [8B review packet](../examples/synthetic/quote-baseline-8b-packet-v1.json)
  and [curator map](../examples/synthetic/quote-baseline-8b-curator-map-v1.json).

The recorded exact-head CI run [38871651124](https://github.com/1337mus/tracemem/actions/runs/38871651124)
passed on signed commit `9f068651ca503f12ede074e80a5890a4f2ab1045`: 854 of 855
source tests passed (one skipped), and 399 of 401 installed-wheel tests passed
(two skipped). That run predates these docs and verifier changes. The current
full `make check` passed: 859 tests attempted, 858 passed, one skipped, 181
files formatted, and configured mypy checked 49 source files. Strict mypy also
passed for the verifier and its test. See the
[validation record](validation-quote-baseline.json).
The new docs and code delivery remain pending; no new delivery is claimed here.
Verify current state by 2026-10-23.
