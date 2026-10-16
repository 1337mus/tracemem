# Quote verifier development results

Updated 2026-10-16 UTC. A single Qwen3-8B call evaluated 39 candidates from
the existing Qwen3-4B `quote-v2` outputs across 12 fictional development
conversations. This experiment did not train a model. The predeclared criterion
failed.

## Gate and verdicts

The strict generator gate found one JSON parse failure among 12 responses and
one citation failure among 40 checked claims, leaving 39 candidates. The two
cases without candidates failed this input gate: one response was fenced JSON
that strict parsing did not repair, and one claim failed citation validation.
Neither is a verifier abstention.

The frozen provisional labels classify the 39 candidates as 36 supported and
eligible, and three bad: two ineligible exercise claims and one claim that
drops its reported source. The verifier returned valid, complete verdicts for
all ten nonempty batches, with no truncation.

| Outcome | Good (36) | Bad (3) |
| --- | ---: | ---: |
| Accepted | 35 | 3 |
| Rejected | 1 | 0 |
| Unscored | 0 | 0 |

| Predeclared criterion | Result |
| --- | --- |
| Reject all three bad claims | Failed: 0/3 rejected |
| Retain at least 35 of 36 good claims | Passed: 35/36 retained |
| Ten valid, complete, untruncated batches | Passed: 10/10 |
| Overall | **Failed** |

Across verdicts, the model marked 38 claims supported and eligible and one
unsupported and ineligible; it returned no uncertain verdicts. The rejected
good postcard statement was supported, but the verifier conflated support
with eligibility. Of the three bad claims it retained, one drops the reporting
words “It says” from a refund claim; the other two preserve exercise metadata
that is ineligible under the task policy. The score recorded zero valid model
abstentions; both no-candidate cases failed the input gate.

## Selected facts and limits

A separate root audit found 17/20 selected facts retained and 6/9 nonempty
cases complete, unchanged from the generator. The rejected postcard statement
was not a selected target. Candidate retention against the 36/3 labels is not
selected-fact retention or exhaustive recall.

Labels are provisional AI-agent judgments from one model family, with zero
human reviewers. This reused development set does not establish independent
truth, generalization, QA benefit, support precision in other data, or product
suitability. No production model was selected; Gate A remains unmet.

## Cost and next step

The worker reported NVIDIA A10 hardware (A10G requested): 144.376159 seconds
function time, 85.564929 seconds of GPU time, 149.756547 seconds client time,
and 1,828 generated tokens. Billing was not measured. There was no retry; the
post-run inventory showed the app stopped with zero tasks.

The failed decision is frozen. Two separate follow-ups remain. The opt-in
[quote-data v3 contract](quote-data-v3-contract.md) changes the version and
12-message bound, preserves literal citation checks, and adds length validation;
it does not define semantic-support or eligibility policy. The prospective
v0.1 [verifier annotation contract](verifier-annotation-contract.md) proposes
separate semantic-support and eligibility labels. It is not human-approved;
unresolved eligibility stays uncertain in future diagnostics. Independent human
review is needed for human-validated or product-quality claims; the policy
ambiguity is not resolved. The sealed longer-conversation candidate remains unopened. No further GPU
call was launched. Do not run a fifth extraction target-tweak SFT.

## Artifacts and replay

See the [protocol](quote-verifier-dev-protocol.md), [CPU score](quote-verifier-dev-results.json),
[selected-fact audit](quote-verifier-selected-fact-audit.json), and
[validation record](validation-quote-verifier.json). The copied
[prepared payload](../examples/synthetic/quote-verifier-prepared-v1.json),
[remote receipt](../examples/synthetic/quote-verifier-remote-v1.json), and
[selection audit](../examples/synthetic/quote-verifier-selection-audit-v1.json)
preserve the fictional run inputs and accounting.

Replay the copied receipt locally on CPU to a new exclusive output path:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.evaluate_quote_verifier \
  --receipt examples/synthetic/quote-verifier-remote-v1.json \
  --output /tmp/quote-verifier-replay-UNIQUE.json
```

The source code is signed and pushed at `69c18e06d7877029cc9a8689169eccd62d8ddf82`.
Exact-head CI run [38877899360](https://github.com/1337mus/tracemem/actions/runs/38877899360)
passed: 887/888 source tests (one skipped) and 399/401 installed tests (two
skipped). Ruff formatted 187 files, configured mypy checked 53 files, and
GitHub verified all three preparation commit signatures. The validation record
is [here](validation-quote-verifier.json). That CI run predates this report and
the later v3 validator; consult current GitHub checks for their delivery state.
