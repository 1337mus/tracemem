# Verifier v2: recover useful memories without losing support checks

The previous verifier caught 22/24 unsupported claims, up from 10/24 for the
base model, but a filter requiring both support and eligibility retained only
10/19 valid personal memories. This experiment tests whether a broader
fictional curriculum improves that tradeoff. The prompt and scope policy stay
fixed. All labels are provisional agent judgments; no human has reviewed them.

Compare the same Qwen3-4B base, the frozen prior adapter, and that adapter after
continuation training on 192 rows from 96 new scenarios. Use two epochs,
96 optimizer steps, learning rate 0.00005, and one A10G with an 1,800-second
limit. Keep the prior adapter read-only and save the new adapter separately.
No private conversations or local model weights are involved.

Freeze data, source hashes, payload and criteria before producing outputs.
Evaluate identical prompts and token IDs across all three arms, with greedy
non-thinking decoding. Retain raw outputs, tokens, finish flags, training
counts/losses, runtime versions, timings and adapter hashes. No automatic retry.

| Development cohort | Supported retained | Unsupported rejected | Supported eligible retained | Decisive eligibility exact | Uncertain eligibility exact |
| --- | --- | --- | --- | --- | --- |
| New 24 scenarios | ≥23/24 | ≥23/24 | ≥11/12 | ≥33/36 | ≥10/12 |
| Prior transfer regression | 24/24 | ≥23/24 | ≥17/19 | ≥38/42 | 6/6 |
| Original paired dev | 6/6 | ≥4/6 | Report separately | Report separately | Report separately |

The continued arm must meet every threshold. On new dev, accept at most one
unsupported eligible claim. Require every output in every arm to be valid and
untruncated, and no increase in unsupported false accepts relative to the prior
adapter on any cohort. Matched base and prior transfer outputs must reproduce
the historical receipt item by item; a mismatch fails the protocol and is
reported. Adaptive regression cohorts are not held out.

A fresh-context author will seal 24 additional fictional test scenarios before
training, with 12 eligible, six ineligible and six uncertain pairs. Evaluate the
single final candidate once after development scoring, even if development
fails, to audit that fixed candidate. Use the new-dev thresholds and report the
test separately. A test pass cannot rescue failed development criteria. Never
choose another checkpoint or tune on these test outputs. Distinct identifiers
alone do not prove semantic independence; disclose template and scenario
similarities. Keep all historical sealed tests unopened.

This is a claim-verifier diagnostic. It does not establish extraction recall,
private-data performance, downstream memory QA, calibration or product quality.

A pre-output policy audit found a historical label tension: transfer case 04
uses fleetingness to mark a real-user fact uncertain, although the fixed prompt
does not require durability. Preserve its original labels and the frozen gate;
show a separate eligibility sensitivity analysis excluding that case. Transfer
case 11 also omits an explicit reporting verb in its accepted positive claim.
These caveats limit the older regression set; they do not justify relabeling it
after training or rescuing a failed criterion.

## Runtime correction before any optimizer step

The first remote attempt failed Modal serialization before starting the worker.
The second reached inference but failed historical replay before training. That
attempt remains failed; its exception did not preserve the full output arrays,
so its complete mismatch count is unknown.

Code comparison found that historical transfer inference called
`prepare_model_for_kbit_training` before evaluation, which casts nonquantized
layers to float32. The v2 runner omitted that preparation. The corrected attempt
will match the historical preparation before base, prior and reloaded continued
inference, and use the same preparation for the later blind test. It must still
pass exact historical replay before training. Matching replay would verify
reproducibility after the correction; the lost outputs prevent reconstructing
every difference in the failed attempt.

On replay failure, save all base/prior outputs and every field difference in a
`protocol_failed` receipt and exit unsuccessfully with zero training steps.
Data, prompts, labels, decoding settings, training schedule and thresholds stay
frozen. No automatic retry or replacement checkpoint is permitted.
