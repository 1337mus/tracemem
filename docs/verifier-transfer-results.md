# Verifier transfer results

Updated 2026-10-16. The [frozen protocol](verifier-transfer-protocol.md) and
[data card](verifier-transfer-data-card.md) define this inference-only
comparison of the unchanged Qwen3-4B adapter against its matched untuned base
model. A separate keep-everything control is also reported. No training or
policy changes occurred.

**Decision: the primary criterion failed by one rejection; four of its five
checks passed.** The adapter retained all supported claims and improved
unsupported-claim rejection, but it missed the required 23/24 rejections and
lost decisive scope accuracy. No product model was selected.

## Results

| Measure | Untuned base | Frozen adapter | Keep everything |
| --- | ---: | ---: | ---: |
| Supported claims retained | 24/24 | 24/24 | 24/24 |
| Unsupported claims rejected | 10/24 | 22/24 | 0/24 |
| Unsupported claims still accepted | 8/24 | 2/24 | 24/24 |
| Unsupported claims labeled uncertain | 6/24 | 0/24 | 0/24 |
| Valid, untruncated model outputs | 48/48 | 48/48 | N/A |

The frozen rule required at least 22 supported claims retained, at least 23
unsupported claims rejected, all 48 outputs valid and untruncated, fewer
unsupported claims accepted than baseline, and no decline in supported
retention. Only the rejection threshold failed: 22/24 fell one short. All
outputs were valid and ended without truncation.

The scope result regressed: decisive eligibility exactness fell from 27/42 to
25/42, while exactness on uncertain references rose from 4/6 to 6/6. By
phenomenon, unsupported attribution claims rejected increased from 5/12 to
12/12; unsupported condition claims rejected increased from 5/12 to 10/12.
The two remaining false accepts are `case_transfer_20::candidate_transfer_20_unsupported`
(“if” versus “only if” for a fictional door) and
`case_transfer_24::candidate_transfer_24_unsupported` (keeping versus
cancelling a booked makerspace room).

A descriptive posthoc conjunction of support and eligibility shows the tradeoff:
supported eligible claims retained fell from 16/19 to 10/19, while unsupported
eligible claims still accepted fell from 4 to 1. This was not a frozen selection
criterion and does not offset the scope regression.

## Run and limits

The comparison used the same frozen adapter, Qwen3-4B NF4 model, inputs,
settings, and criteria. It was inference-only. The successful worker used an
NVIDIA A10 and took 166.31 seconds; the client call took 180.27 seconds. The
baseline and adapter generated 888 and 561 output tokens, respectively.

The initial launch failed before ML imports because the Modal volume appeared
as a symlink. A narrow canonical-mount fix was applied, then one manually
approved run completed with the same model, settings, data, and criteria; no
automatic retry occurred. The earlier paired-verifier failure remains
unchanged, and its frozen test split remains unrun. This transfer diagnostic
cannot change that result or authorize release of that test set.

The 24 cases and 48 claims are fictional, agent-authored development data with
zero human reviewers. There is no independent test set. These results do not
establish human-validated labels, generalization, QA benefit, extraction
recall, or product quality. Separate local private-review preparation contains
36 windows from 22 sources, all in dev with no annotations; it was not supplied
to the model.

## Replay artifacts

The [machine-scored report](verifier-transfer-results.json) preserves the
aggregate scores; the [validation record](validation-verifier-transfer.json)
preserves local verification. The recorded inputs and remote receipt are
[payload](../examples/synthetic/verifier-transfer-payload-v1.json) and
[receipt](../examples/synthetic/verifier-transfer-remote-v1.json).
Recreate the aggregate with:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.evaluate_verifier_transfer \
  --payload examples/synthetic/verifier-transfer-payload-v1.json \
  --receipt examples/synthetic/verifier-transfer-remote-v1.json \
  --output /tmp/verifier-transfer-replay.json
```
