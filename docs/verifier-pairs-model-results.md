# Paired-verifier QLoRA results

Updated 2026-10-16. This report covers one frozen synthetic development
comparison. The [protocol](verifier-pairs-model-protocol.md) and
[data card](verifier-pairs-data-card.md) define the run and its labels.

**Decision: the primary criterion failed; the secondary scope criterion
passed. The test split remains unrun, and no model was selected for product use.**

## Development results

| Measure | Untuned baseline | Tuned adapter |
| --- | ---: | ---: |
| Supported claims retained | 5/6 | 6/6 |
| Unsupported claims explicitly caught | 3/6 | 4/6 |
| Unsupported claims still labeled supported | 2/6 | 2/6 |
| Decisive scope labels correct | 4/8 | 7/8 |
| Uncertain-scope references exact | 1/4 | 4/4 |
| Support labels correct for both candidates in a pair | 3/6 | 4/6 |
| Valid, untruncated outputs | 12/12 | 12/12 |
| Generated output tokens | 218 | 145 |

The [machine-scored report](verifier-pairs-model-results.json) and
[validation record](validation-verifier-pairs-model.json) preserve the
replayable metrics and local verification evidence.

The primary rule required all six unsupported claims to be explicitly caught,
at least five of six supported claims retained, valid untruncated output for
all 12 candidates, more unsupported claims caught than the matched baseline,
and no decline in supported retention. The tuned arm met the latter checks
but caught only four unsupported claims, so the primary criterion failed. Its
one-count catch increase came from changing `case-035::candidate-070` from
`uncertain` to `unsupported`; it did not reduce the two unsupported claims
still labeled `supported`.

The remaining false accepts are `case-005::candidate-010`, which drops that a
handbill is reporting a source, and `case-011::candidate-022`, which drops the
“dry by noon” condition. On `case-035::candidate-069`, the tuned output
correctly retained support for the courtyard claim but changed its scope from
decisive `eligible` to `uncertain`. The scope result is reported separately:
decisive exactness rose from four to seven of eight, and exactness on the four
uncertain references rose from one to four. This passes the frozen secondary
scope criterion; it does not override the primary failure.

Both arms produced 12 valid outputs, all ending with EOS and none truncated.
The standalone baseline output array and the matched pre-training baseline
array were identical item by item, including token IDs and decoded text.
Token counts above come from the recorded output IDs, not estimates.

## Run and limits

The run trained on 48 reviewed synthetic rows for four epochs: 192
microsteps and 48 optimizer steps, with 33,030,144 trainable LoRA parameters.
The receipt reports an NVIDIA A10 GPU. Training took 281.848014 seconds;
worker time was 350.175540 seconds and client time was 362.648002 seconds.
Optimizer-step loss moved from 1.523054 to 0.003348; that training-loss
change is not evidence of verifier quality.

The initial baseline launch failed before model loading because the injected
Modal runtime had no package metadata for `modal`. The runtime check was
repaired; the successful baseline and training calls used the same frozen
prompts, labels, model settings, and criteria. No outputs from the failed call
were scored.

The 12 dev candidates are six curator-visible, correlated fictional pairs
with provisional labels and zero human reviewers. This is a small diagnostic,
not evidence of QA benefit, extraction recall, independent truth, or
generalization. The test split was not run, the sealed longer-conversation
candidate remains unopened, and no product model was selected. The original
four extractor runs and their results are unchanged.

Only newly authored fictional data was used for this experiment. No private
conversation was included in training or evaluation, and none was uploaded.
The adapter remains in the private Modal volume; model weights were not
downloaded to the Mac. Post-run inventory showed all apps stopped and zero
active tasks; automation is paused.

## Replay artifacts

The archived payloads and successful receipts are
[baseline payload](../examples/synthetic/verifier-pairs-baseline-payload-v1.json),
[baseline receipt](../examples/synthetic/verifier-pairs-baseline-remote-v1.json),
[training payload](../examples/synthetic/verifier-pairs-train-payload-v1.json),
and [training receipt](../examples/synthetic/verifier-pairs-train-remote-v1.json).
The training receipt contains both the matched baseline and tuned outputs.
To reproduce the scored training receipt, choose a new output path because
the evaluator creates output files exclusively:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.evaluate_verifier_pairs \
  --payload examples/synthetic/verifier-pairs-train-payload-v1.json \
  --receipt examples/synthetic/verifier-pairs-train-remote-v1.json \
  --output /tmp/verifier-pairs-train-replay.json
```
