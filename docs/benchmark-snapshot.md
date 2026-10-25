# Claim-checking test snapshot

Recorded October 23, 2026. This test compares a base language model with a trained adapter on whether proposed facts are supported by their source and worth keeping under the owner's policy.

## Results

| Measure | Base model | Trained candidate |
| --- | ---: | ---: |
| Source support among accepted claims | 66/78 (84.6%) | 58/61 (95.1%) |
| Useful reference claims retained | 28/28 | 28/28 |

Source support is the share of accepted claims whose source supports them. Useful-claim retention is reported separately: both models retained all 28 reference claims judged worth keeping. These are two measures from this focused test, not a combined score.

## Test and model

The evaluation used selected public LongMemEval oracle evidence: 120 claims across 30 conversation families and 40 sessions. Each candidate was checked against the same source messages across models. Agents reviewed the reference labels before scoring model outputs. The owner's policy counted durable facts and preferences as useful. Source support and worth-keeping eligibility are separate labels: supported, unsupported, or uncertain; and eligible, ineligible, or uncertain.

The candidate uses Qwen3-4B, a pretrained language model with about 4 billion parameters, loaded in 4-bit NF4 with bfloat16 computation. The latest training pass continued a LoRA adapter from earlier fictional-data training, while the base model stayed fixed. It updated 33,030,144 adapter parameters. The model receives source messages and a proposed fact, then returns the two labels; the evaluation preserved each source link separately.

## Training run

Training used 256 newly authored fictional claims for one epoch and 64 optimizer steps. The single-GPU run used an NVIDIA A10G on Modal and took about 5.5 minutes. The training examples were fictional; the selected public benchmark evidence was used only for evaluation. Local search retrieves cited passages, while the model experiment checks proposed facts against their source.

[Machine-readable counts](benchmark-snapshot.json) ·
[Run the demo](../README.md#quick-start) ·
[Experiment archive](README.md#experiments-and-model-research)
