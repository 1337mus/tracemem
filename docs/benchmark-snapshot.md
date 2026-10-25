# Benchmark snapshot

Recorded October 23, 2026. This comparison checks one step: whether a proposed
memory claim agrees with the original messages.

## Highlights

| Measure | Base model | Trained candidate |
| --- | ---: | ---: |
| Source support among accepted claims | 66/78 · 84.6% | **58/61 · 95.1%** |
| Useful reference claims retained | 28/28 | **28/28** |

Of the 61 claims the candidate accepted, 58 were supported by the source under
the reviewed labels. It also kept all 28 useful reference claims: facts that
were both supported and worth remembering under the chosen policy.

## What was measured

The comparison covered **120 claims across 30 conversation families and 40
sessions**, using selected LongMemEval oracle evidence. Each claim was checked
against the same source messages across models. References were reviewed by
agents and settled before the model outputs were scored.

The owner chose a simple memory policy: **remember durable facts and
preferences**. Source support and memory scope were judged separately.
The table reports source support among accepted claims and useful-claim
retention, rather than a combined system score.

## How it was trained

The candidate used Qwen3-4B with **256 newly authored fictional training
claims** and 64 training steps. Benchmark conversations were used for
evaluation only. Model runs used a fixed comparison plan and saved results.

The runnable local demo is a separate path. It searches the repository's
fictional messages and returns their source citations.

[Machine-readable counts](benchmark-snapshot.json) ·
[Run the demo](../README.md#quick-start) ·
[Experiment archive](README.md#experiments-and-model-research)
