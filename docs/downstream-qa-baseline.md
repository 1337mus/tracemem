# Downstream QA baseline

This inference-only baseline used base Qwen3-4B with raw retrieval history and oracle history, plus the default weak signed-hash embedding. It scored 24 adaptive public oracle questions across six task types; it is not a trained-memory efficacy result, a held-out or official benchmark, or a full-history retrieval evaluation.

| Condition | Source-answer correct | Grounded and correct | Abstentions | Unsupported additions |
| --- | ---: | ---: | ---: | ---: |
| Hybrid retrieval | 5/24 | 5/24 | 18/24 | 1 |
| Empty context | 0/24 | 0/24 | 24/24 | 0 |
| Oracle evidence | 7/24 | 6/24 | 17/24 | 1 |

On the 22 unqualified references, the joint grounded-correct rate was 5/22 for hybrid retrieval and 6/22 for oracle evidence. On the two qualified references, it was 0/2 in every condition. Scoring kept answer correctness separate from prompt grounding: a stale earlier value can be supported by the visible history while still answering the current question incorrectly; dated updates should be resolved by their chronology. A stated intention or plan is not evidence that the planned action was completed.

The remote run took 69.1 seconds; client elapsed time was 208.7 seconds. It used 52,166 input tokens and generated 658 tokens. All 72 outputs ended with EOS and none were truncated. A local import failure and a remote GPU-name rejection before model loading preceded the successful run. Receipts were preserved; billed cost is unknown.

This result does not establish a gain. There were no human reviewers; data rights and pretraining overlap remain unresolved, and the reserved test remains sealed. It is not a privacy review or authorization to train or admit records. The next evaluation should use the same prompts and user evidence with stronger task-system instructions before any further training.
