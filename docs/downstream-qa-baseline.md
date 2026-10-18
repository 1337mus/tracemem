# Downstream QA comparison

Changing only the answering instructions, we tested base Qwen3-4B on the same 24 adaptive public questions, with the same three context conditions and user evidence. Only the system instructions changed. This remains a descriptive QA diagnostic: it is not trained-memory efficacy, a held-out or official benchmark, a full-history retrieval evaluation, or a privacy review.

| Condition | Source-answer correct | Prompt-grounded | Grounded and correct | Abstentions | Unsupported additions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid retrieval | 5→13/24 | 10→16/24 | 5→13/24 | 18→7/24 | 1→4 |
| Empty context | 0→0/24 | 24→24/24 | 0→0/24 | 24→24/24 | 0→0 |
| Oracle evidence | 7→12/24 | 7→11/24 | 6→10/24 | 17→6/24 | 1→7 |

Counts show baseline→candidate. On the 22 unqualified references, grounded-correct answers changed from 5/22 to 12/22 for hybrid retrieval and 6/22 to 10/22 for oracle evidence. On the two qualified references, hybrid changed from 0/2 to 1/2; empty context and oracle evidence remained 0/2. These subsets remain included in the full24 totals.

Paired full24 joint-answer outcomes show eight hybrid gains and no regressions; oracle evidence had five gains and one regression; empty context was unchanged. The conservative label for one borderline preference response gives hybrid 13/24 joint answers. If that response is accepted as personalized, the sensitivity count is 14/24; the unsupported-addition cap still fails.

The joint-answer targets set before the run passed for hybrid (13/24 versus 9 required) and oracle evidence (10/24 versus 10 required), but both conditions failed the unsupported-addition cap of one (4 and 7). Empty context passed its exact abstention/joint-answer and unsupported-addition targets. No promotion is supported.

The candidate remote elapsed time was 127.4 seconds, including 7.5 seconds of tokenization, 17.7 seconds of model loading, and 102.2 seconds of generation (138.8 seconds client elapsed). It used 63,686 input tokens and generated 1,977 tokens. Baseline remote elapsed time was 69.1 seconds, with 52,166 input and 658 generated tokens. All 72 candidate outputs ended with EOS; none were truncated. There were zero human reviewers. Root accepted the agent labels on all nine disagreement rows; provisional annotations are preserved locally.

This prompt-only base-model comparison does not establish trained-memory efficacy or memory-pipeline benefit. The sample uses adaptive public questions; pretraining overlap is unknown, sessions are not independent Raj sessions, and data rights remain unresolved. The reserved test remains sealed. Further work should start with an independently sampled set and human-reviewed labels, using matched baseline and candidate conditions before making any efficacy or promotion claim.
