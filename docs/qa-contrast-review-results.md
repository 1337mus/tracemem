# QA contrast review results

This is a descriptive, dev-only review of 46 unique tasks across six fictional source scenarios. The retrieval comparison covers 12 configurations: BM25, hybrid, and vector at complete-prompt token ceilings of 128, 256, 512, and 1024. Each configuration has six evaluation cells, for 72 cells total; repeated tasks make these observations correlated, not 72 independent samples.

## Annotation basis

- One procedurally blinded, self-attested AI agent supplied the original labels. There are no independent human labels.
- Original agent counts: source answer 32 correct, 13 incorrect, 1 uncertain; prompt support 38 supported, 7 unsupported, 1 uncertain.
- Root reread all 46 source/prompt/output pairs and audited the rationales. It agreed on 45 tasks. On `qa_task_504341d3c1ec490871ea295e6906ad62be2d6b43f5992088ee00ab76aaf0f7de`, the output states the tour changed to 3:00 PM, then concludes UNKNOWN. Root treats that concluding abstention as incorrect and unsupported.
- Root counts: source answer 32 correct, 14 incorrect; prompt support 38 supported, 8 unsupported. Root had access to the original labels and rationales, so its review is an unblinded derivative adjudication, not an independent second sample or consensus.

## Retrieval results under root adjudication

Answerable items judged correct out of four by complete-prompt token ceiling:

| Retrieval | 128 | 256 | 512 | 1024 |
|---|---:|---:|---:|---:|
| BM25 | 0/4 | 1/4 | 3/4 | 3/4 |
| Hybrid | 0/4 | 1/4 | 2/4 | 3/4 |
| Vector | 0/4 | 2/4 | 2/4 | 3/4 |

Every configuration also answers both unknown items correctly (2/2); total correct out of six is the answerable score plus two. The frozen literal report remains 0/4 on answerable items and 2/2 on unknown items in every group. These are distinct views: these agent semantic judgments accept supported paraphrases, while the literal report is unchanged.

The semantic review shows that paraphrased evidence can support correct answers, including the dried-fig snack and conditional brass-latch instruction. Missing retrieved evidence leads to UNKNOWN even when the complete source contains the answer; that is a retrieval limitation, though abstention can still be justified by the supplied prompt. It also flags unsupported abstention when the prompt already states the answer, and a container-versus-content error when “compact tin” is given as the snack.

The table uses root labels. In the original blinded labels, only BM25 at 1024 differs: one item is uncertain instead of root’s incorrect, while the correct count stays 3/4. Preserve that uncertainty when citing the original file.

These small, correlated fixtures support descriptive diagnosis only. They do not establish retrieval superiority, statistical significance, or training readiness.

The response JSON and notes remain local under `.local/`. See the [review accounting](qa-contrast-review-accounting.md) and [original answer results](qa-contrast-answer-results.md).

The [validation snapshot](validation-qa-contrast-review-accounting.json) pins the
report and response hashes. Reproducing these actual judgments requires the local
response documents. The code and fictional automated tests run from the repository.
