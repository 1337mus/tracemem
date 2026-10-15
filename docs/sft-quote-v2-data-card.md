# Quote-v2 dataset data card

Updated 2026-10-15.

## Summary

- **Files:** [`train-seeds.json`](../datasets/sft-quote-v2/train-seeds.json), [`train.json`](../datasets/sft-quote-v2/train.json), [`dev-seeds.json`](../datasets/sft-quote-v2/dev-seeds.json), and [`dev.json`](../datasets/sft-quote-v2/dev.json).
- **Format:** version 2, with fictional user, assistant, and tool messages and provisional expected claims grounded in exact user-conversation quotes.
- **Train:** 48 authored families expanded to 768 cases (16 variants per family). Nine families have empty targets, contributing 144 empty-target cases (18.75%).
- **Dev:** 24 authored families expanded to 24 cases (one variant per family). Six cases have empty targets (25%).
- **License and provenance:** MIT; newly authored fiction. No private conversation corpus was used to author or expand these fixtures.
- **Intended use:** a small development and training-format diagnostic for quote-grounded claim extraction and abstention. It is not a held-out test set or a production quality measure.
- **Readiness:** Gate A is unmet. There has been no independent human annotation or review.

## Authoring and partitioning

SHA-256 for the files built and reviewed on 2026-10-15:

| File | SHA-256 |
| --- | --- |
| `train-seeds.json` | `f894fb6fc72956b96b8f3f12cd72bb2411dd88d34547d05f8402c4b63cdc683d` |
| `train.json` | `1f8ac57d19841fba54cb2e73b1cd5ee2e262895a3a29c0cf0904af8cc3b71bf2` |
| `dev-seeds.json` | `579ba26d7333d60b7f774affaa9bdb5d25c35e2089ad92f01fb7b7eadef29914` |
| `dev.json` | `1357ce09175e0d40b5ff5fb8eb5b6aa83a7f713a5147d55df09fdde4afd20030` |

The train and dev examples were authored as separate families before deterministic expansion. The builder uses seed 23 and checks case, scenario, template, entity, and full-conversation isolation between the two files. Train families expand across shared values in the person, product, object, place, and date banks; these variants are correlated. Dev has one variant per family, so its family count and example count happen to match. Neither expanded-row counts nor family counts represent independent phenomena or a population sample. Shared templates, wording patterns, and bank values limit diversity.

Dev messages are short: 22 of 24 cases contain one message and two cases contain two messages; total transcript text ranges from 36 to 120 characters per case. Dev is for development only. It has already been reviewed during curation and may be revised based on that review; it is adaptive and must not be presented as an untouched evaluation set. Before this build, an unblinded agent review examined all 24 dev families, all 48 train families, and 12 train examples with access to the authored messages and labels. That was an agent review, not independent human validation or annotation. There have been zero human reviewers.

## Labels and validation

The labels and messages are agent-authored and provisional. They represent intended extraction targets for these fictional examples; they do not establish that a model should make the same abstraction choice in every context. Empty targets are authored for question-only, hypothetical, quoted-document, assistant-only, or tool-result-only inputs where the user has not stated an eligible personal fact.

The validator checks the data shape, IDs, dates, split tags, unique exact quote occurrences, source role and content kind, and literal date bounds. The builder also checks deterministic expansion and train/dev isolation. These checks do not judge whether a claim is semantically entailed, complete, natural, or the only reasonable paraphrase. They do not establish annotation agreement, calibrated abstention, legal review, model performance, or generalization. The fixture must not authorize automated memory writes, choose a production threshold, or support claims about real users.

## Limitations and future verification

This compact authored set has short conversations, one dev variant per family, and substantial correlation among train variants. A single reviewed example can expose a family template's pattern across many expanded cases. Before any reported evaluation, verify the exact checked-in hashes and builder output, inspect examples after each data change, and obtain independent human review with a documented protocol. Keep a separately authored and reviewed test set untouched during training and development.
