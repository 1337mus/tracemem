# Training target reviews

These JSON files record provisional, agent-authored target reviews for the training seed sources. They do not modify those sources. Each review is bound to the exact source bytes and each family entry includes the original-target hash, disposition, rationale, and selected literal source/target fact spans.

| Review | Source | Families | Selected fact annotations |
| --- | --- | ---: | ---: |
| `v2-train-review.json` | `../sft-quote-v2/train-seeds.json` | 48 | 75 |
| `v3-train-review.json` | `../sft-quote-v3/target-train-seeds.json` | 24 | 45 |

The review status is `agent_reviewed_provisional`; there were zero human reviewers. The seven target corrections and audit limitations are described in [the training-target audit](../../docs/training-target-review.md).

The validator checks raw-source hashes, literal quoted spans, provenance, and selected-fact coverage. It does not prove semantic entailment or exhaustive recall. Use `training.build_reviewed_quote_train` to build the 768-row reviewed corpus into a new output directory; the command and all five required input paths are in the audit document. The builder preserves the original sources and historical training file, validates their deterministic rebuild, and namespaces output IDs with `V4T-`.
