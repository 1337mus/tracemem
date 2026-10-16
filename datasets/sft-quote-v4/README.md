# Quote-v4 dataset provenance

The frozen training selection contains 768 rows across 72 correlated families: 140 empty-target rows and 696 expected claims. Review changed 64 rows in 7 families. The source corpus and training freeze are unchanged. V4T identifiers change token strings, so account for them in comparisons with earlier data. These counts describe selected fact coverage, not an exhaustive inventory.

The frozen evaluation has 12 dev cases (20 claims, 3 empty targets, 48 messages) and 24 test cases (53 claims, 6 empty targets, 104 messages). Every case has two sessions. Test messages comprise 17 four-message cases, 6 five-message cases, and 1 six-message case. E01–E09 were replaced with varied scenarios; E10 remains claim-bearing and unchanged; E19–E24 are abstentions. Root reviewed all 36 final sources and expected targets, including narrow own-quote fixes and the final D03 claim-precision correction. Original pre-correction bytes remain preserved locally.

The first review pass was independent within the same model family, followed by root-agent review; there were zero human reviewers. The literal audit across all 804 cases found no scenario, entity, or template ID overlap, exact duplicate, or cross-split Jaccard match of 0.8 or higher. This is not semantic proof. The evaluation is diagnostic only; family isolation and generalization are unproven. The quote-v3 test set has already been spent and is not an untouched holdout.

See the [quote-v4 protocol](../../docs/sft-quote-v4-protocol.md), [training target review audit](../../docs/training-target-review.md), and [preparation validation record](../../docs/validation-sft-quote-v4-preparation.json) for the frozen evaluation hashes and run state.
