# Next experiment: learn to check a claim's meaning

The last verifier kept three bad claims. This dataset isolates the kinds of
mistake it missed: a claim can quote the right words and still change who said
something, remove a condition, or invent a cause.

Each conversation has two candidate memories. One preserves the source; the
other changes one relation. Both cite exactly the same real span in that
fictional conversation. This prevents a citation check from solving the task.

The [data card](verifier-pairs-data-card.md) describes the seed corpus. It is
small and labeled by AI agents. Its test split is visible to the authors and
reviewers. It can support a diagnostic experiment, not a generalization or
product-quality claim.

## Two answers, kept separate

The new research format asks for `support` and `eligibility`, both as string
labels. `uncertain` is a valid answer on either axis. A statement about a
fictional character can be supported but ineligible for personal memory.
Whether to remember someone's writing exercise remains uncertain under this
policy. Uncertain eligibility stays outside decisive eligibility scoring.
The existing boolean runtime verifier is unchanged.

## Before spending GPU time

1. Review every seed pair, record remaining disagreements, and freeze the
   dataset and prompt hashes. Export only training rows. Keep dev and test
   targets out of the remote payload.
2. Implement and test a strict two-axis response scorer. Compare no filter,
   an untuned model, and the same model with a verifier adapter. Use the same
   prompts, decoding, input limits, and output schema in both model arms.
3. Freeze the exact run configuration, training budget, development decision
   rule and test-release rule before observing model outputs. Review the
   payload and check personal Modal state immediately before launch.

For the seed diagnostic, the proposed development rule is: catch all six bad
claims, retain at least five of six good claims, return valid untruncated
responses for every candidate, and catch more bad claims than the untuned arm
without retaining fewer good claims. These thresholds are a proposal until
bound to the reviewed scorer and exact run configuration. They do not select a
product model. If the diagnostic fails, record it before considering another
change; do not silently edit labels, prompts or thresholds.

Dev and test each have two eligible, two ineligible, and two uncertain
families. Score scope independently on the eight candidates with decisive
scope labels, and report the four uncertain candidates separately. A support
failure must not silently become an ineligible-scope label.

Report support retention and bad-claim catch rate separately from eligibility.
Also report malformed responses, truncations, paired outcomes by scenario,
per-phenomenon counts, output tokens, and added execution time. Uncertain
support labels are excluded from decisive support scoring. This paired dataset
has no exhaustive required-fact labels, so it cannot measure extraction recall
or selected-fact retention. A later unchanged-generator evaluation is needed
for those measures.

The current slice prepares data and CPU tooling. It launches no training or
inference. The older failed experiment and the sealed longer-conversation
candidate remain unchanged. A future runtime must use the new string schema
explicitly; do not pass it through the older boolean parser.
