# Prospective verifier transfer diagnostic

This frozen plan checks whether the unchanged paired-verifier adapter transfers
to a broader, newly authored fictional set about attribution and conditions. It
is an agent-authored development diagnostic with zero human reviewers and no
population-generalization claim. The accompanying
[data card](verifier-transfer-data-card.md) describes the 24 cases and 48
candidates. This protocol was frozen before outputs were collected; see the
[results](verifier-transfer-results.md).

Compare pinned Qwen3-4B prepared in NF4 with the frozen adapter, using the same
inputs and prompt/token IDs. Include a no-filter keep-everything control.
Use greedy decoding with thinking disabled, at most 2,048 input tokens and 128
new tokens, with no truncation. Do not train further or change policy after
seeing outputs.

Report supported claims retained out of 24, unsupported claims explicitly
rejected out of 24, unsupported claims still accepted out of 24, uncertain
unsupported claims out of 24, and valid untruncated outputs out of 48. Score
eligibility separately and keep uncertain references separate. The prospective
promising criteria are at least 22 supported claims retained, at least 23
unsupported claims rejected, all 48 outputs valid and untruncated, strictly
fewer unsupported claims accepted than the matched base, and no fewer supported
claims retained than the base.

This is inference-only, with no private data. The frozen run allows one A10G GPU,
a 1,200-second maximum, and zero retries. Review all 48 claims and exact evidence
before freezing the corpus, renderer, runner, and payload hashes. A positive
result cannot retroactively satisfy the earlier failed criterion or authorize
release of the earlier test set.
