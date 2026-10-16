# Can a verifier reject bad memories without losing useful ones?

This is one development experiment, using the existing Qwen3-4B `quote-v2`
outputs. It does not train another extractor. The earlier prompt comparison
found no gain from a larger generator or broader prompt, so this experiment
checks the remaining candidates before considering another training change.

## Fixed comparison

Compare three stages on the same 12 fictional development conversations:

1. The generator's original response, with strict JSON parsing.
2. The same claims after exact citation, source-role, and literal-date checks.
3. The surviving claims after Qwen3-8B judges support and memory eligibility.

Do not strip fences, repair claims, infer missing citations, or regenerate
rejected output. Reject malformed or truncated generator responses. Check each
claim's citation separately. Cases with no surviving candidates need no model
call. Preserve original claim indices and all rejection reasons in a local
audit. The verifier may discard a claim but cannot rewrite it.

Use pinned Qwen3-8B, BF16, greedy decoding, thinking off, a 4,096-token input
limit and 1,024-token output limit. Never truncate inputs. There is one
synthetic-only Modal call, ten nonempty batches, and no automatic retry. Send
only the source messages and surviving candidates in the prompts, plus bounded
inference metadata. Targets and existing review labels stay local. Freeze
payload and source hashes before launch. Weights stay on personal Modal.

A malformed or truncated verifier response rejects that whole batch. Otherwise
keep only claims marked both `supported` and `eligible: true`. `uncertain`
means rejection for this experiment. Report parsing failures separately, so
empty output from a failure is distinguishable from a successful judgment.

## What would count as useful?

The existing, provisional agent annotations identify 36 supported, eligible
claims and three bad claims among the 39 candidates passing the exact gate.
Two bad claims concern ineligible exercise material; one drops a reported
source. This is a very small, adaptive sample.

Before seeing verifier outputs, set the development criterion to:

- Reject all three previously labeled bad claims.
- Retain at least 35 of the 36 previously labeled good claims.
- Produce valid, complete verdicts for every nonempty batch, with no truncation.

Report the full retained/rejected counts against the frozen labels, not just
precision among survivors. Audit selected-fact retention separately by reading
the surviving statements; good-claim retention is not exhaustive fact recall.
Record added wall time, generated tokens, and GPU time. Failure does not trigger
an automatic prompt revision or another training run.

The labels come from AI agents in one model family, with zero human reviewers.
An 8B verifier is a measured filter, not independent ground truth. Agreement
with those labels on reused development cases cannot establish generalization,
99% support precision, QA benefit, or a product model selection. Keep the longer
candidate sealed until the full development policy is frozen and its versioned
long-conversation contract is reviewed. Fresh evaluation and independent human
annotation remain necessary.

See the [baseline results](quote-baseline-results.md) for source receipts and
annotation limits. Current operational state is in `.local/active-cycle.json`.
