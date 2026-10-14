# Token-budget context packing

`tracemem.token_budget.pack_context_by_tokens` packs ranked search hits as whole
citation/evidence lines under a caller-supplied prompt-token ceiling. It does not
load a tokenizer, call a model, estimate tokens from text length, or persist data.

```python
from tracemem.token_budget import pack_context_by_tokens

system_prompt = "Answer from the cited evidence."
question = "What did the user decide?"


def count_final_prompt_tokens(context: str) -> int:
    final_prompt = render_chat_template(
        system=system_prompt,
        question=question,
        context=context,
    )
    # This must use the same tokenizer, model revision, and options as inference.
    return tokenizer_for_the_inference_route.count_tokens(final_prompt)


packed = pack_context_by_tokens(
    ranked_hits,
    max_prompt_tokens=8_000,
    count_prompt_tokens=count_final_prompt_tokens,
    counter_id="inference-route/model@revision;tokenizer@revision;template-v3",
)
```

The callback receives the proposed context text. Its closure must render and count
the complete final prompt, including fixed system and question text, chat-template
markers, and any other prompt overhead. For reproducible results, keep the counter
deterministic for a given context and pinned prompt, model, tokenizer, and options.
The packer measures the empty context once so it can reject a budget that cannot
fit the prompt by itself. Each later call measures all previously accepted lines
plus the new line, including their newline separators. This is deliberate: token
counts can include shared overhead and need not add up from isolated lines.

`counter_id` is a required, non-empty caller-asserted provenance label. It should
identify the actual counting route, tokenizer/model revisions, options, and prompt
template version. The packer stores the label in its result but cannot verify that
the callback really uses those settings.

The packer walks hits in their given rank order. It accepts a complete line when
the callback count is within the ceiling, skips a line that is too large, and
continues trying later hits. It never truncates evidence or citations. This is a
greedy bounded selection, not an optimal packing algorithm; it makes no
token-additivity or monotonicity assumption. The callback is invoked once for the
empty context and at most once per hit. Callback exceptions propagate, and every
returned count must be a non-negative integer (booleans are rejected).

`prompt_tokens` reports the last accepted callback count, or the empty-prompt
count when no hit is selected. This describes only the caller's measurement. It
does not attest that the count matches a model's actual inference behavior. This
offline foundation has no tokenizer integration and makes no matched-token QA
claim. Use a controlled remote tokenizer route only for reviewed synthetic-only
inputs covered by current privacy authorization. Private source text must remain
local and is not authorized for remote counting. If no approved local counter is
available for private contexts, do not route those contexts to a remote service.
For authorized synthetic-only inputs, pass the remote route as the callback and
record it in `counter_id` with the model, tokenizer, prompt template, and inference
options.
