# Development-only token-budget replay

This offline experiment measures which whole evidence passages survive fixed
**complete input prompt** ceilings. It uses the six fictional development cases
from [contrast preparation](qa-contrast-preparation.md) and the exact 399-prompt
[measured tokenizer receipt](qa-contrast-token-counter.md). No inference or
training is part of replay. References are agent-authored and provisional;
there are zero independent human labels and no held-out test cases here.

The default command is:

```sh
PYTHONPATH=src:. .venv/bin/python -m experiments.replay_qa_contrast_tokens
```

The receipt's exact raw hash is checked before parsing. Regenerated fixture,
catalog and payload hashes and the pinned tokenizer provenance must also match.
The counter looks up the hash of complete system/user messages; missing prompts
fail rather than estimating tokens. Local model or tokenizer packages are not
needed. Original experiments and receipts remain unchanged.

Each question is retrieved using BM25, weak hash vectors and reciprocal-rank
fusion (top five, RRF constant 60). The shared token packer walks the ranked
hits, retains a whole evidence line when it fits, and continues after an
oversized candidate. It preserves citation order. References and evidence-label
metadata stay outside model messages. Natural source passages may contain the
answer.

## Results

Macro authored-label recall over the four answerable development questions:

| Complete prompt ceiling | BM25 | Hash vector | Hybrid | Evidence lines per prompt |
| ---: | ---: | ---: | ---: | ---: |
| 128 | 0.000 | 0.000 | 0.000 | 0 |
| 256 | 0.250 | 0.250 | 0.125 | 1 |
| 512 | 0.750 | 0.875 | 0.750 | 2 |
| 1,024 | 1.000 | 0.875 | 1.000 | 5 |

All 72 cells are feasible. Empty-context prompts cost 82–96 tokens; the
1,024-token ceiling fits all five passages (848–907 tokens). Three packing
decisions accept a later passage after rejecting an earlier one. The two
unknown questions are excluded from label recall in each mode/ceiling cell,
with an explicit reason. Their recall is undefined, not zero. The API can
also diagnose smaller ceilings: an empty prompt that exceeds the ceiling
produces an explicit infeasible cell and is excluded from recall aggregation.

Root independently reconstructed all 72 outcomes using subset masks from the
previously audited catalog and measured counts, without calling the replay
implementation or shared packer. Validation and artifact hashes are recorded in
[the validation receipt](validation-qa-contrast-token-replay.json).
The [compact cell report](qa-contrast-token-replay-dev.json) retains selected
IDs, counts, exclusions and provenance; the CLI reconstructs complete messages.
Full checks passed 496 tests (one optional skip), Ruff on 108 files and strict
mypy on 32 runtime files. Nine focused installed tests passed, and installed
CLI output matched root exactly. All 37 runtime members match the existing
wheel; historical experiments and receipts are unchanged.

These measurements distinguish retrieval rankings and context capacity on this
small development fixture. Recovered authored labels do not establish semantic
evidence sufficiency, answer accuracy, calibrated abstention or superiority of a
retrieval method. Repeated cells are correlated. Hash vectors are lexical
features, not semantic embeddings. Gate A remains unmet; model selection and
training have not begun.

The next useful experiment is a separately reviewed, source-only QA request
bundle for these frozen cells, with references kept in a separate local file.
Any bounded synthetic inference must preserve the answering model, template,
decoding and output limit across cells. This development fixture cannot replace
independent evaluation.
