# Development-only QA contrast preparation

This is a new, agent-authored fictional development fixture, separate from the
previous eight-case QA fixture and its dev/test results. It contains six cases,
12 sessions and 60 turns: four answerable questions and two unknown controls.
The cases cover time corrections, speaker attribution, conditional scope,
two-source recall, missing details and quoted untrusted instructions. References
are provisional; no independent human annotation or held-out test exists.

The final fixture SHA-256 is
`b4fb6ea6242c294574ca9ed15549f5b524a77d3154b4c87b9831d2f357beefed`.
Before the first retrieval, root source review corrected one assistant line
that referred to the downstream evaluation question. The original 16,831-byte
fixture (SHA-256
`0186b52bc4f5d363d25b76ab10bf6c01d72f01a9fec3ee2fda596d67e7b27263`),
original freeze receipt, exact single-line erratum and final 16,841-byte freeze
receipt are preserved locally. Root independently verified that precise change.
No ranking outputs were inspected before this semantic correction. The fixture
was not rewritten to obtain a preferred retrieval result.

`experiments.prepare_qa_contrast.prepare_contrast(raw)` returns a source-only
prompt catalog, separate provisional references, and a retrieval/packing audit.
It uses isolated per-question stores, BM25, 384-dimensional signed hash vectors
and hybrid reciprocal rank fusion (`rrf_k=60`, top five). Hash vectors are weak
lexical features, not neural semantic embeddings. Existing runtime and historical
experiment modules are reused without modification.

The catalog enumerates every order-preserving subset of each ranking and
deduplicates complete system/user prompts by canonical SHA-256. The audit uses
whole-evidence-line packing at UTF-8 byte ceilings 0, 512, 1024 and 2048. Its
turn-label coverage describes the explicit authored labels, not semantic
evidence sufficiency. Unknown cases have undefined label recall. References and
answer-label metadata stay outside answering messages; ordinary source text
can naturally contain the answer.

The reviewed run produced 399 unique prompts, 18 question/mode rankings and
72 byte-packing cells. Ordered mode rankings differ for 6/6 cases. Positive
byte ceilings change retained citation membership within at least one fixed
mode for 6/6 cases; this count excludes differences caused only by ranking
order. Retained explicit-label coverage differs across positive mode/budget
cells for 3/4 answerable cases. The two unknown cases are excluded from recall,
with known label counts zero and recall undefined.

| UTF-8 byte ceiling | BM25 macro label recall | Hash-vector macro label recall | Hybrid macro label recall |
| ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 0 |
| 512 | 0.500 | 0.500 | 0.375 |
| 1024 | 0.875 | 0.875 | 0.875 |
| 2048 | 1.000 | 0.875 | 1.000 |

Each value averages four answerable questions; the two unknowns are exclusions.
These are explicit authored-turn-label recalls, not answer accuracy, calibrated
precision, exhaustive support labels or a method ranking. The same cases repeat
across every cell. Root reconstructed record identities/revisions from raw source
fields and independently checked all subset messages, greedy byte packs and
label outcomes, without adapter or production packing calls.

Full-prompt token ceilings 128, 256, 512 and 1024 were frozen as configuration
only during this preparation. A later, separately reviewed
[CPU-only counter](qa-contrast-token-counter.md) measured all 399 candidate
prompts at 82–907 tokens. Token-budget packing, model answers and QA accuracy
remain unmeasured for this fixture. The historical pinned Modal counter remains
unchanged. Do not substitute byte lengths for token counts or upload private
conversations.

The six scenarios and repeated mode/budget cells are a development diagnostic.
They do not establish dataset diversity, calibrated abstention, model selection,
training value or Gate A. Any later model run must retain exact payload hashes,
raw receipts and these provisional annotation labels.

## Validation and reproduction

Root's final `make check` passed 479 tests (one optional skip), Ruff on 104
files, and strict mypy on 32 runtime source files. Seven focused tests passed
against the existing installed wheel outside the checkout with `PYTHONPATH`
and `PYTHONHOME` unset. That installed replay reproduced catalog, references
and audit byte-for-byte, with all 37 installed runtime members unchanged.
The historical 28-prompt catalog still has its original canonical hash.
RED/GREEN logs, setup failures, local artifacts and exact hashes are retained;
see [the validation receipt](validation-qa-contrast-preparation.json).

This prints the development contrast counts without model or network calls:

```sh
PYTHONPATH=src:. .venv/bin/python - <<'PY'
from pathlib import Path
from experiments.prepare_qa_contrast import prepare_contrast
catalog, references, audit = prepare_contrast(
    Path("examples/synthetic/qa-contrast-dev-v1.json").read_bytes()
)
print(len(catalog["items"]), audit["contrast_counts"])
PY
```
