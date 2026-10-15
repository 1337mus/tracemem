# Cross-session MiniLM comparison

Updated 2026-10-15. This small diagnostic asks whether MiniLM vectors improve
retrieval of explicitly labeled evidence turns across sessions, compared with
BM25, hash vectors, and their hybrid ranking. It reuses the six fictional
questions and 72 turns documented in the [cross-session data card](cross-session-test-data-card.md).

## What was sent and measured

One remote call embedded 78 unique texts: the 72 turns and six questions. No
separate answer keys or label fields were sent; the text itself necessarily
contains facts relevant to the answers. The pinned model was
`sentence-transformers/all-MiniLM-L6-v2`, revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (384 dimensions). The input SHA-256
is `dc7e301a9acb0e0d26293c2d8fa80bf8ebe838ecd543a7917a64bcd8f3cdfedf`; the
embedding payload SHA-256 is
`95158af854e806eb14c19ee4f3635446c912d5a1efddd7933f7e639f5f43aada`. The raw
[remote receipt](../examples/synthetic/minilm-cross-session-remote.json) has
SHA-256 `870abdfdce0c6488d7db001fd4f7c8d345cc1b2e96f9d144a18e5fd9c71f2b77`.

The call used one Tesla T4. It reports 0.789242 seconds of GPU compute,
13.476377 seconds for the remote function, 30.813097 seconds of client wall
time, and peak GPU memory of 59,987,456 bytes. These timings and memory use do
not state or imply a billed cost. No private data or local weights were used;
this was inference, with no training.

## Retrieval results

The table reports macro recall over the four answerable questions and the
number of those questions with every labeled turn present in the top k. The two
`UNKNOWN` questions are excluded from both scores.

| Method | Recall k=1 / 3 / 5 | Complete coverage k=1 / 3 / 5 |
| --- | --- | --- |
| BM25 | 0.50 / 1.00 / 1.00 | 0/4 / 4/4 / 4/4 |
| Hash vector | 0.50 / 0.75 / 1.00 | 0/4 / 2/4 / 4/4 |
| MiniLM vector | 0.50 / 0.875 / 1.00 | 0/4 / 3/4 / 4/4 |
| BM25 + hash vector | 0.50 / 1.00 / 1.00 | 0/4 / 4/4 / 4/4 |
| BM25 + MiniLM vector | 0.50 / 1.00 / 1.00 | 0/4 / 4/4 / 4/4 |

At k=3, MiniLM misses one labeled turn for the Blue Shingle report question.
Both `UNKNOWN` questions received retrieval hits under every method. That is
not an abstention accuracy result: no QA or abstention evaluation ran. MiniLM
does not improve on BM25 at these cutoffs.

## Limits and replay

These are six small authored cases with provisional agent-only labels and zero
human ground truth. Their assigned `test` split is bookkeeping, not held-out or
semantically isolated evidence. Turn-label recall measures retrieval of those
labels only; it does not measure answer quality, evidence sufficiency, or
generalization. The comparison does not establish that any method is generally
better.

Recreate the offline report from the checked-in receipt. Use a fresh output
path because the command refuses to overwrite an existing report:

```sh
PYTHONPATH=src .venv/bin/python -m experiments.evaluate_cross_session_embeddings \
  --receipt examples/synthetic/minilm-cross-session-remote.json \
  --output .local/UNIQUE-report.json
```

The reviewed launch source hash was
`043b2c34e8d226c5dd77fb06a6b254f36886edb0217d841218570da19f89942e`; the exact
source is retained privately as `.local/cross-embedding/reviewed-runner.py`.
Root checked AST identity across later Ruff-only formatting, while the receipt
validator pins the exact reviewed launch-source hash. No remote launch command
is included here.

The final [replay report](cross-session-minilm-retrieval.json) and
[validation record](validation-cross-session-minilm.json) are locally verified.
This does not establish signed/private delivery or exact-head hosted CI; check
`.local/active-cycle.json` for those authoritative states.
