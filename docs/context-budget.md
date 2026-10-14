# Fixed context budget diagnostic

The diagnostic measures how much of the authored synthetic evidence survives the existing
whole-passage context packer at fixed UTF-8 byte ceilings. It reports retrieval and packing
evidence only; it does not measure answer quality, tokenizer tokens, or trained-model quality.
It uses the local signed-token-hash vectors and makes no model, network, or GPU calls.

Run it with:

```sh
PYTHONPATH=src .venv/bin/python experiments/evaluate_context_budgets.py --output /tmp/context-budget.json
```

The output path must be new. The CLI creates it exclusively and writes the diagnostic there;
it does not print fixture content. The report identifies the fixture by SHA-256 and contains
fictional query IDs, message IDs, citations, configuration, and metrics.

The default byte ceilings are 0, 256, 512, 1024, 2048, and 4096 bytes, with retrieval depth
5 for BM25, signed-token-hash vectors, and hybrid reciprocal-rank fusion (`rrf_k=60`). Each
query is searched once per mode. That same ranked hit list is packed at every ceiling. The
packer accounts for citation, role, timestamp, content kind, evidence text, UTF-8 encoding,
and newline separators. It selects whole passages in rank order, skips an oversized passage,
and may still include a later smaller passage.

Per-query rows include retrieved IDs and citations, retrieved relevant-message recall, and
for each byte ceiling the packed IDs and citations, exact packed bytes, packed relevant-message
recall, whether any authored sufficient evidence set survives, and whether the pack is empty.
Unknown queries have null recall and sufficiency fields. Summaries report answerable and
unknown query counts explicitly; answerable recall and sufficiency metrics are null when
there are no answerable queries. An empty pack is only a mechanical packing outcome and is
not calibrated semantic abstention.

The [recorded run](baseline-context-budgets.json) uses 14 answerable queries and
three unknowns. Sufficient-evidence counts among the 14 answerable queries are:

| UTF-8 bytes | BM25 | Hash vectors | Hybrid |
| ---: | ---: | ---: | ---: |
| 512 | 4 | 6 | 7 |
| 1024 | 12 | 9 | 11 |
| 2048 | 14 | 10 | 13 |
| 4096 | 14 | 10 | 13 |

These small authored-fixture results do not establish a general winner. All three
unknown queries still receive nonempty packs at 512 bytes and above. A matching
passage is not evidence that an unknown question can be answered.

Root independently replayed all 306 query/mode/budget combinations against the
previous committed packer and resolved 728 emitted citations to their exact
fictional source text. Packed text, selected-hit manifests, byte ceilings and
recomputed recall/sufficiency fields matched. No private sources or model calls
were used.
