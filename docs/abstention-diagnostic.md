# Synthetic abstention score diagnostic

This is an exploratory replay of the authored retrieval fixture. It reuses the
vectors returned by the embedding run on CPU. It makes no model calls and does
not add an abstention threshold to the runtime.

The embedding payload SHA-256 is `0f402a3474d7c94e979553a2623f58ef6fd72bc072218feebe872ebe10b7ae53`.
Its input SHA-256 is
`76862babda5e6e26be1ed14f5265c64e43627a54b1499dba47585605396637f3`. The model
is `sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The fixture has 40 records and 17
queries: 14 answerable and 3 labeled unknown.

The answerable queries have top-1 cosine scores from 0.351446 to 0.818298. The
unknown queries range from 0.115554 to 0.248817. A cutoff of 0.249 rejects all
three unknowns and none of the 14 answerable queries in this same set. This
cutoff was selected after inspecting these 17 cases. It is an in-sample
observation, not a calibrated threshold or held-out performance result. The
baseline result remains 0 of 3 unknown queries abstained; the exploratory
cutoff is separate and was not added to runtime behavior.

Top-1 to top-2 margins do not separate the labels. Answerable margins range
from 0.002672 to 0.170039; unknown margins range from 0.008034 to 0.013650. A
margin cutoff of 0.014 rejects all three unknowns but also rejects q01, q07,
q08, and q11.

| Unknown query | Top-1 cosine | Margin | Top record |
| --- | ---: | ---: | --- |
| q14 | 0.244719 | 0.013650 | m31 |
| q15 | 0.115554 | 0.010236 | m40 |
| q16 | 0.248817 | 0.008034 | m11 |

Reproduce the CPU-only retrieval scores from the repository root:

```sh
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
import tempfile
from pathlib import Path
from tracemem.benchmark import load_synthetic_data
from tracemem.remote_vectors import PrecomputedEmbedder
from tracemem.retrieval import Retriever
from tracemem.storage import EvidenceStore

receipt = json.loads(Path("examples/synthetic/minilm-remote.json").read_text())
assert receipt["payload_sha256"] == "0f402a3474d7c94e979553a2623f58ef6fd72bc072218feebe872ebe10b7ae53"
records, queries, input_hash = load_synthetic_data()
assert input_hash == receipt["input_sha256"]
embedder = PrecomputedEmbedder(
    model_id=receipt["model"]["model_id"],
    revision=receipt["model"]["revision"],
    dimensions=receipt["model"]["dimensions"],
    vectors=receipt["vectors"],
)
with tempfile.TemporaryDirectory() as directory:
    with EvidenceStore(Path(directory) / "evidence.sqlite") as store:
        store.ingest(records)
        retriever = Retriever(store, embedder)
        for query in queries:
            hits = retriever.search(query["query"], mode="vector", k=2)
            top1 = hits[0].score if hits else None
            margin = top1 - hits[1].score if len(hits) > 1 else None
            print(query["query_id"], top1, margin)
PY
```
