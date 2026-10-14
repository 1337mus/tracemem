# Offline synthetic prompt-count catalog

`experiments/token_catalog.py` prepares complete prompt candidates for a future
remote token counter, then replays a caller-supplied count receipt locally. It
does not call a model, tokenizer, network service, or upload runner. It does not
score QA answers or compare retrieval modes.

## API

`build_catalog() -> dict` returns a versioned catalog with `kind`, `provenance`,
SHA-256 hashes for the fixture, prompts, and catalog source, and an `items` list.
Each item contains exactly `prompt_id` and complete two-message `messages`.
`prompt_id` is the SHA-256 of canonical JSON for those messages. Canonical JSON
uses sorted keys, compact separators, UTF-8, and no ASCII escaping. Items are
sorted by ID, so repeated builds from the same fixture and source are stable.

The fixed eight-case fictional fixture is parsed using the existing adapter and
prompt constants. Each question is ranked once per existing retrieval mode
(`bm25`, `vector`, `hybrid`), with `top_k=5`. For every ranking, the catalog
renders every rank-preserving subset of hits, including the empty context. This
covers all prompts the existing greedy whole-line packer can ask the counter
about, even if counts are nonadditive or nonmonotonic. The theoretical ceiling
is `8 questions × 3 modes × 2^5 subsets = 768` candidate prompts; identical
rendered messages are deduplicated. The current fixture produces 28 unique
prompts from 84 ranking/subset instances. Its rankings are the same across the
three retrieval modes, so this fixture cannot compare their retrieval quality.

To replay counts, pass an exact receipt envelope and a tuple of distinct
nonnegative integer budgets (one to eight values, each at most 2048):

```python
from experiments.token_catalog import build_catalog, pack_from_counts
import hashlib
import json

catalog = build_catalog()
canonical = lambda value: json.dumps(
    value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
).encode("utf-8")
catalog_sha256 = hashlib.sha256(canonical(catalog)).hexdigest()
receipt = {
    "schema_version": 1,
    "catalog_sha256": catalog_sha256,
    "counter_id": "operator-asserted-counter-name-and-version",
    "counts": [
        {"prompt_id": item["prompt_id"], "prompt_tokens": count_complete_messages(item["messages"])}
        for item in catalog["items"]
    ],
}
packed = pack_from_counts(receipt, (512, 1024))
```

`count_complete_messages` above stands for a future caller-provided counter; it
must count the complete system and user messages, not context text alone. The
receipt must include exactly one nonnegative integer count for every catalog
prompt ID, with no duplicates or unknown IDs. Its digest must match the
canonical complete catalog. `counter_id` must be a nonblank string, but is only
caller asserted: the replay cannot verify which tokenizer or provider produced
the numbers. Malformed receipts and budgets are rejected before packing. For
each question/mode/budget, replay uses the shared token packer and reports the
packed messages, selected citation IDs, declared prompt count and ceiling,
split, and retrieval mode. Development and test packs remain in separate
collections. A positive empty-prompt overhead that exceeds a budget is refused
by the shared packer.

## Privacy and limits

Catalog candidate items carry only prompt IDs and rendered messages; they do
not carry answer labels, answer-record IDs, abstention flags, split labels, or
local paths. The fixture, provenance, prompts, retrieval settings, and source
are fixed engineering inputs, not user data. There is no upload code in this
slice. Any future remote counter integration must send only these candidate
items and must independently define its network, retention, and credential
handling boundary.

The fixture contains fictional, authored agent-memory examples. Results do not
establish performance on real conversations or a representative QA benchmark.
This catalog is a prompt-count engineering diagnostic: it does not evaluate
answers, compare retrieval quality, provide tokenizer attestation, or establish
matched-token QA. Tests may use deterministic UTF-8 byte length as a clearly
labeled counter test double; byte lengths are not token measurements.

The next step, if authorized, is a separately reviewed remote-counter adapter
that accepts this exact catalog, returns the strict receipt schema, and records
provider/tokenizer provenance outside the caller-asserted `counter_id`. It must
pin the inference model/tokenizer and render the provider's actual chat template,
including special tokens and generation markers, before counting. The catalog
contains role/content messages rather than provider-serialized prompt bytes, so
the adapter must use that same pinned rendering for every count and retain its
provenance with the receipt. The offline replay API remains useful for
validating receipt binding and packing without that integration.
