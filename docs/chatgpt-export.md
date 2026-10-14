# ChatGPT JSON export adapter

The adapter reads an explicitly supplied local JSON export and yields immutable
`EvidenceRecord` values. It does not discover exports automatically or access
network services.

```python
from tracemem.adapters import AdapterStats
from tracemem.adapters.chatgpt import read_chatgpt_export

stats = AdapterStats()
records = list(read_chatgpt_export("conversations.json", stats=stats))
print(len(records), stats.malformed_records, stats.unsupported_records)
```

The accepted fixture contract is a UTF-8 JSON list. Each conversation has a
nonblank string `id`, a nonblank string `current_node`, and a `mapping` object.
Each selected node has a `parent` that is either `null` or another mapping key,
and a `message` that is either `null` or a message object. The reader follows
only the `current_node` parent chain, then emits its text from root to leaf.
Other branches are ignored regardless of mapping order, timestamps, or child
relationships. A missing node, invalid pointer, cycle, malformed selected
message, or repeated message ID invalidates that conversation and emits no
records for it. Duplicate nonblank conversation IDs reject the entire export
before any record is yielded.

Branch filtering applies to each export read. Ingesting a later snapshot into
the same `EvidenceStore` does not remove evidence from a sibling branch that
was selected by an earlier snapshot. Use a fresh derived store for each export
snapshot until the store supports explicit branch membership and lifecycle.

Only messages whose `content.content_type` is `"text"` and whose `parts` value
is a list are supported. Each string part becomes a separate record, including
empty strings, with its original list index. Text is preserved exactly; it is
not trimmed, concatenated, or normalized. A nonblank `message.id` supplies the
message identity; otherwise the node key is used. Optional finite numeric
`create_time` values become UTC ISO 8601 timestamps. The source locator is a
SHA-256 hash of the resolved path, and `line_number` is `None` because this is
not a JSONL source. Preserved strings remain citeable by Python string
codepoint offsets, and edits create new immutable evidence revisions.

Null messages, non-text content types, and non-string parts are skipped and
counted as unsupported. Malformed conversation, chain, author, message, or text
content shapes are counted as malformed. Recognized roles are `user`,
`assistant`, `tool`, `system`, and `developer`; other string roles become
`unknown`. User and unknown text is marked `mixed_untrusted`, assistant text as
`conversation`, tool text as `tool_output`, and system/developer text as
`instruction`. Every imported string remains untrusted evidence; the adapter
does not execute or interpret it. It never serializes images, attachments,
structured blocks, or tool-call payloads into text.

The input is opened read-only in binary mode and capped at 64 MiB. The reader
rejects malformed UTF-8 or JSON, duplicate JSON object keys, `NaN` and infinity
constants, and non-list roots with sanitized errors. It adds no runtime
dependencies and does not modify the source file or create sidecars.

This is a conservative adapter for the documented synthetic fixture contract.
It has not been validated against an actual ChatGPT export, and export shapes
can differ by product version or export type. Unsupported variants are not
inferred from timestamps or converted into approximate evidence.
