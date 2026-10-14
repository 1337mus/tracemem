# LongMemEval format adapter

`tracemem.adapters.longmemeval.parse_longmemeval_bytes` parses a conservative,
fixture-only subset of the documented LongMemEval JSON shape. The authored,
MIT-licensed fixture at `examples/synthetic/longmemeval-format.json` contains
only fictional content. This adapter does not read files, download data, or
provide a CLI.

The fixture deliberately places an unrelated expected-answer sentinel and evidence
flag beside its source text to test metadata separation. Those fields are protocol
test inputs, not gold QA targets or semantic support annotations.

The parser accepts UTF-8 JSON bytes up to 64 MiB and requires an explicit stable
`namespace` plus a `variant` of `s`, `m`, or `oracle`. It validates the pinned
nine-field question schema, documented question labels, exact turn fields,
parallel history arrays, and question/session identifiers. Duplicate JSON keys,
non-finite constants, malformed UTF-8, excessive nesting, oversized integers,
lone surrogates, and schema mismatches fail with `ValueError`; parsing is
all-or-nothing.

The complete S release file is 277,383,467 bytes and the complete M release
file is 2,737,100,077 bytes, so this adapter does not support parsing either
complete release. The oracle file is 15,388,478 bytes and fits under the cap,
but full oracle compatibility is also unverified. Supporting the larger S and
M files requires separately reviewed streaming or partitioning behavior.

Each result is a frozen `LongMemEvalQuestion`. Its session and turn order follows
the input exactly, including for oracle histories whose input order may not be
chronological. `question`, expected `answer`, dates, labels, and type stay out of
the evidence records. `answer_record_ids` contains only turns whose explicit
`has_answer` value is `true`; the adapter does not infer missing labels from
`answer_session_ids`. `_abs` is represented by the question ID suffix only.

Evidence records use `content_kind="conversation"`, preserve transcript role
and text, and leave `recorded_at` unset. The upstream `haystack_dates` value is
retained on `LongMemEvalSession.date`, because it is a session date rather than a
per-message timestamp. Callers that need temporal context must carry that session
date alongside retrieved evidence.

Evidence identity is scoped by the caller's namespace, variant, question ID,
upstream session ID, and turn index. Unique IDs alone do not isolate retrieval:
create a separate `EvidenceStore` for each question and ingest only that
question's `records`. The retriever has no corpus filter, so combining questions
in one store can contaminate results even when identities are unique.

This fixture does not establish full compatibility with real LongMemEval files.
Dataset-specific rights and compatibility with the upstream evaluator remain
unverified; see [the LongMemEval rights note](longmemeval-rights.md). No
retrieval score, QA accuracy, metric equivalence, or benchmark-gate claim follows
from this adapter.
