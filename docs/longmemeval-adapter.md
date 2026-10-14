# LongMemEval format adapter

`tracemem.adapters.longmemeval.parse_longmemeval_bytes` parses the documented
LongMemEval question shape used by the authored fixture and has a narrow format
check against the pinned oracle object at revision
`98d7416c24c778c2fee6e6f3006e7a073259d48f` (15,388,478 bytes; SHA-256
`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`). The
check covers byte-parser and scanner shape only. See the
[compatibility probe](longmemeval-compatibility.md) for measured scope and
limits. The authored, MIT-licensed fixture at
`examples/synthetic/longmemeval-format.json` contains only fictional content.
This adapter does not read files, download data, or provide a CLI.

The fixture deliberately places an unrelated expected-answer sentinel and evidence
flag beside its source text to test metadata separation. Those fields are protocol
test inputs, not gold QA targets or semantic support annotations.

The parser accepts UTF-8 JSON bytes up to 64 MiB and requires an explicit stable
`namespace` plus a `variant` of `s`, `m`, or `oracle`. It validates the pinned
nine-field question schema, documented question labels, exact turn fields,
parallel history arrays, and question/session identifiers. Duplicate JSON keys,
non-finite constants, malformed UTF-8, excessive nesting, oversized integers,
lone surrogates, and schema mismatches fail with `ValueError`; parsing is
all-or-nothing. The expected answer accepts strings and exact JSON integers,
normalizing integers to canonical decimal strings while preserving strings
exactly. Booleans, floats, nulls, arrays, and objects remain invalid answers.

The complete S release file is 277,383,467 bytes and the complete M release
file is 2,737,100,077 bytes, so this bytes API's 64 MiB cap prevents parsing
either complete release. The 15,388,478-byte pinned oracle object has passed a
narrow parser and scanner format check. The streaming file scanner can process
larger local files within its declared limits, but real S and M releases remain
untested; see the [compatibility probe](longmemeval-compatibility.md).

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

The synthetic fixture and narrow oracle-file check do not establish general
LongMemEval compatibility. Dataset-specific rights and compatibility with the
upstream evaluator remain unverified; see [the historical metadata rights
note](longmemeval-rights.md). No retrieval score, QA accuracy, metric
equivalence, or benchmark-gate claim follows from this adapter.

The separate [bounded file scanner](longmemeval-file.md) validates larger fictional
files incrementally and retains at most five requested questions. It reuses this
adapter for each item. The whole-bytes API and default retrieval CLI retain their
64 MiB limit. Explicit CLI question selectors use the file scanner; see
[selected-file retrieval](longmemeval-selection.md). File scanning does not establish
full upstream compatibility, data rights or large-dataset retrieval/QA accuracy.
