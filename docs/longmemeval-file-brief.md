# Bounded LongMemEval file scan and selection

Root follow-up after f949b9a. The current whole-bytes adapter/CLI caps the entire
input at 64 MiB; public metadata lists S=277383467 bytes and M=2737100077 bytes.
Rights remain unresolved; do not fetch benchmark bodies. Implement only a bounded
file reader/selector with fictional tests, reusing the existing schema/identity
adapter. No inference, retrieval-policy change, private input, network or GPU.

Worker owns only `src/tracemem/adapters/longmemeval_file.py`,
`tests/test_longmemeval_file.py`, `docs/longmemeval-file.md`, and task logs under
`.local/longmemeval-file-*`. Existing adapter/experiment/config/fixtures stay
unchanged. No commit/stage/push. Root reviews source/tests before packaging.

API `scan_longmemeval_file(path, *, namespace, variant, question_ids=())` returns
frozen slotted `LongMemEvalFileScan` with `input_sha256`, `input_bytes`,
`question_count`, and `selected_questions: tuple[LongMemEvalQuestion, ...]`.
`question_ids` must be a tuple of at most five distinct nonblank scalar-Unicode
strings, each at most 4096 UTF8 bytes. Validate namespace/variant using the existing empty-array bytes adapter
before opening a path. Validate selection before I/O. Return selected questions
in source-file order; all requested IDs must exist. Empty selection scans and
validates the whole file but retains no conversations. No CLI needed this slice.

Read one regular non-symlink input with OS O_RDONLY|O_NONBLOCK/O_NOFOLLOW when
available, lstat/fstat identity checks and bounded reads. Reject FIFO/directory,
missing/symlink input and files over 4 GiB without hanging/reading their bodies.
Process a strict top-level JSON array of objects incrementally, with a raw item
cap of 64 MiB minus the two wrapper bytes, at most 10000 questions, and a bounded
nesting cap (e.g.128). Use 64 KiB or 1 MiB read chunks and never unbounded read().
Compute the exact raw file SHA256 and byte count including all whitespace.
No persistent spool or output files, source writes/chmod/parent creation.

The byte scanner only frames object boundaries: honor escaped quotes/backslashes,
string contents and nested arrays/objects; reject malformed separators, nonobject
array members, trailing commas, incomplete/trailing content, JSON-invalid whitespace,
invalid UTF8/BOM and malformed content. Wrap each completed item in b'['+item+b']'
and call existing `parse_longmemeval_bytes` for strict JSON, duplicate keys,
constants, Unicode, schema and evidence identities. Track duplicate question IDs
across the entire file. Enforce 4096 UTF8 bytes per question ID and a cumulative
1 MiB UTF8 budget for the seen-ID set; count bounds alone could retain gigabytes
of metadata. These are file-reader resource bounds, distinct from the unchanged
adapter schema. Do not reproduce semantic/schema validation. Preserve the
exact question/record IDs and metadata produced by the existing bytes adapter.
Drop each unselected parsed question before proceeding; retain only requested
questions and the bounded ID set. Enforce total/item/question/depth bounds during
reads, not only from initial stat. Use efficient chunk whitespace skipping so a
large whitespace-only suffix is not a per-byte Python-loop bottleneck.

Return only after complete input validation, EOF, requested-ID checks and source
stability checks. No generator or externally visible selected prefix before a
late invalid item. Compare final descriptor identity/size/mtime_ns/ctime_ns and
resolved path identity to initial state; refuse ordinary concurrent changes.
Document quiescent input requirement and no atomic/adversarial ABA guarantee.
All failures sanitize to static ValueError text without paths/content/selectors.
Selected objects contain conversation text and must remain private. Boundaries
limit raw buffers/selection, not a certified exact peak-RAM figure. Support is
fictional-format verified only; it does not establish upstream compatibility,
rights, QA accuracy, or a full large-dataset evaluation.

Incremental meaningful TDD. Test equality to whole-bytes parser for all fixture
questions and order-independent selection requests; exact file hash including
whitespace/Unicode; empty/no-selection and missing requested IDs; cross-item ID
duplicates and malformed late items with no partial return; nested/escaped string
boundary cases and varied read boundaries; strict JSON/UTF8/schema rejection;
input kinds, sparse oversized file and no writes/mode/mtime changes; during-scan
mutation; per-item/total/question/depth guards with patched small constants; a
real fictional file >64 MiB using a large whitespace suffix without holding it
all in memory. Keep tests meaningful and fast; no model dependencies. Focused
suite, Ruff check/format and strict mypy on new module. Return released files,
exact hashes and RED/GREEN evidence. Root owns full-suite/installedwheel delivery.
