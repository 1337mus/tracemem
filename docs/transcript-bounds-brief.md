# Bounded transcript line reads

Root architecture,2026-10-14. Phase1 reader streams files but text iteration allocates
an entire JSONL line. Metadata census observed multi-GiB transcript files; this is
not evidence that individual lines are that large. Add a bounded raw-line contract
so a large tool-output envelope cannot cause an unbounded line allocation. No claim
of total-process memory guarantee: parsed objects and yielded records also allocate.
No models, GPU, private corpus reads, downloads or ContextKit changes.

Own only src/tracemem/adapters/__init__.py, src/tracemem/cli.py,
tests/test_transcript_bounds.py, docs/transcript-bounds.md and
.local/transcript-bounds-* logs. No commits/staging/push or other files. Production
change target <=100LOC; partition before500. Root owns independent replay, package
approval,fullsuite,delivery. Read applicable preparation/TDD/engineering discipline.
Use explicit per-behavior callable RED/GREEN; preserve existing record/revision IDs.

API read_transcript gets keyword-only max_line_bytes=DEFAULT_MAX_TRANSCRIPT_LINE_BYTES
(default8*1024*1024). Public ceiling MAX_TRANSCRIPT_LINE_BYTES=64*1024*1024. Validate
strict integer (reject bool),1..ceiling,before opening source. Bad type TypeError;
bad range ValueError; stable sanitized messages. Binary file open and
readline(max_line_bytes+1) bound each raw physical line; any result >limit raises
ValueError('transcript line exceeds the byte limit') immediately, without JSON
parsing, draining or emitting that line. Limit includes raw LF/CRLF terminator.
Exactlylimit accepted; no-final-newline accepted. UTF8 decoding remains strict and
sanitized. JSON syntax/nonobject stats and unsupported shapes remain as before.
Preserve source locator calculated from original path, physical line number,
provider session state, native IDs, role, source text and stats for all allowed input.
Do not change other parsers, stats fields, schema or default evidence behavior.

CLI ingest --max-line-bytes int default same constant; pass through to reader. Existing
EvidenceStore.ingest transaction rolls back if generator raises; verify real late
oversized line does not leave prefix records or changed existing heads. Existing
CLI catches error and prints sanitized text. New DB may exist with empty schema
on refusal; docs state that; don't promise originalDB byte identity (SQLite control
bytes may differ) or source-file atomic snapshot consistency. No truncation/skip or
silent override. Older explicit large lines now refused; raise cap explicitly up
to64MiB. Source files unchanged. Binary reads do not imply a complete-file snapshot.

Focused tests: validClaude+Codex payloads includingUnicode,CRLF,EOF and unsupported/
malformed lines preserve exact EvidenceRecords and stats; exactlylimit/limit+1;
rejectinvalidUTF8; rejectinvalidlimit before open; bounded-read instrument proves no
unbounded read/iteration even with huge unterminated input (use real file too);
late failure rolls back current-head changes/new rows using real EvidenceStore;
CLI succeeds with explicit larger cap and fails sanitized at smaller cap with input
bytes/mode/mtime preserved. Avoid tautology/mock-only tests; one narrow stream guard
may assert read request sizes while real parsing executes. Run focused suite plus
existing adapter tests, Ruff check/format, strict source mypy. Release exact hashes
and RED/GREEN evidence. No fullsuite; root final boundary once. Finish with deliverable
or blocker, no self-waiting.
