# Transcript line size limit

`read_transcript` reads JSONL transcripts one physical line at a time, with a bounded
binary read. Its default `max_line_bytes` is 8 MiB; callers may select a limit from 1
byte through the public 64 MiB ceiling. The `tracemem ingest` command exposes the same
setting as `--max-line-bytes`.

The limit counts every byte in a physical line, including LF or CRLF terminators. A
line exactly at the limit is accepted, including a final line without a newline. If a
read returns one byte beyond the limit, ingestion raises `transcript line exceeds the
byte limit` immediately. The reader does not parse, drain, skip, or truncate that line.
Older transcripts containing a larger individual line must be ingested with an
explicitly larger limit, up to 64 MiB.

For example, to allow lines up to 16 MiB:

```sh
tracemem ingest --provider claude --input transcript.jsonl --db evidence.sqlite --max-line-bytes 16777216
```

Line boundaries are LF or CRLF. CR-only separators between JSON objects are not
supported by this bounded JSONL reader; the previous text-mode iteration treated CR
as a separator.

Invalid limit types raise `TypeError`; values outside the supported range raise
`ValueError`. Invalid UTF-8 remains an error. CLI errors are sanitized and do not print
transcript contents. The source file is never modified. A newly created database may
contain an empty schema after a refused ingest; existing ingests roll back their
records and current-head updates when a later line is rejected.

This contract bounds raw line reads only. Parsed JSON objects and yielded evidence
records also allocate memory, and binary reads do not provide a consistent snapshot if
the source changes during ingestion.
