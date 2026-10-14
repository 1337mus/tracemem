# Optional quiescent transcript reads

Transcript reads remain streaming by default. Callers that want an ordinary
concurrent-change check can opt in with `read_transcript(...,
require_quiescent=True)` or `tracemem ingest --require-quiescent`.

Strict mode requires the leaf input path to name a regular, non-symlink file.
It opens the file without blocking on special files, records file identity and
metadata from both the path and open descriptor, and compares them again after
the iterator reaches natural EOF. If opening or either comparison detects a
change, the read fails with `transcript source is not a quiescent regular file`.
The CLI reports its existing sanitized ingest failure and does not print source
contents. When used with `EvidenceStore.ingest`, a late refusal rolls back the
records and head changes from that ingest transaction.

The final check only runs after the caller consumes the iterator to EOF. Closing
an iterator early releases its descriptor but does not claim that the unconsumed
remainder was checked. Existing line-size, UTF-8, and parsing behavior is shared
with default mode.

This is a best-effort guard against ordinary appends, truncations, in-place
edits, and path replacement during a complete read. It does not create an
atomic filesystem snapshot and cannot rule out edits whose metadata is restored
before the final check. Default mode keeps its previous streaming behavior and
continues to follow a leaf symlink.
