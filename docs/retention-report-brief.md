# Persisted-store retention report

Continuation of the local prototype, after signed95f456e. Root owns architecture,
review and delivery. Reuse the existing pure planner; do not change it, existing
stores/schemas/fixtures/CLI modules or historical receipts. No private inputs,
inference, network, GPU or erasure. Worker owns only
`src/tracemem/retention_report.py`, `tests/test_retention_report.py`,
`docs/retention-report.md`, task logs under `.local/`.

API: `inspect_source_retention(evidence_path, ledger_path, *, source, session_id)
-> dict[str, Any]`. CLI: `python -m tracemem.retention_report --evidence-db ...
--ledger-db ... --source ... --session-id ... --output NEW_PATH`.

Read both files exclusively through OS read-only regular-file descriptors
(`O_RDONLY|O_NONBLOCK`, `O_NOFOLLOW` where available, `fstat`, bounded64MiB+1read).
Refuse symlink/nonregular/missing inputs, same file/inode for both stores,
oversized input, any existing `-wal`, `-shm`, `-journal` sidecar (including dangling
symlinks), and WAL-format main files even without sidecars. Validate SQLite header
read/write format bytes18/19 are rollback format1. Do not open input paths with
sqlite3 or writable store constructors, create source parents, chmod sources or
recover/checkpoint a journal. Use `sqlite3.connect(':memory:')`, deserialize copied
bytes, `PRAGMA trusted_schema=OFF`, `query_only=ON` and integrity_check. Work is
bounded offline analysis, not live/WAL support.

Validate selector before reading. Reuse EvidenceStore._validate_existing,
CandidateLedger._validate (static, memory connections only) and ledger's canonical
_entry_from_json. Load ALL historical evidence; recompute each stored record ID
and revision from its fields and require exact match. Cross-check candidate_lineage
SQL edges against payload supersedes exactly. Every provided canonical citation
must identify a supplied evidence revision and have nonempty in-bounds offsets;
reject dangling citations. Rejected entries with no citations remain analyzable.
Then call plan_source_erasure, which validates the graph and candidates. This
checks snapshot integrity, not semantic truth or complete source ownership.

Read/check both source files again after analysis before returning: hashes and
descriptor identity/size/mtime_ns/ctime_ns must match initial snapshots; check
sidecars on each read. Detect ordinary file replacement/content changes. Document
that inputs must be quiescent rollback-mode databases: two copied files are not
a guaranteed atomic cross-store transaction and adversarial/ABA concurrent changes
are outside this diagnostic. No lock/atomicity guarantee or full live-store claim.

Return schema_version, source snapshots' SHA256+byte counts (no paths), evidence
revision/candidate counts, asdict(impact), and static snapshot/limitations fields.
Omit source/session selectors, source locators, raw text, claims/quotes, verifier
errors and paths. Hashes/IDs remain private. Sanitize API/CLI failures without
echoing database content, paths or selector values.

CLI serializes before output creation; exclusively create new0600report, refusing
existing files/symlinks. Reject output destinations equal to EITHER input path or
its three sidecars, resolving parent aliases. No output on invalid inputs. Do not
claim atomicity on disk-write failure; a partial newly created report may remain.
No erasure, filesystem cleanup, schema migration, external-copy audit or report
upload. Keep actual runtime implementation reasonably bounded (~350-450lines).

Use incremental meaningful TDD (stub import succeeds, first RED tests behavior).
Tests use new fictional stores and realSQLite. Cover historical Unicode citations,
direct/ambiguous/descendant plan equality, rejected candidate handling, source bytes
AND mode/mtime preservation, schema swaps/alien/corrupt inputs, stored identity or
payload hash/lineage mismatch, missing citation evidence, no writable constructors
or input-path sqlite calls, missing/symlink/FIFO/sparse oversized/same-inode input,
sidecars/WAL-format refusal, detected changes during analysis, input selector
validation before reads, private-content-free report/errors, and actual CLI exclusive
0600output/input preservation/reserved-sidecar protection. Use focused tests,
Ruff/check+format and strict mypy on changed files. Root handles full suite,
installedwheel,CI/docs integration and signed private delivery. No stage/commit/push.
