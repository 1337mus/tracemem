# Persisted-store retention report

`tracemem.retention_report` runs the existing source-retention impact planner on
two local SQLite snapshots: an evidence store and a candidate ledger. It reads
all historical evidence revisions and ledger entries, validates their stored
identities, citations, and correction lineage, then reports the planner's
impact for one source and session. The report is a local diagnostic. It does
not erase data, modify either store, run inference, or contact another service.

## Run it

Use Python with `sqlite3.Connection.deserialize` support in its linked SQLite
library. The input databases must be quiescent rollback-mode databases; live
WAL databases are unsupported.

```sh
python -m tracemem.retention_report \
  --evidence-db /private/path/evidence.sqlite3 \
  --ledger-db /private/path/ledger.sqlite3 \
  --source fictional-provider \
  --session-id fictional-session \
  --output /private/path/retention-report.json
```

The output parent directory must already exist. The command exclusively creates
a new report file with mode `0600`; it will not replace an existing file or
symlink. It serializes the report before creating the destination. If the write
fails after creation, a partial report may remain. Choose an output destination
that is separate from both databases and their `-wal`, `-shm`, and `-journal`
sidecars. Parent-directory aliases are resolved. For safety across
case-insensitive or Unicode-normalizing filesystems, destination comparisons
also fold case and normalize Unicode to NFC. This can conservatively reject
equivalent names on a case-sensitive filesystem too.

The Python API is `inspect_source_retention(evidence_path, ledger_path, *,
source, session_id)`. It returns the same report dictionary without writing an
output file.

## Snapshot and validation requirements

The command accepts only distinct, regular, non-symlink files no larger than
64 MiB each. It rejects existing SQLite sidecars and WAL-format database files,
including a WAL-format main file when no sidecar is present. Inputs must use
SQLite's rollback-journal header format. The implementation reads input bytes
through read-only file descriptors, deserializes those bytes into in-memory
SQLite connections, and runs integrity and foreign-key checks there. It does
not open the source paths through SQLite, recover or checkpoint a journal,
change permissions, or create source directories.

The two files are copied and analyzed separately; they are not a guaranteed
atomic cross-store transaction. Keep both stores quiescent for the full run.
The command rereads them after analysis and checks hashes and file identity,
size, and timestamps to detect ordinary changes. It provides no locking or
atomic snapshot guarantee. Adversarial concurrent changes, including ABA
changes that restore observable file state, are outside this diagnostic.

Validation covers SQLite integrity, expected schemas, evidence record IDs and
revisions, canonical ledger payloads, SQL lineage edges, and canonical
citations that point to supplied evidence revisions with valid offsets. A
dangling citation or inconsistent lineage fails the report. Rejected ledger
entries with no citations can still be analyzed. These checks establish
snapshot consistency, not semantic truth or complete source ownership.

## Reading the report

The JSON contains `schema_version`, SHA-256 hashes and byte counts for each
snapshot, the evidence revision and candidate counts, the existing planner's
impact result, and explicit limitations. It omits source and session selectors,
database paths, source locators, transcript text, claim text, and verification
errors. Snapshot hashes and record or candidate IDs are still private
identifiers; keep the report in access-controlled storage.

The planner does not inspect or certify FTS or vector-cache contents or counts.
The diagnostic also cannot discover exports, backups, unlinked copies, free
SQLite pages, or in-memory caches. It does not guarantee detection of semantic
dependence or all information flows, and it does not establish physical
deletion.
