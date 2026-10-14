# Fresh-store ChatGPT snapshot import

2026-10-14. Local Phase 1 follow-on. Root architecture; bounded implementation to
cheaper worker. No private exports, models, network, training or publication.

Problem: the explicit ChatGPT reader filters one selected branch, but ingesting a
changed export into an existing append-only store leaves prior sibling heads.
Provide a usable CLI/API that always creates a new derived store, without sync,
erasure, migrations or changes to EvidenceStore and the existing ingest command.

Own only new src/tracemem/snapshots.py, tests/test_chatgpt_snapshot.py,
docs/chatgpt-snapshot.md, and narrow CLI additions in src/tracemem/cli.py. Wait
for root's stable-adapter notice before validation. Do not edit adapter/schema/
existing tests/README. Production ceiling ~250 added LOC; propose split if more.

API import_chatgpt_snapshot(source, destination) -> dict[str, object], returning
aggregate inserted_revisions, malformed_records, unsupported_records, role_counts
and snapshot_policy='fresh_store_selected_branch'. Resolve explicit source as a
regular existing file read-only; destination parent must already exist. Reject
source-directory subtree, existing destination of any kind (including dangling
symlink), and existing destination SQLite sidecars (-journal,-wal,-shm). Preserve
all such paths byte-for-byte. No auto-discovery or source copying.

Build a new EvidenceStore under a TemporaryDirectory in destination's parent
(mode0700), stream read_chatgpt_export into it, then close SQLite. Any malformed
conversation count rejects the entire import; unsupported content is allowed
and reported. Empty valid export may create an empty snapshot. Publish only the
closed new database with exclusive os.link(temp_db, destination); never use
replace/rename-overwrite/unlink on destination. Recheck destination sidecars before
publication. The temporary context cleans only its own new directory. Existing
output must never be opened. File mode0600 inherited from EvidenceStore. On
failure no destination is created; sanitize exception text/path/content and
suppress nested exception chaining. No guarantee against hostile concurrent
parent-directory mutations or crash durability beyond SQLite/local filesystem.

CLI add import-chatgpt --input PATH --db PATH, with aggregate JSON output. Existing
search/inventory work on the resulting DB. Generic CLI failure status2 stays as
is. Do not extend --provider on existing ingest, which permits reimports.

TDD one behavior at a time; meaningful integration tests with fictional export:
branch import -> real reopened BM25 search -> exact Unicode citation; switching
current_node into a second fresh store includes only newly selected branch and
leaves first snapshot byte/hash/mtime unchanged; existing DB/arbitrary output/
symlink rejected unchanged; source path and source-subdirectory rejected with no
sidecars/source changes; malformed selected conversation or duplicate IDs/JSON
failure leaves no destination/temp debris; existing destination sidecars preserved;
0600 output and aggregate CLI success/failure without transcript output. Avoid
mock-heavy implementation tests. A narrow publication collision test may patch
os.link to create an output first, then call real os.link and prove no overwrite.

Document explicit-source CLI use, source/output separation, snapshot freshness,
unsupported/invalid behavior, permissions, fixture-only compatibility, no actual
export validation, and old snapshot retention/privacy. No deletion recommendation
or lifecycle claim. Report RED/GREEN and stable files to root, which reviews all
diffs before the integrated suite/wheel. Run focused tests, Ruff, source mypy.
