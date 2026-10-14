# ChatGPT export snapshots

TraceMem can import one explicitly selected ChatGPT JSON export into a new
SQLite evidence database. Each import reads the export's `current_node` parent
chain and creates a fresh store, so changing the selected branch or importing a
revised export does not carry records from an earlier import into the new
database.

## CLI

Keep the export and derived database in separate directories. The destination
directory must already exist, and the output path must be unused:

```sh
tracemem import-chatgpt \
  --input /path/to/source/conversations.json \
  --db /path/to/derived/chatgpt-snapshot.sqlite3
```

The input is an explicit regular file opened read-only. The importer does not
search for exports or copy the source. It rejects an output in the source
directory tree, any existing output (including a symlink), and any existing
SQLite journal, WAL, or shared-memory sidecar for that output. It does not
create a missing output directory.

Successful imports print aggregate JSON counts for inserted revisions,
malformed conversations, unsupported content, and roles. Transcript text is
not printed. The result also identifies the policy as
`fresh_store_selected_branch`. Failures use the CLI's generic error and exit
status 2; source paths and export content are omitted from the error message.

The same operation is available as `tracemem.snapshots.import_chatgpt_snapshot`
for Python callers; it returns the same aggregate object and applies the same
path and publication rules.

## Snapshot behavior

The importer reads the export through the read-only ChatGPT adapter and writes
to a temporary database in the destination directory. It closes the database
before publishing it with exclusive hard-link creation. Publication requires
filesystem support for hard links. The output is mode
`0600`; the temporary directory is mode `0700`. A destination collision leaves
the existing path untouched.

Malformed JSON or any malformed conversation rejects the whole import and
publishes no database. Unsupported content, such as non-text parts, is skipped
and counted. A valid empty export is allowed and produces an empty database.
The adapter contract, including its selected-branch traversal and text handling,
is described in [chatgpt-export.md](chatgpt-export.md).

Each destination is independent. Earlier snapshot files remain as they are;
the importer does not update, erase, or manage other databases. Plan access and
retention for derived files according to your own data-handling requirements.

## Compatibility and limits

The adapter supports the documented synthetic fixture contract only. It has
not been validated against a real ChatGPT export, whose shape may vary by
product version or export type. Review fixture coverage and compatibility
before relying on an import. The importer performs no network access, model
loading, training, or publication.
