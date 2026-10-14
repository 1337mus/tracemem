# Context Kit inventory

Updated: 2026-10-13. Read-only discovery for the local TraceMem prototype.

Machine-local checkout: `~/Projects/ctxkit`, inspected at commit
`20096bcb1095883262c77a37bbbb458732d9cf3a`. Its working tree was clean before
and after this work. The legacy executable is `~/.local/bin/ctx`; its live
Markdown corpus is `~/.llm-context`. These locations identify infrastructure;
individual transcript paths remain in ignored, permission-restricted receipts.

## Findings

[ctxkit](https://github.com/1337mus/ctxkit) is a Python 3.11+ MIT-licensed CLI with no runtime dependencies. It keeps the knowledge base as Markdown files in Git and writes measurements and hook bookkeeping as small JSON/TSV files. It has no database or embedding store. The repository version is separate from the older machine-installed `ctx` script.

The inspected ctxkit checkout had no selected config file or environment override. Its built-in defaults enable Claude Code JSONL and Codex rollout JSONL discovery, with the default corpus root absent. The actual live legacy `ctx` points to a different corpus. Metadata-only inspection found 551 Markdown files and 9,771,953 bytes in that live corpus, with file modification dates from 2026-07-05 through 2026-10-14 UTC. These are counts and timestamps only; no corpus text was read for this inventory.

The initial transcript census found 185 Claude files totaling 495,565,759 bytes and 1,409 Codex files totaling 42,394,554,124 bytes. File modification dates were 2026-09-14 to 2026-10-14 UTC for Claude and 2026-02-21 to 2026-10-14 UTC for Codex. The full-file SHA-256 census found zero byte-identical copies. A later streaming parse saw 1,412 Codex files, three more than the census, consistent with active sessions creating files during inspection. The data is live and does not form one atomic snapshot.

The full streaming JSONL pass parsed 112,117 Claude lines and 4,106,321 Codex lines. It found zero malformed JSON lines and zero read errors. A bounded metadata sample selected up to 12 files per provider, one representative from each immediate source directory, evenly spaced across sorted directories. All 24 sampled files had matching hashes before and after parsing. The sample contained no known offset fields. Timestamp coverage was 1,579 of 1,928 Claude records and 4,923 of 4,923 Codex records. Sample event timestamps ranged from 2026-08-29 to 2026-10-11 UTC for Claude, and 2026-02-21 to 2026-10-14 UTC for Codex. These sample results are not extrapolated to the full corpus. For this metadata sample, no transcript text, titles, IDs, or source paths were printed or retained.

A separate privacy-indicator pass used a fresh deterministic 24-file cohort from `ctxkit.transcripts.transcript_files(cfg)` and its configured nonrecursive globs. It selected up to 12 files per provider, one lexical-first file per immediate parent directory, with directories evenly spaced across sorted parents. TraceMem parsed 519 Claude and 682 Codex text blocks. One live source file changed between the first scan and the corrected scanner pass; all 24 files had matching SHA-256 hashes before and after the final scan. Claude blocks had 274 email and 16 AWS access-key indicators across 13 and 6 blocks, respectively. Codex blocks had 22 email and 1 AWS access-key indicators across 14 and 1 blocks. No GitHub token, OpenAI token, Bearer token, or PEM private-key indicators matched. There were 19 Claude and 14 Codex blocks with any indicator; category block counts can overlap. These are heuristic hits, not confirmed secrets, and they are not extrapolated to either corpus. Code and tool output can produce false positives. Health, financial, relationship, and proprietary sensitivity are not classified. The exact sample paths and hashes are retained only in the private `.local/context-kit-inventory.json`; the report contains aggregate counts only.

## Architecture and interfaces

- `src/ctxkit/config.py` resolves TOML config, expands paths, and exposes the corpus root, state directory, and enabled source globs. Lookup order is explicit flag, `CTXKIT_CONFIG`, default config file, then built-in defaults.
- `src/ctxkit/transcripts.py` discovers files with `transcript_files(cfg)` and emits `Event(kind, value)` from Claude and Codex JSONL. `Session` stores tool, file path, file mtime, assistant-turn count, and sets of corpus reads and writes.
- Claude parsing recognizes assistant lines, `Read`, `Write`/`Edit`/`MultiEdit`/`NotebookEdit`, and `Bash` tool calls. Codex parsing recognizes assistant `response_item` messages, calls whose text is in `input` or `arguments`, and successful `patch_apply_end` file changes.
- `src/ctxkit/stats.py` uses those events for pull compliance, popular reads, dead weight, and index routing. `hook.py` uses them for edit and milestone thresholds. `doctor.py` audits Markdown corpus structure and Git diffs. `recall.py` makes derived question sets and score files. `digest.py` writes a derived health report.
- Supported CLI commands are `init`, `doctor`, `stats`, `hook stop`, `snapshot`, `digest`, `distill`, `schedule print`, `recall build|run|report`, and `config`.
- The repository has 14 `unittest` modules. The documented test command from its root is `python -m unittest discover -s tests`. Tests were inspected but not run.

No ctxkit CLI command was run. Some apparently observational commands write state: `doctor` advances fat-file history; canonical-window `stats` records `compliance.tsv`; `digest` writes reports; recall build/run write artifacts; `hook stop` writes payload-shape and nudge markers. `snapshot` stages and commits the corpus. `init` creates files and Git state. `schedule print` only renders output. `config` is read-only but can print local path values.

## Source evidence limits and TraceMem reuse

Context Kit is a usage-measurement adapter, not an evidence store. Its parser emits no message body, role, native message ID, timestamp, or source byte/line offset in its `Event` or `Session` objects. It counts assistant turns and infers reads/writes. Codex reads are command-text heuristics; shell redirects are not fully represented. Malformed transcript lines are skipped without an error record. The session ID in a hook payload is used for a nudge marker, not preserved in stats sessions.

The live raw files do contain structured source metadata. In the bounded sample, Claude lines had session identifiers on 1,926 of 1,928 records and record identifiers on 1,441; Codex records had session identifiers on 2,034 of 4,923 and record identifiers on 3,375. These counts look across the top-level object and common nested message/payload objects. Neither sample exposed known byte/codepoint offset fields. Roles observed were user and assistant in Claude, and user, assistant, and developer in Codex. The adapter ignores user-message bodies and most non-assistant conversation records for its usage metrics.

Reuse `ctxkit.config.load_config` and `ctxkit.transcripts.transcript_files` only for optional configuration and file discovery. Parse raw source records with TraceMem adapters. Preserve each supported message or tool result with its native identifiers, role, timestamp, source line/locator, and text spans; report malformed and unsupported shapes in adapter stats. Keep Context Kit notes and reports as explicitly derived material with provenance. Do not treat those summaries or usage events as exact conversation evidence.

The installed legacy `ctx` is a separate 1,432-line Python script. It has its own hardwired corpus and state paths, plus a read-only SQLite query for a Cursor aggregate. The inspected ctxkit repository includes only Claude and Codex transcript sources; no ChatGPT transcript adapter is present there.

Legacy state category counts observed without reading file contents: compliance 1, digest artifacts 10, fat history 1, hook payload shapes 2, nudge markers 53, other files 6.

## TraceMem adapter validation

A later size-only census, separate from the first-pass counts above, found:

| Provider | Files | Total bytes | Mean bytes/file | Largest file bytes |
| --- | ---: | ---: | ---: | ---: |
| Claude | 186 | 498,022,049 | 2,677,537.9 | 133,778,106 |
| Codex | 1,412 | 42,406,464,573 | 30,032,906.9 | 6,926,829,381 |

The final TraceMem parser was checked against the same 24-file privacy cohort.
This is a newer observation than the privacy-indicator counts, which used an
earlier parser. No records were copied into the persistent TraceMem database.

| Provider | Text blocks | Role counts | Missing timestamps | Mean/max characters |
| --- | ---: | --- | ---: | --- |
| Claude | 519 | user 71; assistant 135; tool 313 | 0 | 2,556.3 / 48,491 |
| Codex | 653 | developer 72; user 61; assistant 189; tool 331 | 0 | 2,806.7 / 48,156 |

All 24 files were hash-stable during this final read. One Codex file differed
from the older privacy snapshot. Unsupported counters were 1,363 Claude and
4,822 Codex events/blocks; these include deliberately skipped usage/control
events, duplicate Codex `event_msg` text, attachments and structured tool output.
The Claude malformed-shape counter was 48, all explained by system/control
envelopes without a `message` object. Codex's counter was zero. These counters
are not counts of malformed JSON, which the earlier full JSON pass measured
separately. Distinguishing control events from malformed message envelopes is
a future adapter-metrics refinement. Counts are sample-only, with no claim of
complete ingestion coverage.

The adapter keeps string tool outputs verbatim and skips structured output
rather than manufacturing evidence by JSON rendering. Conflicting Claude role
fields become `unknown` with `mixed_untrusted` content. Block identifiers retain
original positions even when an earlier unsupported block becomes readable.
Private sample manifests, hashes and the final adapter hash are in
`.local/context-kit-inventory.json` and `.local/adapter-validation.json`.

## Scope limits

This is an inventory, not a corpus-quality or retrieval-quality evaluation. Duplicate detection covers byte-identical whole transcript files only; semantic or record-ID duplicate rates were not measured. Missing-timestamp counts are sample-only and include non-message events. Offset checks cover common explicit offset field names in the sample; TraceMem must keep its own line and text-span positions. Active source files may change after this observation.
