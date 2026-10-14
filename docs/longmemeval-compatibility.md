# LongMemEval format compatibility probe

This is a local-only format probe against `xiaowu0162/longmemeval-cleaned` at
revision [`98d7416c24c778c2fee6e6f3006e7a073259d48f`](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/tree/98d7416c24c778c2fee6e6f3006e7a073259d48f).
The pinned oracle object is 15,388,478 bytes with SHA-256
`821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c`.
Source benchmark file stayed read-only throughout probing and replay.

The initial probe found 500 questions, 948 sessions, and 10,960 turns. It parsed
the 468 questions with string-valued answers and rejected 32 with integer-valued
answers; no other schema failures were observed. The first five questions in
source order were frozen before inspection; each had a string answer and passed
individual parsing.

Before the integer-answer handling change, both whole-file and selected-question
scans rejected the object because they validate every question before returning
a full scan or selected subset. A subsequent root replay against the same
read-only file passed the narrow format check: all 500 questions parsed, all 468
string answers stayed unchanged, and the 32 integer answers became canonical
decimal strings. All 500 question/evidence tuples matched a transient
string-normalized clone. The whole-file scanner reported 500 questions while
retaining none; the first-five selected scan matched the bytes parser.

The input hash, size, mode `0600`, and modification time were unchanged before
and after replay. The combined JSON clone and two scanner passes took 3.160915
seconds and observed process peak RSS was 226,820,096 bytes. This includes the
clone and both passes; it is not standalone parser latency, a memory guarantee,
or a scale estimate. Installed focused validation passed 69 tests and the
fictional evaluator/import check; the adapter imported from site-packages with
reviewed source SHA-256
`dc292e7f3eeb0eeb4ee253f7227c92c3b36eb8ea2e85ff2e14a75dd2d2b8551e`.
The exact run and checks are recorded in
[`validation-longmemeval-integer-answer.json`](validation-longmemeval-integer-answer.json).

The streaming scanner exists, and its synthetic tests cover inputs within its
declared limits. Those tests do not establish real upstream S or M release
compatibility or performance; neither real release has been tested here.

This replay establishes only format parsing and scanner behavior for this one
pinned oracle object. It does not establish retrieval compatibility, QA behavior,
evaluator or score equivalence, data-use rights, or Gate A. The rights
discussion remains in the [historical metadata audit](longmemeval-rights.md);
this replay makes no new rights conclusion. No benchmark content or question
identifiers are included in this report.
