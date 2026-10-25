# TraceMem

*Find the passage. Keep the source.*

TraceMem helps you find what you said and decided in past conversations. Each result includes the original message, who said it, and when.

[Current architecture](#current-architecture) · [Highlights](#highlights) · [Benchmark snapshot](#benchmark-snapshot) · [Quick start](#quick-start) · [The goal](#the-goal) · [Documentation](#documentation)

## Current architecture

Today, TraceMem is a local search tool. It collects conversation records in SQLite, finds passages by words, vectors, or both, and returns each passage with a citation to its source.

![Current architecture: conversation sources flow into SQLite, search, and cited passages fitted to a chosen size limit.](docs/assets/architecture.svg)

## Highlights

- Read-only imports from Claude Code and Codex, plus explicit ChatGPT JSON exports.
- SQLite keeps messages, dates, and their history.
- Search by keywords, vectors, or both.
- Whole passages with citations fit a size limit you choose.

A fictional demo asks, “Where is the quarterly planning meeting?” TraceMem returns, “The quarterly planning session is in room Juniper, on the fourth floor,” with a citation to the source passage.

## Benchmark snapshot

The October 23, 2026 claim-checking test uses selected public benchmark evidence: 120 claims across 30 conversation families and 40 sessions. Labels were agent-reviewed under the owner's policy: remember durable facts and preferences.

| Measure | Result |
| --- | ---: |
| Source support among 61 accepted claims | 58/61 (95.1%) |
| Useful claims retained | 28/28 |

A separate training set contains 256 fictional claims. See the [benchmark snapshot](docs/benchmark-snapshot.md) for the measurement details.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once. It sets up Python 3.12 for you.

```sh
git clone https://github.com/1337mus/tracemem.git
cd tracemem
uv run --python 3.12 tracemem demo
uv run --python 3.12 tracemem benchmark
```

The demo uses fictional messages and Python's built-in SQLite. No API key or model download is needed.

For local transcript ingestion and search, see [the source adapter and ingestion notes](docs/README.md#source-adapters-and-ingestion) and [data governance](docs/data-governance.md).

## The goal

The design goal is to suggest facts and preferences worth keeping from conversation history. Each fact is checked against its source, saved with its history, and used in answers that point back to the original passages.

![The goal: conversations become proposed facts, checked against their source and whether they are worth keeping, preserved as versioned facts, and used for source-backed answers.](docs/assets/goal.svg)

## Documentation

- [Documentation index](docs/README.md) for implementation notes, experiments, and historical reports.
- [Data governance](docs/data-governance.md) for local source handling.
- [MIT License](LICENSE).
