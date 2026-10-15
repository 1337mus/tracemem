# QA contrast answer diagnostic results

**Development-only, provisional diagnostic.** The [report](qa-contrast-answer-dev.json) analyzes one Qwen/Qwen3-4B call on six fictional, agent-authored development questions. There are no human labels, the test partition is empty, and Gate A remains unmet. The 46 unique answers are reused across 72 correlated cells; those cells are not 72 independent trials.

## Literal results

Scoring is exact string equality after casefolding and collapsing whitespace. Punctuation, aliases, paraphrases, substrings, and semantic judging are not normalized. Every retrieval-mode / prompt-ceiling group had **0/4 answerable literal matches**, **2/2 literal `UNKNOWN` matches on unknown cases**, and **2/6 all-reference matches**. The `0/4` is literal reference-string agreement; it does not measure semantic correctness and must not be reported as semantic accuracy of zero.

The table shows the count of answerable outputs that equal the literal string `UNKNOWN`, out of four answerable cases in each group:

| Retrieval mode | 128 tokens | 256 tokens | 512 tokens | 1,024 tokens |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 4/4 | 0/4 | 1/4 | 0/4 |
| Vector | 4/4 | 1/4 | 2/4 | 1/4 |
| Hybrid | 4/4 | 0/4 | 2/4 | 1/4 |

Some generated text narrates `UNKNOWN` or paraphrases an unknown response. Such text is not the exact `UNKNOWN` output and is not scored as an abstention. `non_unknown_outputs_on_unknown` is a literal category; it does not establish that an answer is semantically unsupported. Empty output is counted separately.

## Run and limits

The runner sent 46 unique prompts containing 23,533 measured input tokens; the 72-cell mapping sums to 28,944 because 26 cells reuse prompts. The generation cap was 128 tokens per unique prompt (5,888 total); generation produced 479 tokens. All 46 outputs ended at EOS. There were zero empty outputs, length stops, cap hits, or truncated outputs.

One T4 call used Qwen/Qwen3-4B revision `1cfa9a7208912126459214e8b04321603b3df60c`, float16, greedy decoding, and no prompt truncation. The receipt records 78.373s model load, 39.131s GPU event duration, 129.995s remote elapsed time, 234.096s client elapsed time, and 8,822,848,512 peak allocated GPU-memory bytes. These are run measurements, not cost or billing data.

The image pins its direct packages: torch 2.7.1, transformers 4.51.3, accelerate 1.6.0, tokenizers 0.21.1, and jinja2 3.1.6. Other packages resolved during the image build; the saved startup log is evidence of those resolved versions, not a fully hashed transitive lock.

## Artifacts and verification

The runner created the original local [primary receipt](../examples/synthetic/qwen3-4b-qa-contrast-answer-remote.json) and [raw receipt](../examples/synthetic/qwen3-4b-qa-contrast-answer-raw.json) with mode 0600; Git checkout file permissions are not guaranteed to retain that mode. Their decoded result objects are equal, while their wrappers and whole-file hashes differ. The [fictional development fixture](../examples/synthetic/qa-contrast-dev-v1.json) remains the only question set. The [machine-readable report](qa-contrast-answer-dev.json) retains every cell, raw output, reference, token count, stop flag, and separate retrieval-label diagnostic.

An independent audit rebuilt all 72 cells and 12 groups from frozen pre-output expectations and the raw 46 outputs without importing the analyzer. It matched the report and receipts; the final Modal inventory showed zero active GPU jobs. A fresh installed replay also compared source and copied CLI output byte-for-byte against the report, using the existing pinned wheel and blocking Modal, torch, transformers, and tokenizers imports.

The signed runner commit `7f7c08f7cee05c73d377d437b0200020f9dac6f2` passed [CI run 38671571328](https://github.com/1337mus/tracemem/actions/runs/38671571328): source tests 513 passed / 1 skipped; installed-wheel tests 260 passed / 2 skipped. The root full check attempted 520 tests: 519 passed and 1 was skipped in 27.861s; Ruff checked 115 files, and mypy checked 32 source files. The analyzer, report and fictional receipts were signed and privately pushed as
[`743bb2f`](https://github.com/1337mus/tracemem/commit/743bb2fe76b67d652d4d349a5b01f369019e01ef).
[Exact CI 38673059955](https://github.com/1337mus/tracemem/actions/runs/38673059955)
passed: 519 source tests passed / 1 skipped; 266 installed tests passed / 2 skipped,
including actual-report byte equality outside the checkout. The validation JSON
is the preserved pre-push snapshot; this paragraph records later delivery evidence.

These results do not establish semantic QA accuracy, calibrated abstention, test performance, model superiority, training readiness, or cost per correct answer. They do not support training or model selection.
