# Extraction failure audit

The accountable agent read all 24 Qwen3-4B outputs against their fictional source
messages on 2026-10-13. This is a qualitative engineering audit, not an independent
human annotation study or a semantic-support precision estimate. The unchanged
[raw receipt](../examples/synthetic/qwen3-4b-remote.json) and
[computed report](baseline-qwen3-4b-extraction.json) are the reproducible evidence.

All 24 responses decode as whole JSON. Four responses fail the strict candidate
schema: cases 010, 011, 019 and 022. The other responses contain 24 parsed claims;
all fail evidence integrity. There are 17 quote-mismatch errors, seven out-of-bounds
errors and one attribution-role conflict. Errors can overlap. No output reached
the 512-token limit. Two of four no-memory cases abstained correctly.

Offset failure dominates the integrity metric, but it conceals other mistakes:

| Case | Observed failure in the raw output |
| --- | --- |
| 001 | Tentative wording becomes intention without preserving uncertainty |
| 002 | A preference gets an invented valid-from date; negation is duplicated in the object and polarity |
| 003 | Current storage is represented as a completed action |
| 004 | Assistant text gets direct-user attribution and a CAUSES relation |
| 006 | A third-party report becomes a CAUSES relation with unclear speaker provenance |
| 009 | Unresolved “it,” “there” and “other one” become a retained plan |
| 010 | Explicitly unknown causality produces a CAUSES relation and out-of-vocabulary predicates |
| 011 | An unstated October 1 start date is invented; string `"null"` fails date validation |
| 012 | Current storage becomes a plan; “last week” gets an unsupported exact day |
| 014 | A conditional hypothetical becomes an intention |
| 017 | A thermometer reading becomes a CAUSES relation |
| 018 | A delivery destination becomes the shipping label's location |
| 019 | Unresolved reports get exact dates with an inferred year; an unsupported predicate invalidates the response |
| 022 | An abandoned historical plan becomes intention with an invented January 1 start and unsupported predicate |
| 023 | The affected spacer is used as the causal subject instead of the overtightened clamp |
| 024 | Tentative ownership becomes an asserted location relation |

Several simpler cases express a recognizable source fact but still have wrong
offsets or use the recording date as a validity bound. Exact-match F1 is zero;
that combines citation, schema, label and wording failures. It does not mean every
generated statement is semantically false. The gold set itself has only one
rendering and intentionally omits some incidental facts.

The initial prompt also underdefines field semantics. It does not clearly say
whether speaker role means the outer message or the quoted author, how recording
time differs from validity, or which hypothetical claims the memory policy retains.
Those ambiguities limit conclusions about the checkpoint. An explicit prompt
revision is a necessary comparator before attributing these failures to a need
for fine-tuning; it remains development on these same diagnostic cases.

The v1 anchoring pass raised evidence integrity to 21/24 valid claims; three remain
invalid because of one attribution conflict and two source-role mismatches.
Exact-match F1 remained zero. The repair changes offsets only and does not establish
entailment or justify auto-acceptance; these same-case results do not justify
fine-tuning or selecting Qwen over other candidates. See the
[computed anchoring report](baseline-qwen3-4b-anchoring.json).

## Explicit-v2 prompt comparison

Qwen3-4B's explicit-v2 output was audited against all 24 source cases on 2026-10-13.
These are prompt-development diagnostics, not independent evaluation data. Evidence is in the
[raw v2 receipt](../examples/synthetic/qwen3-4b-explicit-v2-remote.json), the
[raw-output report](baseline-qwen3-4b-explicit-v2-extraction.json), and the
[anchored report](baseline-qwen3-4b-explicit-v2-anchoring.json).

There are 24 outputs: case 008 reaches the 512-token limit and produces the single
parse failure. Case 022's year-only date `2023` and out-of-vocabulary `ABANDONED`
predicate produce the single schema failure. The other 21 parsed claims all have
invalid raw offsets: 16 quote mismatches and five out-of-bounds spans. Anchoring
repairs all 21; evidence integrity becomes 21/21 valid while exact-match F1 remains
zero. Abstention is 3/4 unknown cases, but the model also abstains on answerable
cases 006, 010, 017 and 019.

| Case | Observed behavior in the v2 output |
| --- | --- |
| 001 | Tentative wording still becomes intention |
| 002 | Negation appears in both object and polarity; `HAS` replaces negative `DECIDED` |
| 003 | Storage becomes asserted ownership with unsupported `HAS` |
| 004 | Keeps an assistant-only decoy and an awkward negative claim |
| 005 | A quoted invoice becomes a direct-user asserted report |
| 011 | Still invents the event date 2026-10-01 |
| 012 | Bought becomes `USES`; the user is located in the case rather than the clip, missing a cross-message referent |
| 014 | A counterfactual yields asserted telescope ownership and a user-attic location |
| 016 | The preference is recognizable, but a leading article differs from the single gold rendering |
| 018 | Loses the report source, changing a shipping-label / third-party statement to direct user speech |
| 020–021 | Still use `recorded_at` as event dates |
| 023 | Uses the user as `CAUSES` subject instead of the overtightened clamp |
| 024 | Tentative night-crew ownership becomes asserted user ownership |

The longer prompt reduced schema failures from four to one, but it also had one
parse failure, more abstentions, and unchanged exact-match F1 on the same 24 cases.
That does not establish an overall improvement or generalization. Offset validity
is separate from semantic support; this audit does not assign semantic-support
precision, estimate independent or human-review agreement, or select a model.
