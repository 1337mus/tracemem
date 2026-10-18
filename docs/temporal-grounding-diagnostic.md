# Temporal-grounding verifier diagnostic

## Original comparison without calendar evidence

This inference-only check compared the base verifier with two existing adapters on 32 newly written fictional train-only claims. The claims formed 16 paired scenarios across four families: date order, elapsed time, event or report date, and intention versus completion. They are not a held-out evaluation.

| Existing system | Support labels matched | Supported claims retained | Unsupported claims rejected | Unsupported claims accepted | Eligibility labels matched |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 27/32 | 16/16 | 11/16 | 5/16 | 32/32 |
| Historical 256-exposure adapter | 27/32 | 12/16 | 15/16 | 1/16 | 31/32 |
| Latest matched adapter | 27/32 | 14/16 | 13/16 | 3/16 | 32/32 |

All three systems matched the support labels on 8/8 date-order examples, 4/8 elapsed-time examples, 7/8 event-or-report-date examples, and 8/8 intention-versus-completion examples. Elapsed-time reasoning was the shared weak area, despite different error patterns. The latest adapter rejected two more unsupported claims than the base, while retaining two fewer supported claims; all three had the same overall support score. It improved supported retention over the historical adapter by two claims, while rejecting two fewer unsupported claims. The comparison shows a trade-off, not an overall gain.

The author created the claims and supplied support labels. A second agent labeled them without seeing those labels, and root independently labeled them before scoring against the frozen references. There were zero human reviewers. The examples use templated language with limited variation. Eligibility was disclosed in the protocol and every example was personal, so this was not a blind scope-calibration test. There were no uncertainty references: the latest adapter made one uncertain-support prediction, while the historical adapter made one uncertain-eligibility prediction. Support and eligibility are separate scoring axes.

All 96 outputs were valid and ended with EOS; none were truncated. The three arms used identical input-token arrays for all 32 claims. Remote runtime was 144.4 seconds and client elapsed time was 158.5 seconds. Each arm used 13,034 input tokens; generated tokens were 576 for the base and 353 for each adapter, 1,282 total. Billed cost is unknown.

No model was trained or selected in this diagnostic, and it does not establish held-out QA benefit. The earlier answering-instructions comparison remains unchanged at 13/24 grounded-correct answers with hybrid retrieval and 10/24 with answer-marked evidence; it is a separate prompt-only result, not evidence of verifier or trained-memory benefit.

## Calendar-evidence augmentation

A separate run used the same 32 references and three frozen model/adapter versions with the same settings. A deterministic helper appended calendar-day intervals computed from all source-date pairs, preserving the original prompt prefix and giving paired candidates identical additions; no labels were inputs. Only the prompt tail changed. These results are separate from the original comparison above.

| System with appended evidence | Support labels matched | Elapsed-time labels matched | Supported claims retained | Unsupported claims rejected | False accepts | Eligibility labels matched |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 28/32 | 4/8 | 16/16 | 12/16 | 1/16 | 29/32 |
| Historical adapter | 29/32 | 6/8 | 13/16 | 16/16 | 0/16 | 32/32 |
| Latest matched adapter | 28/32 | 5/8 | 15/16 | 13/16 | 1/16 | 32/32 |

All three systems matched 27/32 support labels without the added evidence. The latest adapter improved elapsed-time exactness from 4/8 to 5/8, below the fixed minimum of 6/8; it also rejected only 1/4 unsupported elapsed-time claims, below the required three. It retained one more supported claim than before (15/16 versus 14/16), but unsupported rejection remained 13/16 overall. Its false accepts fell from 3/16 to 1/16, alongside two uncertain-support predictions; the references include no uncertain-support examples. The historical adapter reached 6/8 elapsed-time exactness but retained only 2/4 supported elapsed-time claims. The fixed 3/4 retention requirement applied to the latest adapter, which retained 4/4; its exactness and rejection checks still failed. Base eligibility exactness fell from 32/32 to 29/32, with two new eligibility errors outside the elapsed-time family. The full set of fixed checks failed, so no model was promoted.

All 96 outputs had the required two-key JSON, ended with EOS, and were not truncated. Input-token arrays matched across the three arms. Input use rose from 13,034 tokens per arm in the original run to 30,726 per arm here. Output counts were 582, 352, and 354 tokens (1,288 total). Remote runtime was 201.1 seconds and client elapsed time was 216.8 seconds; billed cost is unknown. A full `make check` exited successfully: 1,158 tests ran, 1,157 passed and one optional test was skipped; Ruff covered 233 files and mypy covered 76.

An independent tally matched the aggregate and family-level counts on both scoring axes, and root verified that agreement. The run used only fictional train-only claims, with templated language across 16 scenarios, no held-out set, no human reviewers, no new training, and no model selection. The helper computes Gregorian calendar-day differences from cited literal date spans and their record revisions; it does not determine clock-hour duration or event identity. Do not attribute all remaining errors to arithmetic alone: the references use the same Gregorian date-difference convention, and the base system also added eligibility errors outside the elapsed-time family.

No further run with unchanged examples and prompts is justified. The next evaluation should use more varied date-event and precision examples, followed by independent real-session data with human review and calibration.
