# Third extraction experiment

The second adapter sometimes turned a correction of a statement into a story
about a changed decision. It also blurred who owned an object and who kept it.
This experiment changes the training examples to address those errors, keeping
the model and training budget fixed. It is an exploratory synthetic experiment.

## What changes

Training uses 768 fictional examples: 12 variants from each of the previous 48
training families, plus eight variants from 24 new families. Half the new
families concern chronology; half concern ownership, custody and explicit
subjects. These variants are correlated examples, not 768 independent scenarios.
Expected outputs remain agent-authored provisional labels.

The base is the same pinned Qwen3-4B checkpoint, with the same quote output
schema, prompt, seed 17, QLoRA settings and 192 optimizer steps. The run compares
the base, the saved second adapter and a new adapter on identical prompts with
greedy decoding and a 512-token response limit. The saved adapter must be
hash-checked and unloaded without merging before a fresh adapter is trained.
Compare the fresh adapter's initial parameter hash with the second run's saved
initial hash. A difference is a failed initialization control to investigate;
do not attribute the result solely to changed training examples.

## Evaluation frozen before generation

There are three separate cohorts, with no pooled accuracy score:

| Cohort | Cases | Purpose |
| --- | ---: | --- |
| Previous development set | 24 | Adaptive regression check |
| New development set | 24 | Eight chronology, eight subjects, eight controls |
| New test set | 48 | Sixteen chronology, sixteen subjects, sixteen controls |

A separate agent authors the new development and test cases without reading
training examples or previous model outputs. The training author cannot read
the new evaluation fixtures. Root freezes training before reading test contents.
These precautions limit adaptation; they do not establish independent human
labels or representative coverage. Evaluation targets stay on the Mac; only
fictional prompt text is sent to the worker. The frozen test gets one use in
this experiment, without another tuning round against its errors.

Strict JSON/schema, exact quote and citation checks are reported for every arm,
including failures and truncations. A separately reported diagnostic removes
only an exact outer JSON Markdown fence; it never repairs content. Reference
string matches measure wording, not semantic correctness.

One fresh blinded reviewer sees shuffled prior/new responses and their sources
for all 48 test cases, including readable malformed responses. The reviewer
does not see arm names, targets, training data or metrics. Root then adjudicates
support and checks coverage against the frozen case requirements. Both stages
are provisional agent review, with zero human reviewers. Formatting may reveal
the arms. Base responses receive structural metrics; this protocol does not
claim a new semantic comparison against the base.

## Predeclared decision

Call the targeted change promising only if, on the new test:

- Fewer of the 32 chronology/subject responses contain unsupported or uncertain
  claims than the second adapter, and neither subgroup gets worse.
- The count of responses covering all required source facts does not decrease,
  so empty or incomplete answers cannot manufacture an improvement.
- Correct abstentions on the 12 empty-target cases and strict schema compliance
  do not decrease.

Report regression and control changes separately, including any tradeoff.
If those conditions fail, report a negative or mixed result. Report paired
case counts and denominators; this small single-seed experiment cannot establish
general accuracy, statistical reliability, or downstream memory usefulness.

## Compute and data boundaries

One A10G worker, one container, zero retries, a 3,600-second timeout and one
remote call. Check the personal Modal workspace and live inventory immediately
before dispatch. A new volume stores the new adapter; the old volume is an input
and must not be committed or modified. Freeze the exact job, source and prior
adapter hashes before launch. No weights or tokenizer run on this Mac.

Your Context Kit corpus is excluded. Earlier local inspection measured formats,
roles, timestamps and privacy indicators. Neither completed training run used
your conversations; both used newly authored fiction. This experiment also
uses fiction. A real-history evaluation is still needed before claiming it
works for your day-to-day conversations.
