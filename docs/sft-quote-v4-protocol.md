# Fourth experiment: can better targets recover missing facts?

Status: Protocol frozen before generation; see handoff for live run status. This is a small,
fiction-only diagnostic, with provisional agent labels and zero human reviewers.

The third adapter made fewer adverse responses but omitted more required facts.
The training audit did not identify a confirmed critical fact missing from an
entire target set. We therefore cannot attribute that result to target omissions.
This experiment tests seven specific corrections to tense, reported speech,
standalone references and time scope. See [the audit](training-target-review.md).

Training stays at 768 rows from 72 correlated families, 192 optimizer steps,
seed 17, the same pinned Qwen3-4B base and QLoRA settings, and a 2048-token
training limit. The reviewed targets contain 696 claims. Original transcripts,
seeds, prior training sets and historical results remain unchanged.
The new corpus also prefixes record IDs with `V4T-`, which changes prompt and
target tokens. This is not a pure target-text ablation.

The training sources and review files were hash-frozen before a fresh-context
author was assigned 12 development and 24 test conversations. That author was
given no training examples, prior evaluation cases or model outputs. The new
cases must pass source/target review and split checks before their final freeze.
This process reduces direct reuse; it does not prove semantic independence.
The previous v3 test is spent and is excluded from this experiment.

Three arms will read the same new conversations: the pinned base model, the
unchanged v3 adapter, and a fresh adapter trained on the reviewed targets.
All arms use greedy generation with a 1024-token output cap and a 4096-token
evaluation context. The output cap is higher than v3's historical 512-token
cap; compare arms within this experiment, not raw counts across experiments.
Keep all raw output, truncations, parse failures and citation failures.

Before generation, freeze the final fixture hashes, job hash and review rubric.
Review outputs with anonymous arm IDs, then adjudicate source support and
required-fact coverage separately. An adverse response contains any unsupported
or uncertain claim. An answerable case has complete coverage only when every
required fact is present with the right actor, modality and time scope.
Formatting and reference-string matches are separate diagnostic measurements.
Required facts are a selected set, not an exhaustive list of useful memories.
Semantic review reads the full source context and allows unambiguous pronoun
resolution. It also notes when the cited quote alone lacks necessary context;
literal citation validity does not establish semantic support.

The predeclared promising-result threshold is at least three additional complete
nonempty test cases versus the prior v3 adapter, with no increase in adverse
test responses or invalid citations. Report every case, including failures,
and keep dev/test results separate. Do not tune on test outputs. Passing this
small diagnostic would justify a larger evaluation; it would not establish
generalization, Gate A, production readiness or a downstream QA improvement.
If it fails, preserve the negative result and compare alternative base models
and a stronger prompt-only baseline before another round of target tweaks.

One reviewed synthetic-only Modal call may run at a time in the personal
workspace. Keep the prior volume read-only, verify its exact adapter hashes,
and save the new adapter in a separate volume. No personal conversations or
private derived data enter the job or Git repository. Model weights stay on Modal.
