# Getting better training data

The current model has learned from short fictional dialogues. Real conversations
are messier: people change plans, quote documents, speak about someone else,
and leave decisions unfinished. More variations of the same fictional template
will not tell us whether the model handles those situations.

We need three complementary sources of evidence.

## 1. A private pilot from real sessions

Use selected personal conversations through the existing read-only adapters.
Keep original messages and evidence hashes intact. Create a separate, restricted
working copy containing only the context needed to judge each claim. Exclude
secrets, employer-confidential material and identifying details about other
people. Redactions need an explicit mapping to the original local evidence;
a paraphrase is not an exact quote from the original.

A review of the first four-window shortlist found no unambiguous direct
human-authored memory statements. Three windows contain injected environment
context and assistant plans; the fourth contains tool-transcript material.
Metadata checks found that all 22 selected sources are subagent sessions.
Their 36 prepared windows are not yet a useful personal-memory training set.
Preserve that snapshot as a failed selection and choose primary conversations
with substantive user dialogue. A provider's `user` role can include machine
instructions; role alone does not establish human authorship.

The next collection step follows the verified personal-project parent session,
then selects complete, bounded messages with original line/hash anchors. Keep
system metadata and tool transcripts available for local audit, rather than
mistaking them for the user's preferences or decisions. One project can support
an internal pilot; broad claims need more projects, people and providers.

Before model evaluation, assign complete sessions to train, development and
test groups. Keep overlapping sessions, copied documents and conversation
continuations together. Reserve a later time period or another personal
project when available. If the existing development snapshot is repartitioned,
disclose that curators have already seen it: it is an internal evaluation,
not a blind benchmark. Do not split claims randomly across these groups.

The owner permits private training and evaluation on personal Modal. Private
transcripts, claims, prompts, raw receipts and adapters trained on them stay out
of GitHub and public releases. Use a separate private experiment path; the
fiction-only launchers intentionally reject this provenance.

## 2. Labels with evidence and independent checks

For each candidate, record two answers:

- **Supported?** Does the conversation actually say this, with the same actor,
  reporting source, conditions, time and certainty?
- **Useful personal memory?** Does the claim fit the stated memory policy?
  A faithful statement about a fictional exercise can still be out of scope.

Keep `uncertain` available for both. Save the exact evidence quote and offsets,
the rationale, annotator identity and whether the annotator was an agent or a
human. Exact offsets prove where the words came from, not that the claim means
the same thing.

An initial annotator proposes claims. A second annotator sees the source and
claims without the first labels or model predictions. Compare their labels,
then adjudicate disagreements against the evidence. Report agreement before
adjudication, per label axis. Two agents are two annotation passes; they are
not a human gold standard or proof of independent model judgment.

Use owner review where it helps most: ambiguous personal-memory scope,
conflicting context, and a random sample of agreed labels. Keep agent-only
and human-reviewed results separate. Owner review can improve this personal
pilot; a broader benchmark needs more reviewers and more source diversity.
Human review is not a prerequisite for a disclosed provisional experiment.

For training, include faithful positives, genuine model mistakes, correctly
empty windows, and minimally altered negative claims. Examples should cover
assistant suggestions mistaken for user decisions, dropped conditions,
reversed negation, changed preferences, reported facts and uncertain actors.
Do not force equal numbers of good and bad claims in a naturalistic evaluation.
Use a separate balanced challenge set to measure specific failure modes.

## 3. A shareable benchmark with different sources

Inspect current licenses and terms before acquiring public conversations.
LongMemEval and LoCoMo are candidate evaluation sources, not automatically
approved training data. Pin each source version and retain its required
attribution. Keep benchmark evaluation conversations out of training, and
report possible contamination of the base model.

Author new shareable fiction from general failure patterns rather than
redacting and publishing real transcripts. Renaming a person does not make a
private conversation public training data. Separate authors/scenarios across
splits and include longer conversations and unrelated distractors.

## How we will decide whether the data helps

Freeze the same base model, existing adapter and one private-data candidate.
Score them on the same reserved sessions before choosing another training run.
Measure unsupported claims accepted, useful supported claims retained,
attribution and temporal errors, and uncertainty. Include confidence intervals
clustered by conversation; tiny pilots will have wide intervals.

Then test the complete memory pipeline: can it answer questions correctly
with the same answering model and evidence budget as ordinary transcript
retrieval? Better verifier scores alone do not establish a better assistant.

The first [private pilot](private-pilot-results.md) is complete: seven real user
messages, 14 reviewed contrast candidates, and a frozen three-model comparison.
The continued adapter accepted 4/7 supported eligible memories, matching the
base model. The source-linked private contract and train/evaluation exporters
work, but no private training rows have been used. The pilot stays in development.

The comparison exposes scope uncertainty on real project dialogue, along with
one false rejection of a supported question. Next collect distinct primary
sessions with actual adapter metadata, reusable preferences, temporary task
context and no-memory windows. Following the two other candidate parent chains
returned to the same original primary session; those chains do not provide
independent training conversations.

A bounded expansion screened four additional transcripts and 17 user-role
records. Fifteen records were excluded as machine/control text or possible
sensitive content. Two complete human messages from a second session passed
source and privacy review. They now supply four provisional training judgments:
two source-derived claims and two deliberately incorrect variants. Two agent
annotation passes agreed on all four support and eligibility labels. There
are still zero human reviewers. A draft that incorrectly identified an agent
privacy reviewer as human was rejected and corrected before acceptance.

The local corpus now has four training rows from one session, separate from
14 development candidates from the original session. There is no private test
cohort and no private training run. All current eligibility references are
eligible, so this batch cannot teach or measure scope discrimination. Retention
notes mark the new material as task-local; they do not turn an editing request
into a permanent preference. Private text, labels and source receipts remain
in ignored, restricted local files. The initial collection target below remains
unmet.

## Bounded batches and a stopping rule

These are prospective collection targets, not counts already available:

1. **Pilot:** 20 substantive windows from at least five primary sessions, with
   roughly 60–100 candidate judgments. Review all citations and privacy
   exclusions. Evaluate the existing adapters first; do not train on this
   development pilot and then call it held out.
2. **First training batch:** expand toward 100 useful windows across distinct
   primary sessions. Preserve natural positives and add targeted negative
   variants only in the training/challenge lane. Have the owner review policy
   disagreements plus a random sample of agreed labels. Keep the remaining
   agent-only labels explicitly provisional.
3. **Learning curve:** if the first private comparison exposes relevant errors,
   compare increasing training subsets against the same development cohort.
   A larger batch earns another run only if support errors fall without losing
   useful memories. Reserve a separately labeled test cohort for the final
   candidate and a matched downstream QA comparison.

Track collection yield (usable windows / inspected windows), pre-adjudication
label agreement, time per reviewed example, duplicate exclusions and residual
uncertainty. A failed selection is recorded rather than padded into the dataset.
There is no guarantee that more examples improve a 4B model; compare a stronger
prompt-only model on the same evidence to distinguish data gaps from capacity.

A 23/24 diagnostic catch rate cannot substantiate a 99% reliability claim.
Even zero errors among 100 independent accepted claims would be insufficient:
a one-sided exact 95% lower bound exceeds 99% only after at least 299 independent
error-free accepted claims. Correlated examples or repeated templates provide
less evidence. Report the observed denominators and uncertainty instead of
rounding a tiny diagnostic into production accuracy.
