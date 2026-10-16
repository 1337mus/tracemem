# Quote data version 3 contract

`validate_dataset(payload, split)` remains the default v2 validator. It accepts only an exact
integer `version` of `2`, allows one to six messages, and keeps its existing error wording and
other validation rules.

`validate_long_dataset(payload, split)` is a separate, explicit v3 entry point. It accepts only
an exact integer `version` of `3` and allows one to twelve messages. Both validators use the same
validation path for all other rules:

- Required fields, `MIT` license, `agent_authored` annotation, and `newly_authored_fiction`
  provenance.
- The requested split, unique case, entity, and message IDs, and consistent scenario/template
  groups.
- User, assistant, or tool roles; conversation, quoted-document, or tool-result content kinds;
  and timezone-aware ISO `recorded_at` dates.
- Claims with one exact quote from a user conversation message. Explicit `valid_from` and
  `valid_to` dates must be ISO dates that appear in the quote, and the start cannot follow the end.

Version 3 is opt-in. Existing v2 callers continue to call `validate_dataset`; no training caller
automatically accepts v3. The validator does not slice messages, upgrade versions, or relabel
existing data. Existing v2 fixtures and receipts remain unchanged.

## What validation does not establish

This contract checks structure and citation mechanics. It does not judge whether a claim is
correct, useful, supported by enough context, eligible for training, or likely to improve a model.
Those decisions remain unjudged.

## Future data changes

A future reader must explicitly copy or author data for v3, set its version and provenance, and
obtain a separate root review before using it for training. Do not infer approval from this
validator or from a passing structural check.

See the [validation record](validation-quote-data-v3.json) for the root review, tests, and limits.
