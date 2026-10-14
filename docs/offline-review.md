# Offline review page

The offline review page is a local interface for a validated blind bundle. It reduces JSON editing friction for independent review; it does not create human labels by itself or satisfy Gate A. Keep the separate answer key out of the reviewer's workspace.

Create a new HTML file from a blind bundle:

```sh
PYTHONPATH=src python3 -m experiments.export_review_html \
  --bundle /path/to/blind-bundle.json \
  --output /path/to/review.html
```

The command accepts only the bundle and output paths. It uses the bundle validator and strict JSON parsing, rejects input/output path aliases, and refuses to overwrite an existing output. On POSIX systems it creates the HTML file with owner-only permissions. It does not modify the bundle or source data. The HTML contains full source messages and candidate text; draft files contain reviewer identity and entered responses, which may quote source text. Keep both on the reviewer's local machine and share them only through an approved private channel. Open the HTML file locally in a browser; it has no external assets, network requests, `localStorage`, or service worker.

Review one task at a time. Each task shows the full candidate and complete source messages. Choose **Supported**, **Unsupported**, or **Uncertain** and enter a rationale. No label is selected initially. Citation integrity is shown separately: the page checks the exact quote against the stated source message and Unicode character offsets. That check does not decide whether the source semantically supports the claim. Reviewer ID and reviewer kind are required for final export, and both are self-attested.

Save unfinished work with **Save draft** and restore it later with **Restore draft from file**. Draft files are tied to the bundle hash; the page validates every field and task ID before replacing the current draft. Drafts may be incomplete. No work is saved automatically, so unsaved changes are lost when the page is closed or reloaded.

**Export complete responses** stays disabled until every task has a label, nonblank rationale, reviewer ID, and reviewer kind. It downloads the version-1 response document accepted by `tracemem.annotations.validate_responses`. Keep the response document with its matching blind bundle for downstream validation and reporting.

Automated checks exercise the embedded app in a JavaScript VM and verify the CSP hashes, but do not perform a browser-level open of the local `file://` page in this environment. Open the generated file directly in a browser when browser-specific behavior needs validation.
