"""Render blind review bundles as self-contained offline HTML pages."""

from __future__ import annotations

import base64
import hashlib
import html
import json
from collections.abc import Mapping
from typing import Any

from .annotations import _validate_blind_bundle

_STYLE = """\
:root { color-scheme: light dark; font: 16px/1.5 system-ui, sans-serif; }
body { max-width: 76rem; margin: 0 auto; padding: 1rem; }
header, main, footer { margin-block: 1.5rem; }
.notice {
  border-inline-start: .3rem solid #b36b00;
  padding: .75rem 1rem;
  background: color-mix(in srgb, CanvasText 7%, Canvas);
}
.toolbar, .identity, .choices, .navigation {
  display: flex;
  flex-wrap: wrap;
  gap: .75rem;
  align-items: center;
}
.identity label, .field-row { display: grid; gap: .25rem; }
input, select, textarea, button { font: inherit; }
input[type="text"], select, textarea { box-sizing: border-box; max-width: 100%; padding: .45rem; }
textarea { width: min(100%, 52rem); min-height: 7rem; }
fieldset, article, .citation {
  border: 1px solid color-mix(in srgb, CanvasText 25%, Canvas);
  border-radius: .4rem;
  padding: 1rem;
  margin-block: 1rem;
}
dl { display: grid; grid-template-columns: minmax(9rem, 1fr) 3fr; gap: .35rem 1rem; }
dt { font-weight: 650; }
dd { margin: 0; white-space: pre-wrap; overflow-wrap: anywhere; }
.message-text, .quote { white-space: pre-wrap; overflow-wrap: anywhere; }
.integrity-good { color: #16713b; }
.integrity-bad { color: #a52b20; }
[hidden] { display: none !important; }
button:disabled { opacity: .55; }
@media (max-width: 40rem) { body { padding: .65rem; } dl { grid-template-columns: 1fr; } }
"""

_APP_SCRIPT = """\
(() => {
  "use strict";

  const bundle = JSON.parse(document.getElementById("review-bundle-data").textContent);
  const tasks = bundle.tasks;
  const labels = ["supported", "unsupported", "uncertain"];
  const radios = labels.map((label) => document.getElementById(`label-${label}`));
  const rationale = document.getElementById("rationale");
  const reviewerId = document.getElementById("reviewer-id");
  const reviewerKind = document.getElementById("reviewer-kind");
  const taskPanel = document.getElementById("task-panel");
  const status = document.getElementById("status");
  const rows = document.getElementById("candidate-fields");
  const sourceList = document.getElementById("source-messages");
  const citations = document.getElementById("citations");
  const responses = new Map(tasks.map((task) => [task.task_id, { label: "", rationale: "" }]));
  let currentIndex = 0;

  function element(tag, className) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function setStatus(message) {
    status.textContent = message;
  }

  function codePointLength(value) {
    return Array.from(value).length;
  }

  function validUnicode(value) {
    for (const character of value) {
      const point = character.codePointAt(0);
      if (point >= 0xd800 && point <= 0xdfff) return false;
    }
    return true;
  }

  function isPythonWhitespace(character) {
    const point = character.codePointAt(0);
    return (point >= 0x09 && point <= 0x0d) ||
      (point >= 0x1c && point <= 0x20) || point === 0x85 || point === 0xa0 ||
      point === 0x1680 || (point >= 0x2000 && point <= 0x200a) ||
      point === 0x2028 || point === 0x2029 || point === 0x202f ||
      point === 0x205f || point === 0x3000;
  }

  function hasNonWhitespace(value) {
    return Array.from(value).some((character) => !isPythonWhitespace(character));
  }

  function selectedLabel() {
    const checked = radios.find((radio) => radio.checked);
    return checked ? checked.value : "";
  }

  function saveVisibleResponse() {
    if (!tasks.length) return;
    responses.set(tasks[currentIndex].task_id, {
      label: selectedLabel(),
      rationale: rationale.value,
    });
  }

  function isValidIdentity() {
    return typeof reviewerId.value === "string" &&
      hasNonWhitespace(reviewerId.value) &&
      codePointLength(reviewerId.value) <= 256 &&
      validUnicode(reviewerId.value) &&
      ["human", "agent"].includes(reviewerKind.value);
  }

  function isValidResponse(row) {
    return labels.includes(row.label) &&
      typeof row.rationale === "string" &&
      hasNonWhitespace(row.rationale) &&
      codePointLength(row.rationale) <= 4000 &&
      validUnicode(row.rationale);
  }

  function isComplete() {
    return tasks.length > 0 && isValidIdentity() &&
      tasks.every((task) => isValidResponse(responses.get(task.task_id)));
  }

  function refresh() {
    const completeCount = tasks.filter((task) =>
      isValidResponse(responses.get(task.task_id))).length;
    document.getElementById("progress").textContent = tasks.length
      ? `${completeCount} of ${tasks.length} complete · Task ${currentIndex + 1} of ${tasks.length}`
      : "No tasks in this bundle";
    document.getElementById("previous").disabled = currentIndex <= 0 || !tasks.length;
    document.getElementById("next").disabled = currentIndex >= tasks.length - 1;
    document.getElementById("save-draft").disabled = !tasks.length;
    document.getElementById("export-responses").disabled = !isComplete();
  }

  function addField(name, value) {
    const term = element("dt");
    term.textContent = name;
    const description = element("dd");
    description.dataset.field = name;
    description.textContent = value === null ? "Not specified" : String(value);
    rows.append(term, description);
  }

  function findSource(task, span) {
    return task.messages.find((message) =>
      message.session_id === span.session_id && message.message_id === span.message_id);
  }

  function renderCitation(task, span, number) {
    const card = element("section", "citation");
    card.dataset.citationIndex = String(number);
    const heading = element("h4");
    heading.textContent = `Evidence ${number + 1}`;
    card.append(heading);

    const source = findSource(task, span);
    let intact = false;
    if (source) {
      const characters = Array.from(source.text);
      intact = span.char_start <= characters.length && span.char_end <= characters.length &&
        characters.slice(span.char_start, span.char_end).join("") === span.quote;
    }
    const integrity = element("p", intact ? "integrity-good" : "integrity-bad");
    integrity.dataset.citationIntegrity = intact ? "match" : "mismatch";
    integrity.textContent = intact
      ? "Citation integrity: quote matches the stated Unicode character offsets."
      : "Citation integrity: quote does not match the stated source and offsets.";
    card.append(integrity);

    const offsets = element("p");
    offsets.textContent =
      `Source IDs: ${span.session_id} / ${span.message_id}; ` +
      `offsets ${span.char_start}–${span.char_end} (end exclusive).`;
    card.append(offsets);
    const quoteLabel = element("strong");
    quoteLabel.textContent = "Cited quote";
    const quote = element("p", "quote");
    quote.textContent = span.quote;
    card.append(quoteLabel, quote);
    return card;
  }

  function renderTask() {
    if (!tasks.length) {
      taskPanel.hidden = true;
      document.getElementById("empty-state").hidden = false;
      refresh();
      return;
    }
    taskPanel.hidden = false;
    document.getElementById("empty-state").hidden = true;
    const task = tasks[currentIndex];
    const saved = responses.get(task.task_id);
    document.getElementById("task-heading").textContent = `Review task ${currentIndex + 1}`;
    taskPanel.dataset.taskId = task.task_id;

    rows.replaceChildren();
    for (const field of [
      "subject", "predicate", "object", "speaker_role", "attribution",
      "modality", "polarity", "valid_from", "valid_to",
    ]) {
      addField(field, task.candidate[field]);
    }
    sourceList.replaceChildren();
    task.messages.forEach((message, index) => {
      const article = element("article");
      article.dataset.sourceMessageId = message.message_id;
      article.dataset.sourceMessageIndex = String(index);
      const heading = element("h4");
      heading.textContent = `${message.role} · ${message.recorded_at || "time unavailable"}`;
      const metadata = element("p");
      metadata.textContent =
        `Message ${message.message_id} · ${message.session_id} · ${message.content_kind}`;
      const text = element("p", "message-text");
      text.dataset.sourceText = "true";
      text.textContent = message.text;
      article.append(heading, metadata, text);
      sourceList.append(article);
    });
    citations.replaceChildren();
    task.candidate.evidence.forEach((span, index) => {
      citations.append(renderCitation(task, span, index));
    });
    radios.forEach((radio) => { radio.checked = saved.label === radio.value; });
    rationale.value = saved.rationale;
    refresh();
  }

  function download(name, value) {
    const blob = new Blob([`${JSON.stringify(value, null, 2)}\n`], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = element("a");
    link.href = url;
    link.download = name;
    link.hidden = true;
    document.body.append(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function draftDocument() {
    saveVisibleResponse();
    return {
      version: 1,
      bundle_sha256: bundle.bundle_sha256,
      reviewer_id: reviewerId.value,
      reviewer_kind: reviewerKind.value,
      responses: tasks.map((task) => ({ task_id: task.task_id, ...responses.get(task.task_id) })),
    };
  }

  function exactKeys(value, expected) {
    if (value === null || typeof value !== "object" || Array.isArray(value)) return false;
    const actual = Object.keys(value).sort();
    const sortedExpected = expected.slice().sort();
    return actual.length === sortedExpected.length &&
      actual.every((key, index) => key === sortedExpected[index]);
  }

  function validateDraft(value) {
    if (!exactKeys(value, [
      "version", "bundle_sha256", "reviewer_id", "reviewer_kind", "responses",
    ])) {
      throw new Error("Draft fields are malformed.");
    }
    if (value.version !== 1 || value.bundle_sha256 !== bundle.bundle_sha256) {
      throw new Error("Draft belongs to a different bundle or version.");
    }
    if (typeof value.reviewer_id !== "string" || codePointLength(value.reviewer_id) > 256 ||
        !validUnicode(value.reviewer_id) ||
        (value.reviewer_id && !hasNonWhitespace(value.reviewer_id))) {
      throw new Error("Draft reviewer ID is malformed.");
    }
    if (typeof value.reviewer_kind !== "string" ||
        !["", "human", "agent"].includes(value.reviewer_kind)) {
      throw new Error("Draft reviewer kind is malformed.");
    }
    if (!Array.isArray(value.responses) || value.responses.length !== tasks.length) {
      throw new Error("Draft must contain one row for every task.");
    }
    const taskIds = new Set(tasks.map((task) => task.task_id));
    const seen = new Set();
    const checked = new Map();
    for (const row of value.responses) {
      if (!exactKeys(row, ["task_id", "label", "rationale"]) || typeof row.task_id !== "string" ||
          !taskIds.has(row.task_id) || seen.has(row.task_id)) {
        throw new Error("Draft contains an unknown, duplicate, or malformed task row.");
      }
      if (typeof row.label !== "string" || !["", ...labels].includes(row.label) ||
          typeof row.rationale !== "string" || codePointLength(row.rationale) > 4000 ||
          !validUnicode(row.rationale)) {
        throw new Error("Draft response fields are malformed.");
      }
      seen.add(row.task_id);
      checked.set(row.task_id, { label: row.label, rationale: row.rationale });
    }
    if (seen.size !== taskIds.size) throw new Error("Draft is missing a task row.");
    return { reviewerId: value.reviewer_id, reviewerKind: value.reviewer_kind, responses: checked };
  }

  function exportResponses() {
    saveVisibleResponse();
    if (!isComplete()) {
      setStatus("Complete every label, rationale, reviewer ID, and reviewer kind before export.");
      refresh();
      return;
    }
    const timestamp = new Date().toISOString().slice(0, 19) + "Z";
    const result = {
      version: 1,
      bundle_sha256: bundle.bundle_sha256,
      responses: tasks.map((task) => ({
        task_id: task.task_id,
        ...responses.get(task.task_id),
        reviewer_id: reviewerId.value,
        reviewer_kind: reviewerKind.value,
        timestamp,
      })),
    };
    download("review-responses.json", result);
    setStatus("Complete response document saved. Reviewer identity is self-attested.");
  }

  document.getElementById("previous").addEventListener("click", () => {
    saveVisibleResponse();
    currentIndex -= 1;
    renderTask();
  });
  document.getElementById("next").addEventListener("click", () => {
    saveVisibleResponse();
    currentIndex += 1;
    renderTask();
  });
  radios.forEach((radio) => radio.addEventListener("change", () => {
    saveVisibleResponse();
    refresh();
  }));
  rationale.addEventListener("input", () => { saveVisibleResponse(); refresh(); });
  reviewerId.addEventListener("input", refresh);
  reviewerKind.addEventListener("change", refresh);
  document.getElementById("save-draft").addEventListener("click", () => {
    download("review-draft.json", draftDocument());
    setStatus("Draft saved to an explicit file. Keep it with the matching bundle.");
  });
  document.getElementById("export-responses").addEventListener("click", exportResponses);
  document.getElementById("draft-file").addEventListener("change", async (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    try {
      const parsed = JSON.parse(await file.text());
      const checked = validateDraft(parsed);
      responses.clear();
      for (const [taskId, row] of checked.responses) responses.set(taskId, row);
      reviewerId.value = checked.reviewerId;
      reviewerKind.value = checked.reviewerKind;
      currentIndex = 0;
      renderTask();
      setStatus("Draft restored from the selected file.");
    } catch (error) {
      setStatus(`Draft not imported: ${error instanceof Error ? error.message : "invalid file"}`);
    } finally {
      event.target.value = "";
    }
  });

  renderTask();
  setStatus("Unsaved work is lost when this page is closed or reloaded.");
})();
"""


def render_review_html(bundle: Mapping[str, Any]) -> str:
    """Validate and render only the blind bundle as a self-contained page."""
    _validate_blind_bundle(bundle)
    safe_bundle = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    safe_bundle = (
        safe_bundle.replace("&", r"\u0026")
        .replace("<", r"\u003c")
        .replace(">", r"\u003e")
        .replace("\u2028", r"\u2028")
        .replace("\u2029", r"\u2029")
    )
    script_hash = base64.b64encode(hashlib.sha256(_APP_SCRIPT.encode("utf-8")).digest()).decode(
        "ascii"
    )
    style_hash = base64.b64encode(hashlib.sha256(_STYLE.encode("utf-8")).digest()).decode("ascii")
    policy = (
        "default-src 'none'; "
        f"script-src 'sha256-{script_hash}'; "
        f"style-src 'sha256-{style_hash}'; "
        "img-src 'none'; connect-src 'none'; font-src 'none'; media-src 'none'; "
        "object-src 'none'; base-uri 'none'; form-action 'none'"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="{html.escape(policy, quote=True)}">
<title>Offline blind review</title>
<style>{_STYLE}</style>
<script id="review-bundle-data" type="application/json">{safe_bundle}</script>
</head>
<body>
<header>
  <h1>Offline blind review</h1>
  <p>This page contains one task at a time and uses only the supplied blind bundle.</p>
  <p class="notice">Unsaved work is lost when this page is closed or reloaded.
    Save a draft to an explicit file. Reviewer identity is self-attested and is not verified.</p>
</header>
<main>
  <p id="status" role="status" aria-live="polite"></p>
  <p id="progress" data-testid="task-progress"></p>
  <p id="empty-state" hidden>This valid bundle has no review tasks.</p>
  <section id="task-panel" data-testid="review-task" hidden>
    <h2 id="task-heading"></h2>
    <section aria-labelledby="candidate-heading">
      <h3 id="candidate-heading">Full candidate</h3>
      <dl id="candidate-fields" data-testid="candidate-fields"></dl>
    </section>
    <section aria-labelledby="source-heading">
      <h3 id="source-heading">Complete source messages</h3>
      <div id="source-messages" data-testid="source-messages"></div>
    </section>
    <section aria-labelledby="citation-heading">
      <h3 id="citation-heading">Citation integrity</h3>
      <p>Exact quote and offsets are checked separately from the claim's semantic support.</p>
      <div id="citations" data-testid="citations"></div>
    </section>
    <fieldset>
      <legend>Semantic support</legend>
      <p>Does the source support the meaning of this candidate? A matching citation does not prove
        support, and a citation mismatch does not settle meaning.</p>
      <div class="choices">
        <label>
          <input id="label-supported" type="radio" name="label" value="supported"> Supported
        </label>
        <label>
          <input id="label-unsupported" type="radio" name="label" value="unsupported">
          Unsupported
        </label>
        <label>
          <input id="label-uncertain" type="radio" name="label" value="uncertain"> Uncertain
        </label>
      </div>
      <label for="rationale">Required rationale</label>
      <textarea id="rationale" required autocomplete="off"></textarea>
    </fieldset>
  </section>
  <section class="identity" aria-label="Reviewer identity">
    <label for="reviewer-id">Reviewer ID (self-attested)
      <input id="reviewer-id" type="text" autocomplete="off" required>
    </label>
    <label for="reviewer-kind">Reviewer kind
      <select id="reviewer-kind" required>
        <option value="" selected disabled>Select a kind</option>
        <option value="human">Human</option>
        <option value="agent">Agent</option>
      </select>
    </label>
  </section>
  <nav class="navigation" aria-label="Task navigation">
    <button id="previous" type="button">Previous</button>
    <button id="next" type="button">Next</button>
  </nav>
</main>
<footer>
  <div class="toolbar">
    <button id="save-draft" type="button">Save draft</button>
    <label for="draft-file">Restore draft from file</label>
    <input id="draft-file" type="file" accept=".json,application/json">
    <button id="export-responses" type="button" disabled>Export complete responses</button>
  </div>
</footer>
<script id="review-app">{_APP_SCRIPT}</script>
</body>
</html>
"""
