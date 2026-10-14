"""Offline review-page rendering and export tests."""

from __future__ import annotations

import base64
import contextlib
import copy
import hashlib
import html
import importlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from tracemem.annotations import _canonical_json, _sha256, export_review_bundle, validate_responses

_NODE_VM_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const input = JSON.parse(fs.readFileSync(0, "utf8"));
const downloads = [];
const blobs = new Map();
let nextUrl = 0;

class MockElement {
  constructor(tag = "div") {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.listeners = {};
    this.files = [];
    this.value = "";
    this.textContent = "";
    this.checked = false;
    this.disabled = false;
    this.hidden = false;
  }
  append(...nodes) { this.children.push(...nodes); }
  appendChild(node) { this.children.push(node); return node; }
  replaceChildren(...nodes) { this.children = [...nodes]; }
  addEventListener(name, listener) { (this.listeners[name] ||= []).push(listener); }
  async emit(name, event = {}) {
    for (const listener of this.listeners[name] || []) {
      await listener({ target: this, ...event });
    }
  }
  async click() {
    if (this.tag === "a" && this.download) {
      const blob = blobs.get(this.href);
      downloads.push({ name: this.download, text: blob.text });
      return;
    }
    await this.emit("click");
  }
  remove() { this.removed = true; }
}

class MockBlob {
  constructor(parts) { this.text = parts.join(""); }
}

function boot(bundle) {
  const ids = [
    "review-bundle-data", "label-supported", "label-unsupported", "label-uncertain",
    "rationale", "reviewer-id", "reviewer-kind", "task-panel", "status", "candidate-fields",
    "source-messages", "citations", "progress", "empty-state", "task-heading", "previous",
    "next", "save-draft", "export-responses", "draft-file",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, new MockElement()]));
  elements["review-bundle-data"].textContent = JSON.stringify(bundle);
  elements["reviewer-kind"].value = "";
  for (const label of ["supported", "unsupported", "uncertain"]) {
    elements[`label-${label}`].value = label;
  }
  const document = {
    body: new MockElement("body"),
    getElementById: (id) => elements[id],
    createElement: (tag) => new MockElement(tag),
  };
  const context = {
    document,
    Blob: MockBlob,
    URL: {
      createObjectURL: (blob) => {
        const url = `blob:review-${nextUrl++}`;
        blobs.set(url, blob);
        return url;
      },
      revokeObjectURL: () => {},
    },
    setTimeout: (callback) => callback(),
  };
  vm.runInNewContext(input.script, context);
  return elements;
}

const elements = boot(input.bundle);
const badCitationElements = boot(input.outOfBoundsBundle);
const emptyElements = boot(input.emptyBundle);

function check(condition, message) {
  if (!condition) throw new Error(message);
}
function select(label) {
  for (const name of ["supported", "unsupported", "uncertain"]) {
    elements[`label-${name}`].checked = name === label;
  }
  return elements[`label-${label}`].emit("change");
}
async function saveDraft() {
  await elements["save-draft"].click();
  const file = downloads[downloads.length - 1];
  check(file.name === "review-draft.json", "draft should download as JSON");
  return JSON.parse(file.text);
}
async function importDraft(draft) {
  const fileInput = elements["draft-file"];
  fileInput.files = [{ text: async () => JSON.stringify(draft) }];
  await fileInput.emit("change");
}

(async () => {
  const taskIds = input.bundle.tasks.map((task) => task.task_id);
  check(taskIds.length === 2, "fixture must have two tasks");
  check(!elements["label-supported"].checked && !elements["label-unsupported"].checked &&
    !elements["label-uncertain"].checked, "labels must have no default");
  check(elements["reviewer-kind"].value === "", "reviewer kind must have no default");
  check(elements["export-responses"].disabled, "incomplete responses must not export");
  check(elements.citations.children[0].children[1].dataset.citationIntegrity === "match",
    "an exact Unicode citation must pass integrity checking");
  check(
    badCitationElements.citations.children[0].children[1].dataset.citationIntegrity === "mismatch",
    "an end offset beyond source text must fail integrity checking");
  check(!emptyElements["empty-state"].hidden && emptyElements["task-panel"].hidden &&
    emptyElements["export-responses"].disabled,
    "an empty bundle should show a safe no-task state");
  check(elements["source-messages"].children[0].children[2].textContent.includes("onerror=alert"),
    "untrusted source text must be assigned as text");

  elements["reviewer-id"].value = "reviewer-agent";
  await elements["reviewer-id"].emit("input");
  elements["reviewer-kind"].value = "agent";
  await elements["reviewer-kind"].emit("change");
  elements["rationale"].value = `${"a".repeat(3999)}🧭`;
  await elements["rationale"].emit("input");
  await select("supported");
  check(elements.progress.textContent.startsWith("1 of 2 complete"),
    "progress should count valid completed task responses");
  await elements["next"].click();
  elements["rationale"].value = "The second candidate is unsupported. 🍵";
  await elements["rationale"].emit("input");
  await select("unsupported");
  const saved = await saveDraft();
  check(saved.responses[0].task_id === taskIds[0], "draft rows should follow bundle order");
  check(saved.responses[0].rationale.length === 4001, "draft should preserve Unicode text");

  elements["rationale"].value = "unsaved work remains after a failed import";
  await elements["rationale"].emit("input");
  const malformed = JSON.parse(JSON.stringify(saved));
  malformed.responses[0].label = "uncertain";
  malformed.responses[1].label = "invalid";
  const wrongBundle = JSON.parse(JSON.stringify(saved));
  wrongBundle.bundle_sha256 = "f".repeat(64);
  const unknownTask = JSON.parse(JSON.stringify(saved));
  unknownTask.responses[1].task_id = "task_unknown";
  const duplicateTask = JSON.parse(JSON.stringify(saved));
  duplicateTask.responses[1].task_id = duplicateTask.responses[0].task_id;
  const whitespaceReviewer = JSON.parse(JSON.stringify(saved));
  whitespaceReviewer.reviewer_id = "\u0085";
  for (const invalid of [malformed, wrongBundle, unknownTask, duplicateTask, whitespaceReviewer]) {
    await importDraft(invalid);
    check(elements.status.textContent.startsWith("Draft not imported:"),
      "invalid drafts must be rejected");
  }
  const afterFailure = await saveDraft();
  check(afterFailure.responses[0].label === "supported",
    "invalid import must not partially replace earlier rows");
  check(afterFailure.responses[1].rationale === "unsaved work remains after a failed import",
    "invalid import must preserve unsaved work");

  await importDraft(saved);
  check(elements["task-panel"].dataset.taskId === taskIds[0],
    "restore should return to first task");
  check(elements["label-supported"].checked, "restore should reinstate the first label");
  check(elements["rationale"].value.length === 4001, "restore should reinstate Unicode rationale");
  await elements["next"].click();
  check(elements["label-unsupported"].checked, "restore should reinstate the second label");
  check(elements["rationale"].value === "The second candidate is unsupported. 🍵",
    "restore should reinstate the second rationale");

  elements["reviewer-id"].value = "reviewer-\uD800";
  await elements["reviewer-id"].emit("input");
  check(elements["export-responses"].disabled, "lone surrogates in reviewer IDs must be rejected");
  elements["reviewer-id"].value = "\u0085";
  await elements["reviewer-id"].emit("input");
  check(elements["export-responses"].disabled,
    "Python whitespace-only reviewer IDs must be rejected");
  elements["reviewer-id"].value = "reviewer-agent";
  await elements["reviewer-id"].emit("input");
  elements["rationale"].value = "bad-\uD800";
  await elements["rationale"].emit("input");
  check(elements["export-responses"].disabled, "lone surrogates in rationales must be rejected");
  elements["rationale"].value = "\u0085";
  await elements["rationale"].emit("input");
  check(elements["export-responses"].disabled,
    "Python whitespace-only rationales must be rejected");
  elements["rationale"].value = saved.responses[1].rationale;
  await elements["rationale"].emit("input");
  check(!elements["export-responses"].disabled,
    "4000 Unicode code points including an astral symbol must be accepted");
  await elements["export-responses"].click();
  const exported = JSON.parse(downloads[downloads.length - 1].text);
  check(downloads[downloads.length - 1].name === "review-responses.json",
    "complete responses should download to an explicit JSON file");
  process.stdout.write(JSON.stringify({ saved, exported }));
})().catch((error) => {
  process.stderr.write(error.stack || String(error));
  process.exitCode = 1;
});
"""


def _bundle(*, out_of_bounds: bool = False) -> dict[str, Any]:
    quote = "At dinner: café 🍵, </script><img src=x onerror=alert(1)>, and 東京."
    char_end = len(quote) + 100 if out_of_bounds else len(quote)
    case = {
        "case_id": "private-case",
        "scenario_id": "private-scenario",
        "split": "test",
        "tags": ["preference"],
        "messages": [
            {
                "session_id": "private-session",
                "message_id": "private-message",
                "role": "user",
                "text": quote,
                "recorded_at": "2026-10-01T12:00:00Z",
                "content_kind": "conversation",
            }
        ],
        "expected": [
            {
                "subject": "the user",
                "predicate": "PREFERS",
                "object": "quiet rooms </script> café",
                "speaker_role": "user",
                "attribution": "direct_user_statement",
                "modality": "preference",
                "polarity": "positive",
                "valid_from": None,
                "valid_to": None,
                "evidence": [
                    {
                        "session_id": "private-session",
                        "message_id": "private-message",
                        "char_start": 0,
                        "char_end": char_end,
                        "quote": quote,
                    }
                ],
            }
        ],
    }
    foil = copy.deepcopy(case["expected"][0])
    foil["object"] = "a separate candidate"
    return export_review_bundle([case], "a" * 64, {case["case_id"]: [foil]}, seed=7)[0]


def _empty_bundle() -> dict[str, Any]:
    body = {"version": 1, "dataset_sha256": "b" * 64, "tasks": []}
    return {**body, "bundle_sha256": _sha256(_canonical_json(body))}


class ReviewHtmlTests(unittest.TestCase):
    def test_renderer_embeds_only_safe_blind_content_and_unicode(self) -> None:
        module = importlib.import_module("tracemem.review_html")
        renderer = getattr(module, "render_review_html", None)
        self.assertTrue(callable(renderer), "render_review_html must be implemented")

        bundle = _bundle()
        page = renderer(bundle)
        match = re.search(
            r'<script id="review-bundle-data" type="application/json">(.*?)</script>',
            page,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match, "the page must contain an inert bundle data block")
        assert match is not None
        embedded = match.group(1)
        self.assertNotIn("</script>", embedded)
        self.assertIn("café", embedded)
        self.assertIn("東京", embedded)
        self.assertEqual(json.loads(embedded), bundle)
        for hidden in ("private-case", "private-scenario", "private-session", '"split"'):
            self.assertNotIn(hidden, page)
        self.assertNotIn("answer_key", page)
        self.assertNotRegex(page, r'type="radio"[^>]*checked')
        self.assertIn('<option value="" selected disabled>Select a kind</option>', page)
        self.assertIn('id="rationale" required', page)

    def test_csp_hashes_match_the_exact_inline_script_and_style(self) -> None:
        page = importlib.import_module("tracemem.review_html").render_review_html(_bundle())
        style = re.search(r"<style>(.*?)</style>", page, re.DOTALL)
        script = re.search(r'<script id="review-app">(.*?)</script>', page, re.DOTALL)
        policy = re.search(r'<meta http-equiv="Content-Security-Policy" content="([^"]+)">', page)
        self.assertIsNotNone(style)
        self.assertIsNotNone(script)
        self.assertIsNotNone(policy)
        assert style is not None and script is not None and policy is not None
        decoded_policy = html.unescape(policy.group(1))
        for block in (style.group(1), script.group(1)):
            digest = base64.b64encode(hashlib.sha256(block.encode("utf-8")).digest()).decode()
            self.assertIn(f"'sha256-{digest}'", decoded_policy)
        self.assertIn("default-src 'none'", decoded_policy)
        self.assertIn("connect-src 'none'", decoded_policy)
        self.assertNotIn("unsafe-inline", decoded_policy)

    def test_emitted_application_script_has_no_html_tokenizer_replacement_bytes(self) -> None:
        page = importlib.import_module("tracemem.review_html").render_review_html(_bundle())
        script = re.search(r'<script id="review-app">(.*?)</script>', page, re.DOTALL)
        self.assertIsNotNone(script)
        assert script is not None
        self.assertNotIn("\x00", script.group(1))
        self.assertNotIn("\r", script.group(1))

    def test_export_cli_writes_a_page_for_the_blind_bundle(self) -> None:
        cli = importlib.import_module("experiments.export_review_html")
        bundle = _bundle()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle_path = root / "blind.json"
            output_path = root / "review.html"
            bundle_path.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")

            result = cli.main(["--bundle", str(bundle_path), "--output", str(output_path)])

            self.assertEqual(result, 0)
            self.assertIn("Offline blind review", output_path.read_text(encoding="utf-8"))

    def test_renderer_rejects_a_bundle_with_a_changed_hash(self) -> None:
        bundle = _bundle()
        bundle["tasks"][0]["candidate"]["object"] = "tampered"
        renderer = importlib.import_module("tracemem.review_html").render_review_html

        with self.assertRaisesRegex(ValueError, "bundle hash mismatch"):
            renderer(bundle)

    def test_cli_preserves_existing_and_aliased_paths(self) -> None:
        cli = importlib.import_module("experiments.export_review_html")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle_path = root / "blind.json"
            original_bundle = json.dumps(_bundle())
            bundle_path.write_text(original_bundle, encoding="utf-8")
            existing_output = root / "existing.html"
            existing_output.write_text("keep this page", encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                aliased_result = cli.main(
                    ["--bundle", str(bundle_path), "--output", str(bundle_path)]
                )
                existing_result = cli.main(
                    ["--bundle", str(bundle_path), "--output", str(existing_output)]
                )

            self.assertEqual(aliased_result, 2)
            self.assertEqual(existing_result, 2)
            self.assertEqual(bundle_path.read_text(encoding="utf-8"), original_bundle)
            self.assertEqual(existing_output.read_text(encoding="utf-8"), "keep this page")

    def test_cli_rejects_duplicate_json_keys_without_creating_output(self) -> None:
        cli = importlib.import_module("experiments.export_review_html")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle_path = root / "duplicate.json"
            output_path = root / "review.html"
            valid_json = json.dumps(_bundle(), separators=(",", ":"))
            duplicate_json = valid_json.replace('{"version":1,', '{"version":999,"version":1,', 1)
            bundle_path.write_text(duplicate_json, encoding="utf-8")

            with contextlib.redirect_stderr(io.StringIO()):
                result = cli.main(["--bundle", str(bundle_path), "--output", str(output_path)])

            self.assertEqual(result, 2)
            self.assertFalse(output_path.exists())

    @unittest.skipUnless(os.name == "posix", "POSIX file mode check")
    def test_cli_creates_review_page_with_private_file_permissions(self) -> None:
        cli = importlib.import_module("experiments.export_review_html")
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            bundle_path = root / "blind.json"
            output_path = root / "review.html"
            bundle_path.write_text(json.dumps(_bundle()), encoding="utf-8")

            self.assertEqual(
                cli.main(["--bundle", str(bundle_path), "--output", str(output_path)]), 0
            )

            self.assertEqual(stat.S_IMODE(output_path.stat().st_mode) & 0o077, 0)

    def test_javascript_draft_restore_rejection_and_complete_export(self) -> None:
        node = shutil.which("node")
        if node is None:
            self.skipTest("Node.js is unavailable for the deterministic DOM harness")
        bundle = _bundle()
        page = importlib.import_module("tracemem.review_html").render_review_html(bundle)
        script = re.search(r'<script id="review-app">(.*?)</script>', page, re.DOTALL)
        self.assertIsNotNone(script)
        assert script is not None
        result = subprocess.run(
            [node, "-e", _NODE_VM_HARNESS],
            input=json.dumps(
                {
                    "bundle": bundle,
                    "outOfBoundsBundle": _bundle(out_of_bounds=True),
                    "emptyBundle": _empty_bundle(),
                    "script": script.group(1),
                },
                ensure_ascii=False,
            ),
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        run = json.loads(result.stdout)
        exported = run["exported"]
        self.assertEqual(validate_responses(bundle, exported), exported["responses"])
        self.assertEqual(
            len(exported["responses"][0]["rationale"]),
            4000,
            "the final response should be measured in Python-compatible code points",
        )


if __name__ == "__main__":
    unittest.main()
