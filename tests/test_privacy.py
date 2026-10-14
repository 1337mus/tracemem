"""Heuristic privacy signal tests. Synthetic samples only."""

import importlib
import importlib.util
import unittest
from dataclasses import FrozenInstanceError, fields


class PrivacyScanTests(unittest.TestCase):
    def privacy_module(self):
        spec = importlib.util.find_spec("tracemem.privacy")
        self.assertIsNotNone(spec, "privacy scanner module is not implemented")
        return importlib.import_module("tracemem.privacy")

    def test_detects_supported_patterns_with_ranges_and_safe_controls(self) -> None:
        privacy = self.privacy_module()
        cases = [
            ("github_token", "ghp_" + "A" * 24),
            ("openai_token", "sk-proj-" + "B" * 24),
            ("aws_access_key_id", "AKIA" + "C" * 16),
            ("bearer_token", "Bearer " + "D" * 24),
            (
                "pem_private_key",
                "-----BEGIN PRIVATE KEY-----\nSYNTHETIC-KEY-MATERIAL\n-----END PRIVATE KEY-----",
            ),
            ("email", "synthetic.user@example.invalid"),
        ]

        for category, indicator in cases:
            with self.subTest(category=category):
                text = "safe prefix " + indicator + " safe suffix"
                findings = privacy.scan(text)

                self.assertEqual(len(findings), 1)
                self.assertEqual(
                    (findings[0].category, findings[0].char_start, findings[0].char_end),
                    (category, len("safe prefix "), len("safe prefix ") + len(indicator)),
                )
                self.assertEqual(
                    {field.name for field in fields(findings[0])},
                    {"category", "char_start", "char_end"},
                )
                with self.assertRaises(FrozenInstanceError):
                    findings[0].category = "changed"

        safe_controls = (
            "ghp_short",
            "sk-short",
            "AKIA1234",
            "prefix" + "AKIA" + "E" * 16,
            "Bearer short",
            "synthetic user at example.invalid",
            "-----BEGIN PUBLIC KEY-----\nSYNTHETIC\n-----END PUBLIC KEY-----",
        )
        for text in safe_controls:
            with self.subTest(safe_control=text[:12]):
                self.assertEqual(privacy.scan(text), [])

    def test_redaction_preview_replaces_only_detected_text(self) -> None:
        privacy = self.privacy_module()
        original = "safe prefix synthetic.user@example.invalid safe suffix"

        preview = privacy.redaction_preview(original)

        self.assertEqual(
            preview,
            "safe prefix [REDACTED] safe suffix",
        )
        self.assertEqual(original, "safe prefix synthetic.user@example.invalid safe suffix")


if __name__ == "__main__":
    unittest.main()
