from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from tests.test_remote_quote_v3 import _run_quote_training, _write_fake_prior_adapter


class RemoteQuoteV4Tests(unittest.TestCase):
    def test_v3_prior_and_fresh_adapter_share_the_1024_token_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = _write_fake_prior_adapter(root / "prior-v3")
            run = _run_quote_training(
                root / "v4-run",
                metadata,
                profile_name="quote-v4",
                dev_max_new_tokens=1024,
            )

        result = run["result"]
        calls = run["generation_calls"]
        self.assertEqual(result["status"], "completed")
        self.assertEqual([call[0] for call in calls], ["prepared", "prior", "fresh"])
        self.assertTrue(all(call[2] == 1024 for call in calls))
        self.assertTrue(all(call[1] == calls[0][1] for call in calls[1:]))
        self.assertEqual(result["generation"]["max_new_tokens"], 1024)
        self.assertEqual(result["generation"]["prior_outputs"][0]["model"], "prior")
        self.assertEqual(result["comparison"]["prior_adapter"], metadata)
        self.assertTrue(result["comparison"]["verified_before"])
        self.assertTrue(result["comparison"]["verified_after"])
        self.assertTrue(result["comparison"]["rng_restored"])
        self.assertTrue(result["comparison"]["unloaded_without_merge"])

    def test_v4_prior_hash_mismatch_stops_before_baseline_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = _write_fake_prior_adapter(root / "prior-v3")
            metadata["sha256"]["adapter_model.safetensors"] = "0" * 64
            tracking: dict[str, object] = {}

            with self.assertRaisesRegex(RuntimeError, "quote-v4 prior adapter hash mismatch"):
                _run_quote_training(
                    root / "v4-run",
                    metadata,
                    profile_name="quote-v4",
                    dev_max_new_tokens=1024,
                    tracking=tracking,
                )

        self.assertEqual(tracking["generation_calls"], [])
        self.assertFalse(any(event[0] == "load-prior" for event in tracking["events"]))


if __name__ == "__main__":
    unittest.main()
