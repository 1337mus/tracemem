"""Offline tests for the explicit public-model cache launcher."""

from __future__ import annotations

import builtins
import contextlib
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments import modal_verifier_pairs
from experiments import prepare_public_model_cache as cache


class PublicModelCacheTests(unittest.TestCase):
    def test_cli_is_offline_by_default(self):
        output = io.StringIO()
        original_import = builtins.__import__

        def reject_modal(name, *args, **kwargs):
            if name == "modal":
                raise AssertionError("offline prepare imported Modal")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=reject_modal):
            with contextlib.redirect_stdout(output):
                self.assertEqual(cache.main([]), 0)
        prepared = json.loads(output.getvalue())
        self.assertIs(prepared["remote_started"], False)
        self.assertEqual(prepared["payload"]["model_id"], "Qwen/Qwen3-4B")

    def test_envelope_is_cpu_only_source_free_and_mounts_only_public_cache(self):
        captured = {}

        class Volume:
            def __init__(self, name):
                self.name = name

            @classmethod
            def from_name(cls, name, **kwargs):
                captured["volume_options"] = kwargs
                captured["volume"] = cls(name)
                return captured["volume"]

        class Image:
            @classmethod
            def debian_slim(cls, **kwargs):
                return cls()

            def pip_install(self, *packages):
                captured["packages"] = packages
                return self

            def add_local_file(self, *args, **kwargs):
                raise AssertionError("local source upload is prohibited")

        class App:
            def __init__(self, name):
                self.name = name
                self.app_id = None

            def function(self, **options):
                captured["options"] = options
                return lambda function: function

            def run(self, **kwargs):
                raise AssertionError("envelope test must not start a Modal app")

        with patch.dict(
            sys.modules, {"modal": types.SimpleNamespace(Volume=Volume, Image=Image, App=App)}
        ):
            app, function = cache.build_modal_function("a" * 64, "b" * 64)

        opts = captured["options"]
        self.assertEqual(captured["packages"], modal_verifier_pairs._IMAGE_PACKAGES)
        self.assertEqual(captured["volume_options"], {"create_if_missing": True})
        self.assertEqual(opts["cpu"], 1)
        self.assertNotIn("gpu", opts)
        self.assertEqual(opts["timeout"], 600)
        self.assertEqual(opts["retries"], 0)
        self.assertEqual(opts["max_containers"], 1)
        self.assertTrue(opts["serialized"])
        self.assertFalse(opts["include_source"])
        self.assertEqual(opts["volumes"], {cache._VOLUME_MOUNT: captured["volume"]})
        self.assertEqual(captured["volume"].name, cache._VOLUME_NAME)
        self.assertEqual(
            set(function.__globals__),
            {
                "__name__",
                "_MODEL_ID",
                "_REVISION",
                "_ARTIFACT_FILES",
                "_VOLUME_MOUNT",
                "_APPROVED_SOURCE_SHA256",
                "_APPROVED_PAYLOAD_SHA256",
                "_REMOTE_VOLUME",
            },
        )
        self.assertEqual(function.__code__.co_freevars, ())
        self.assertEqual(app.name, "tracemem-public-model-cache")

    def test_wrong_source_or_payload_hash_fails_before_launch_or_receipt(self):
        payload = cache.build_payload()
        payload_hash = hashlib.sha256(cache._canonical(payload)).hexdigest()
        with patch.object(cache, "build_modal_function", side_effect=AssertionError("launched")):
            with tempfile.TemporaryDirectory() as directory:
                for source_hash, approved_payload in (
                    ("0" * 64, payload_hash),
                    (cache.source_sha256(), "1" * 64),
                ):
                    with self.subTest(
                        source_hash=source_hash[:4], payload_hash=approved_payload[:4]
                    ):
                        output = Path(directory) / "receipt.json"
                        with self.assertRaises(ValueError):
                            cache.run_once(source_hash, approved_payload, output)
                        self.assertFalse(output.exists())

    def test_source_free_remote_download_runs_with_only_public_files(self):
        captured = {}

        class Volume:
            def commit(self):
                captured["committed"] = True

        class Adapter:
            def __init__(self, **kwargs):
                captured["adapter_options"] = kwargs

        class Session:
            def mount(self, prefix, adapter):
                captured.setdefault("mounts", []).append((prefix, adapter))

        def snapshot_download(**kwargs):
            captured["download_options"] = kwargs
            root = Path(kwargs["cache_dir"]).parent / "snapshot"
            root.mkdir(parents=True)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors").write_bytes(b"public weights")
            (root / "private.bin").write_bytes(b"must not be reported")
            return str(root)

        requests = types.ModuleType("requests")
        requests.Session = Session
        adapters = types.ModuleType("requests.adapters")
        adapters.HTTPAdapter = Adapter
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = snapshot_download
        hub_utils = types.ModuleType("huggingface_hub.utils")
        hub_utils.configure_http_backend = lambda **kwargs: kwargs["backend_factory"]()
        fake_modules = {
            "requests": requests,
            "requests.adapters": adapters,
            "huggingface_hub": hub,
            "huggingface_hub.utils": hub_utils,
        }

        class ModalVolume:
            @classmethod
            def from_name(cls, name, **kwargs):
                captured["volume_name"] = name
                return Volume()

        class Image:
            @classmethod
            def debian_slim(cls, **kwargs):
                return cls()

            def pip_install(self, *packages):
                return self

        class App:
            def __init__(self, name):
                self.name = name

            def function(self, **kwargs):
                return lambda function: function

        payload = cache.build_payload()
        payload_hash = hashlib.sha256(cache._canonical(payload)).hexdigest()
        source_hash = cache.source_sha256()
        modal = types.SimpleNamespace(Volume=ModalVolume, Image=Image, App=App)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(sys.modules, {**fake_modules, "modal": modal}),
        ):
            _, remote = cache.build_modal_function(source_hash, payload_hash)
            remote_globals = remote.__globals__
            self.assertIn("_VOLUME_MOUNT", remote_globals)
            remote_globals["_VOLUME_MOUNT"] = directory
            result = remote(payload, payload_hash, source_hash)

        self.assertEqual(result["status"], "success", result)
        self.assertTrue(captured["committed"])
        self.assertEqual(captured["volume_name"], cache._VOLUME_NAME)
        self.assertEqual(captured["download_options"]["token"], False)
        self.assertEqual(
            captured["download_options"]["allow_patterns"], list(cache._ARTIFACT_FILES)
        )
        self.assertEqual(
            [file["name"] for file in result["files"]],
            ["config.json", "model.safetensors"],
        )
        self.assertEqual(result["file_count"], 2)
        self.assertNotIn("private.bin", {file["name"] for file in result["files"]})

    def test_receipt_directory_and_files_are_private_and_exclusive(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory) / "cache"
            output = output_dir / "receipt.json"
            with patch.object(cache, "_OUTPUT_DIR", output_dir):
                fd = cache._reserve_receipt(output)
                with os.fdopen(fd, "wb") as receipt:
                    receipt.write(b"first receipt")
                self.assertEqual(stat.S_IMODE(output_dir.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
                with self.assertRaises(FileExistsError):
                    cache._reserve_receipt(output)
                self.assertEqual(output.read_bytes(), b"first receipt")


if __name__ == "__main__":
    unittest.main()
