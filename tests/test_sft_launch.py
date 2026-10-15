from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import stat
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def _build_job(mode: str = "preflight") -> dict[str, object]:
    from experiments.modal_sft import build_payload

    if mode == "train":
        return build_payload(mode, run_id="test-run")
    return build_payload(mode)


class SFTLaunchApprovalTests(unittest.TestCase):
    def test_hash_or_approved_job_drift_fails_before_network_or_resources(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        tampered = copy.deepcopy(job)
        tampered["request"]["dev_prompts"][0]["prompt"] += " changed"

        def unexpected_call(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("approval drift must fail before network or resource creation")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            with (
                patch.object(launch, "_require_supported_python"),
                patch.object(launch, "_LAUNCH_LOCK", Path(temporary) / "launch.lock"),
                patch.object(launch, "verify_personal_workspace", side_effect=unexpected_call),
                patch.object(launch, "verify_modal_inventory", side_effect=unexpected_call),
                patch.object(launch, "_build_modal_function", side_effect=unexpected_call),
            ):
                for candidate, approval in (
                    (job, "0" * 64),
                    (tampered, job["job_sha256"]),
                ):
                    with self.subTest(approval=approval[:8]):
                        with self.assertRaises(ValueError):
                            launch.run_once(
                                candidate,
                                approved_job_sha256=approval,
                                output_path=output,
                            )
                        self.assertFalse(output.exists())


class SFTLaunchInventoryTests(unittest.TestCase):
    def test_inventory_accepts_empty_or_explicitly_stopped_zero_task_apps(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        stopped = {
            "app_id": "ap-0123456789012345678901",
            "description": "old pilot app",
            "state": "stopped",
            "tasks": "0",
            "created_at": "2026-10-15 10:00:00-07:00",
            "stopped_at": "2026-10-15 10:01:00-07:00",
        }

        self.assertEqual(launch.validate_modal_inventory("[]"), [])
        self.assertEqual(launch.validate_modal_inventory([stopped]), [stopped])

    def test_inventory_rejects_malformed_unknown_active_or_busy_apps(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        stopped = {
            "app_id": "ap-0123456789012345678901",
            "description": "old pilot app",
            "state": "stopped",
            "tasks": "0",
            "created_at": None,
            "stopped_at": None,
        }
        invalid = []
        for state in ("deployed", "ephemeral", "initializing...", "unknown", "STOPPED"):
            invalid.append({**stopped, "state": state})
        invalid.extend(
            (
                {**stopped, "tasks": "1"},
                {**stopped, "tasks": 0},
                {key: value for key, value in stopped.items() if key != "state"},
                {**stopped, "unexpected": True},
                {**stopped, "app_id": ""},
            )
        )
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                with self.assertRaises((ValueError, RuntimeError)):
                    launch.validate_modal_inventory([candidate])
        for candidate in ("{", "{}", "[null]"):
            with self.subTest(candidate=candidate):
                with self.assertRaises((ValueError, RuntimeError)):
                    launch.validate_modal_inventory(candidate)

    def test_fresh_inventory_uses_installed_cli_and_sixty_second_timeout(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        result = SimpleNamespace(returncode=0, stdout="[]", stderr="")
        with patch.object(launch.subprocess, "run", return_value=result) as run:
            self.assertEqual(launch.verify_modal_inventory(), [])

        args = run.call_args.args[0]
        self.assertEqual(args[1:], ["app", "list", "--json"])
        self.assertEqual(Path(args[0]), Path(sys.prefix) / "bin" / "modal")
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        self.assertEqual(run.call_args.kwargs["cwd"], ROOT)


class SFTLaunchReceiptAndLockTests(unittest.TestCase):
    def test_exclusive_receipt_preserves_existing_file(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        original = b"keep this prior receipt\n"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "receipt.json"
            output.write_bytes(original)
            with (
                patch.object(launch, "_require_supported_python"),
                patch.object(launch, "_LAUNCH_LOCK", root / "launch.lock"),
                patch.object(
                    launch,
                    "verify_personal_workspace",
                    side_effect=AssertionError("existing output must fail before checks"),
                ),
            ):
                with self.assertRaises(FileExistsError):
                    launch.run_once(
                        job,
                        approved_job_sha256=job["job_sha256"],
                        output_path=output,
                    )
            self.assertEqual(output.read_bytes(), original)

    def test_run_once_records_redacted_failure_and_launch_check_evidence(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "failure.json"
            with (
                patch.object(launch, "_require_supported_python"),
                patch.object(launch, "_LAUNCH_LOCK", root / "launch.lock"),
                patch.object(
                    launch,
                    "verify_personal_workspace",
                    side_effect=RuntimeError("API_KEY=secret-value"),
                ) as workspace,
                patch.object(launch, "verify_modal_inventory") as inventory,
                patch.object(
                    launch,
                    "_build_modal_function",
                    side_effect=AssertionError("failed workspace check must not build Modal"),
                ) as build,
            ):
                with self.assertRaisesRegex(RuntimeError, "API_KEY"):
                    launch.run_once(
                        job,
                        approved_job_sha256=job["job_sha256"],
                        output_path=output,
                    )

            workspace.assert_called_once()
            inventory.assert_not_called()
            build.assert_not_called()
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertFalse(receipt["remote_started"])
            self.assertFalse(receipt["launch_checks"]["workspace_verified"])
            self.assertNotIn("secret-value", output.read_text(encoding="utf-8"))

    def test_success_receipt_contains_remote_result_and_fresh_check_evidence(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        events: list[str] = []

        class App:
            def run(self, *, detach: bool):
                self.detach = detach
                return nullcontext()

        class Function:
            def remote(self, request: dict[str, object], approved_hash: str) -> dict[str, object]:
                events.append("remote")
                self.asserted_request = request
                self.asserted_hash = approved_hash
                return {"mode": "preflight", "status": "ready"}

        app = App()
        function = Function()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "success.json"
            with (
                patch.object(launch, "_require_supported_python"),
                patch.object(launch, "_LAUNCH_LOCK", root / "launch.lock"),
                patch.object(
                    launch,
                    "verify_personal_workspace",
                    side_effect=lambda: events.append("workspace"),
                ),
                patch.object(
                    launch,
                    "verify_modal_inventory",
                    side_effect=lambda: events.append("inventory") or [],
                ),
                patch.object(
                    launch,
                    "_build_modal_function",
                    side_effect=lambda _job: events.append("resources") or (app, function),
                ),
            ):
                receipt = launch.run_once(
                    job,
                    approved_job_sha256=job["job_sha256"],
                    output_path=output,
                )

            self.assertEqual(events, ["workspace", "inventory", "resources", "remote"])
            self.assertIs(app.detach, False)
            self.assertEqual(function.asserted_request, job["request"])
            self.assertEqual(function.asserted_hash, job["job_sha256"])
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, receipt)
            self.assertEqual(saved["result"], {"mode": "preflight", "status": "ready"})
            self.assertTrue(saved["launch_checks"]["workspace_verified"])
            self.assertTrue(saved["launch_checks"]["modal_inventory_checked"])
            self.assertEqual(saved["launch_checks"]["modal_app_count"], 0)
            self.assertIsNotNone(saved["launch_checks"]["started_at_utc"])
            self.assertIsNotNone(saved["launch_checks"]["completed_at_utc"])

    def test_nonblocking_lock_rejects_second_launcher_before_modal_creation(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock_path = root / "launch.lock"
            with (
                patch.object(launch, "_require_supported_python"),
                patch.object(launch, "_LAUNCH_LOCK", lock_path),
            ):
                held_lock = launch._acquire_launch_lock()
                try:
                    output = root / "busy.json"
                    with (
                        patch.object(launch, "verify_personal_workspace") as workspace,
                        patch.object(launch, "verify_modal_inventory") as inventory,
                        patch.object(
                            launch,
                            "_build_modal_function",
                            side_effect=AssertionError("busy lock must precede Modal creation"),
                        ) as build,
                    ):
                        with self.assertRaisesRegex(RuntimeError, "already in progress"):
                            launch.run_once(
                                job,
                                approved_job_sha256=job["job_sha256"],
                                output_path=output,
                            )
                    workspace.assert_not_called()
                    inventory.assert_not_called()
                    build.assert_not_called()
                    self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o600)
                    self.assertFalse(json.loads(output.read_text())["remote_started"])
                finally:
                    os.close(held_lock)


class SFTLaunchModalPackageTests(unittest.TestCase):
    def test_preflight_uploads_only_two_pinned_files_with_fixed_resources(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job()
        captured: dict[str, object] = {}

        class Image:
            @staticmethod
            def debian_slim(*, python_version: str) -> Image:
                captured["python_version"] = python_version
                return Image()

            def pip_install(self, *packages: str) -> Image:
                captured["packages"] = packages
                return self

            def add_local_file(self, local_path: Path, *, remote_path: str, copy: bool) -> Image:
                added = captured.setdefault("files", [])
                assert isinstance(added, list)
                added.append((Path(local_path), remote_path, copy))
                return self

        class App:
            def __init__(self, name: str) -> None:
                captured["app_name"] = name

            def function(self, *, image: Image, **options: object):
                captured["options"] = options
                return lambda function: function

        fake_modal = SimpleNamespace(
            Image=Image,
            App=App,
            Volume=SimpleNamespace(from_name=AssertionError("preflight must not use a Volume")),
        )
        with patch.dict(sys.modules, {"modal": fake_modal}):
            app, function = launch._build_modal_function(job)

        self.assertIsInstance(app, App)
        self.assertEqual(captured["python_version"], "3.12")
        self.assertEqual(captured["packages"], job["dependency_pins"])
        files = captured["files"]
        self.assertIsInstance(files, list)
        self.assertEqual(
            [local.relative_to(ROOT).as_posix() for local, _remote, _copy in files],
            ["training/train_sft.py", "training/remote_sft.py"],
        )
        self.assertEqual(
            [remote for _local, remote, _copy in files],
            ["/root/training/train_sft.py", "/root/training/remote_sft.py"],
        )
        self.assertTrue(all(copy_flag is True for _local, _remote, copy_flag in files))
        self.assertEqual(captured["app_name"], "tracemem-sft-pilot")
        expected_options = dict(job["resource_options"])
        expected_options["env"] = {"TRACEMEM_SFT_REMOTE": "1"}
        self.assertEqual(captured["options"], expected_options)
        self.assertIs(function.__closure__, None)
        self.assertEqual(function.__globals__["__name__"], "__main__")

    def test_training_attaches_only_the_fixed_volume_and_allows_creation(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job("train")
        captured: dict[str, object] = {}

        class Image:
            @staticmethod
            def debian_slim(*, python_version: str) -> Image:
                return Image()

            def pip_install(self, *_packages: str) -> Image:
                return self

            def add_local_file(self, *_args: object, **_kwargs: object) -> Image:
                return self

        class App:
            def __init__(self, _name: str) -> None:
                pass

            def function(self, *, image: Image, **options: object):
                captured["options"] = options
                return lambda function: function

        volume = object()

        def from_name(name: str, *, create_if_missing: bool) -> object:
            captured["volume_name"] = name
            captured["create_if_missing"] = create_if_missing
            return volume

        fake_modal = SimpleNamespace(
            Image=Image,
            App=App,
            Volume=SimpleNamespace(from_name=from_name),
        )
        with patch.dict(sys.modules, {"modal": fake_modal}):
            launch._build_modal_function(job)

        self.assertEqual(captured["volume_name"], "tracemem-sft-pilot")
        self.assertIs(captured["create_if_missing"], True)
        options = captured["options"]
        self.assertIsInstance(options, dict)
        self.assertEqual(options["volumes"], {"/mnt/tracemem-sft-pilot": volume})
        self.assertEqual(options["env"], {"TRACEMEM_SFT_REMOTE": "1"})
        self.assertIs(options["include_source"], False)


class SFTLaunchRemoteWrapperTests(unittest.TestCase):
    def _fixture(self, root: Path, mode: str = "preflight") -> tuple[dict[str, object], Path]:
        launch = importlib.import_module("experiments.sft_launch")
        job = _build_job(mode)
        training_dir = root / "training"
        training_dir.mkdir()
        train_source = (
            "import os, sys\n"
            "def run_remote_request(request):\n"
            "    if __name__ not in sys.modules:\n"
            "        raise RuntimeError('entry point missing from sys.modules')\n"
            "    if os.environ.get('TRACEMEM_SFT_REMOTE') != '1':\n"
            "        raise RuntimeError('remote environment missing')\n"
            "    return {'mode': request['mode'], 'status': 'ok'}\n"
        )
        remote_source = "REMOTE_SOURCE_MARKER = True\n"
        module_hashes = job["module_sha256"]
        self.assertIsInstance(module_hashes, dict)
        (training_dir / "train_sft.py").write_text(train_source, encoding="utf-8")
        (training_dir / "remote_sft.py").write_text(remote_source, encoding="utf-8")
        module_hashes["training/train_sft.py"] = hashlib.sha256(
            train_source.encode("utf-8")
        ).hexdigest()
        module_hashes["training/remote_sft.py"] = hashlib.sha256(
            remote_source.encode("utf-8")
        ).hexdigest()
        job["job_sha256"] = launch.canonical_job_sha256(job)
        return job, training_dir

    def test_wrapper_hashes_sources_loads_entrypoint_and_checks_full_job_before_import(
        self,
    ) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, training_dir = self._fixture(root)
            marker = root / "entry-loaded.txt"
            train_source = (
                "from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('loaded', encoding='utf-8')\n"
                "import os, sys\n"
                "def run_remote_request(request):\n"
                "    if __name__ not in sys.modules:\n"
                "        raise RuntimeError('entry point missing from sys.modules')\n"
                "    if os.environ.get('TRACEMEM_SFT_REMOTE') != '1':\n"
                "        raise RuntimeError('remote environment missing')\n"
                "    return {'mode': request['mode'], 'status': 'ok'}\n"
            )
            (training_dir / "train_sft.py").write_text(train_source, encoding="utf-8")
            module_hashes = job["module_sha256"]
            self.assertIsInstance(module_hashes, dict)
            module_hashes["training/train_sft.py"] = hashlib.sha256(
                train_source.encode("utf-8")
            ).hexdigest()
            job["job_sha256"] = launch.canonical_job_sha256(job)
            entry = launch._make_remote_entry(job, volume=None, training_dir=training_dir)
            request = job["request"]
            with patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}):
                changed_request = json.loads(json.dumps(request))
                changed_request["dev_prompts"][0]["prompt"] += " changed"
                with self.assertRaisesRegex(RuntimeError, "approved hash"):
                    entry(changed_request, job["job_sha256"])
                self.assertFalse(marker.exists())
                with self.assertRaisesRegex(RuntimeError, "approval mismatch"):
                    entry(request, "0" * 64)
                self.assertFalse(marker.exists())
                self.assertEqual(
                    entry(request, job["job_sha256"]),
                    {"mode": "preflight", "status": "ok"},
                )
                self.assertTrue(marker.exists())

            (training_dir / "remote_sft.py").write_text("changed source\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "source hash mismatch"):
                entry(request, job["job_sha256"])

    def test_training_wrapper_commits_volume_when_worker_raises(self) -> None:
        launch = importlib.import_module("experiments.sft_launch")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            job, training_dir = self._fixture(root, "train")
            train_source = (
                "def run_remote_request(request):\n"
                "    raise RuntimeError('tuned evaluation failed')\n"
            )
            (training_dir / "train_sft.py").write_text(train_source, encoding="utf-8")
            module_hashes = job["module_sha256"]
            self.assertIsInstance(module_hashes, dict)
            module_hashes["training/train_sft.py"] = hashlib.sha256(
                train_source.encode("utf-8")
            ).hexdigest()
            job["job_sha256"] = launch.canonical_job_sha256(job)

            class Volume:
                commits = 0

                def commit(self) -> None:
                    self.commits += 1

            volume = Volume()
            entry = launch._make_remote_entry(job, volume=volume, training_dir=training_dir)
            with patch.dict(os.environ, {"TRACEMEM_SFT_REMOTE": "1"}):
                with self.assertRaisesRegex(RuntimeError, "tuned evaluation failed"):
                    entry(job["request"], job["job_sha256"])
            self.assertEqual(volume.commits, 1)


if __name__ == "__main__":
    unittest.main()
