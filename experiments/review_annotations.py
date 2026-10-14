"""Export blind scenario reviews and compare local reviewer response documents."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tracemem.annotation_report import build_review_report
from tracemem.annotations import export_review_bundle
from tracemem.challenge import _reject_constant, _unique_object
from tracemem.scenarios import load_scenarios


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="write a blind task bundle and separate answer key")
    export.add_argument("--bundle", required=True, type=Path)
    export.add_argument("--answer-key", required=True, type=Path)
    export.add_argument("--foils", type=Path, help="local JSON object mapping case IDs to claims")
    export.add_argument(
        "--unlabeled",
        type=Path,
        help="local JSON object mapping case IDs to model-generated candidates",
    )
    export.add_argument("--seed", required=True, type=int)
    report = commands.add_parser("report", help="compare complete local reviewer documents")
    report.add_argument("--bundle", required=True, type=Path)
    report.add_argument("--answer-key", required=True, type=Path)
    report.add_argument("--responses", required=True, nargs="+", type=Path)
    report.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            candidate_inputs = tuple(
                path for path in (args.foils, args.unlabeled) if path is not None
            )
            outputs = _preflight((args.bundle, args.answer_key), candidate_inputs)
            cases, manifest, dataset_sha = load_scenarios()
            foils = _read_json(args.foils) if args.foils else {}
            unlabeled = _read_json(args.unlabeled) if args.unlabeled else {}
            bundle, key = export_review_bundle(
                cases,
                dataset_sha,
                foils,
                seed=args.seed,
                case_metadata=manifest["case_metadata"],
                unlabeled_by_case=unlabeled,
            )
            _write_many(outputs, (bundle, key))
        else:
            inputs = (args.bundle, args.answer_key, *args.responses)
            outputs = _preflight((args.output,), inputs)
            bundle, key = _read_json(args.bundle), _read_json(args.answer_key)
            responses = [_read_json(path) for path in args.responses]
            result = build_review_report(bundle, key, responses)
            _write_many(outputs, (result,))
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"review_annotations: {exc}", file=sys.stderr)
        return 2
    return 0


def _preflight(outputs: Sequence[Path], inputs: Sequence[Path]) -> list[Path]:
    resolved_inputs = {path.resolve() for path in inputs}
    resolved_outputs = [path.resolve() for path in outputs]
    if len(set(resolved_outputs)) != len(resolved_outputs):
        raise ValueError("output paths alias each other")
    if set(resolved_outputs).intersection(resolved_inputs):
        raise ValueError("an output path aliases an input path")
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite existing output: {path}")
    return list(outputs)


def _write_many(paths: Sequence[Path], values: Sequence[Any]) -> None:
    _preflight(paths, ())
    created: list[tuple[Path, int, int]] = []
    try:
        for path, value in zip(paths, values, strict=True):
            with path.open("x", encoding="utf-8") as stream:
                info = os.fstat(stream.fileno())
                created.append((path, info.st_dev, info.st_ino))
                json.dump(value, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
    except Exception:
        for path, device, inode in created:
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == (device, inode):
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


def _read_json(path: Path) -> Any:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_unique_object,
        parse_constant=_reject_constant,
    )


if __name__ == "__main__":
    raise SystemExit(main())
