"""Export a blind review bundle as a self-contained offline HTML page."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from experiments.review_annotations import _preflight, _read_json
from tracemem.review_html import render_review_html


def main(argv: Sequence[str] | None = None) -> int:
    """Read one strict blind bundle and create a new offline HTML file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        output = _preflight((args.output,), (args.bundle,))[0]
        page = render_review_html(_read_json(args.bundle))
        _write_html(output, page)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(f"export_review_html: {exc}", file=sys.stderr)
        return 2
    return 0


def _write_html(path: Path, page: str) -> None:
    """Create a new output and remove only our own partial file on failure."""
    _preflight((path,), ())
    created: tuple[int, int] | None = None
    descriptor = -1
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        info = os.fstat(descriptor)
        created = (info.st_dev, info.st_ino)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        descriptor = -1
        with stream:
            stream.write(page)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if created is not None:
            try:
                info = path.lstat()
                if (info.st_dev, info.st_ino) == created:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
