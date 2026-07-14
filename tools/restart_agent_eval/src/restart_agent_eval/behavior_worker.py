"""Isolated worker for deterministic product behavior capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from .behavior import build_fixture_in_worker


def main(
    argv: list[str] | None = None,
    *,
    fixture_builder: Callable[[Path, Path], dict] = build_fixture_in_worker,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--product-repo", type=Path, required=True)
    args = parser.parse_args(argv)
    fixture = fixture_builder(args.log.resolve(), args.product_repo.resolve())
    print(json.dumps(fixture, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
