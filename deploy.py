#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DIST = ROOT / "dist"
DEFAULT_BIN = ROOT / "bin"


def _copy_binaries(src_dir: Path, dest_dir: Path) -> list[tuple[Path, Path]]:
    if not src_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")
    if not src_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {src_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)

    copied: list[tuple[Path, Path]] = []
    for src_path in sorted(src_dir.iterdir()):
        if not src_path.is_file():
            continue
        dest_path = dest_dir / src_path.name
        shutil.copy2(src_path, dest_path)
        copied.append((src_path, dest_path))
    if not copied:
        raise RuntimeError(f"No files copied from {src_dir} (directory empty?)")
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deploy built binaries from the dist directory into bin/."
    )
    parser.add_argument(
        "--src",
        default=str(DEFAULT_DIST),
        help=f"Source directory containing built binaries (default: {DEFAULT_DIST})",
    )
    parser.add_argument(
        "--dest",
        default=str(DEFAULT_BIN),
        help=f"Destination directory for deployed binaries (default: {DEFAULT_BIN})",
    )
    args = parser.parse_args(argv)

    src_dir = Path(args.src).resolve()
    dest_dir = Path(args.dest).resolve()

    copied = _copy_binaries(src_dir, dest_dir)
    print("Deployed binaries:")
    for src_path, dest_path in copied:
        print(f"- {src_path} -> {dest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
