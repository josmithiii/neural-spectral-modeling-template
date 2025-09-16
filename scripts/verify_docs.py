#!/usr/bin/env python3
"""Documentation sanity checks for NSMT."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOC_DIRS = [PROJECT_ROOT / "docs", PROJECT_ROOT]
IGNORE_DIR_NAMES = {".venv"}
BANNED_PATTERNS = [
    r"make\s+sds",
    r"make\s+sdl",
    r"make\s+sdm",
    r"make\s+sdma",
]

LINK_PATTERN = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^\)]+)\)")
HEADING_PATTERN = re.compile(r"^(#+)\s+(.+)$")


def iter_markdown_files() -> list[Path]:
    files: list[Path] = []
    for base in DOC_DIRS:
        for path in base.rglob("*.md"):
            if path.name.startswith("."):
                continue
            if any(part in IGNORE_DIR_NAMES for part in path.parts):
                continue
            files.append(path)
    return files


def check_banned_patterns(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    for pattern in BANNED_PATTERNS:
        if re.search(pattern, text, flags=re.IGNORECASE):
            failures.append(f"{path}: contains deprecated command reference matching '{pattern}'")
    return failures


def check_links(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    for match in LINK_PATTERN.finditer(text):
        target = match.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        if target.startswith("#") or target.startswith("?"):
            continue
        target_path_str = target.split("#", 1)[0]
        if not target_path_str:
            continue
        resolved = (path.parent / target_path_str).resolve()
        if not resolved.exists():
            failures.append(f"{path}: broken link to '{target}'")
    return failures


def check_headings(path: Path, text: str) -> list[str]:
    failures: list[str] = []
    in_code_fence = False
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if in_code_fence:
            continue
        match = HEADING_PATTERN.match(line)
        if not match:
            continue
        heading_text = match.group(2).strip()
        if not heading_text:
            failures.append(f"{path}:{line_no}: empty heading")
            continue
        first_char = heading_text[0]
        if not (first_char.isalpha() or first_char.isdigit() or first_char == "`"):
            failures.append(
                f"{path}:{line_no}: heading should start with letter/number/backtick (found '{first_char}')"
            )
    return failures


def main() -> int:
    failures: list[str] = []
    for path in iter_markdown_files():
        text = path.read_text(encoding="utf-8")
        failures.extend(check_banned_patterns(path, text))
        failures.extend(check_links(path, text))
        failures.extend(check_headings(path, text))
    if failures:
        for failure in failures:
            print(failure)
        print(f"\n{len(failures)} documentation issues detected.")
        return 1
    print("Documentation checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
