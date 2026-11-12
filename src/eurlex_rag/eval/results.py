"""Result files: the only source of a published number.

CLAUDE.md rule 1 says no figure appears in the README, a pull request or a
decision note unless it comes from a file in ``results/``. This module writes
those files, one per (rung, split), with everything needed to know where the
number came from: the code version (git commit, and whether there were
uncommitted changes), the retriever's settings, the chunking, and a hash of the
exact units and gold files used. If any of those differ, two numbers are not
comparable, and the file says so.

A re-run of the same rung on the same split overwrites its file, so the
directory always holds the latest measurement of each configuration.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCHEMA_VERSION = 1
DEFAULT_RESULTS_DIR = Path("results")


def sha256_file(path: Path | str) -> str | None:
    """Hex SHA-256 of a file's bytes, or ``None`` if it does not exist."""
    path = Path(path)
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 16), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str | None:
    """Output of a git command, or ``None`` when git or a repository is missing."""
    try:
        completed = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip()


def git_state() -> dict:
    """Current commit and whether the working tree has uncommitted changes."""
    commit = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain") if commit else None
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def result_path(out_dir: Path | str, rung: str, split: str) -> Path:
    return Path(out_dir) / f"{rung}__{split}.json"


def write_result(
    out_dir: Path | str,
    rung: str,
    split: str,
    evaluation: dict,
    retriever_info: dict,
    settings: dict,
) -> Path:
    """Write ``<out_dir>/<rung>__<split>.json`` and return its path.

    ``settings`` is whatever the caller used to build the run (chunking, k,
    file paths and hashes, counts); it is stored as given.
    """
    git = git_state()
    record = {
        "schema_version": SCHEMA_VERSION,
        "rung": rung,
        "split": split,
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git["commit"],
        "git_dirty": git["dirty"],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "argv": sys.argv,
        "retriever": retriever_info,
        "settings": settings,
        "evaluation": evaluation,
    }
    path = result_path(out_dir, rung, split)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def load_results(results_dir: Path | str) -> list[dict]:
    """Every result file in ``results_dir``, sorted by file name.

    Only files named ``<rung>__<split>.json`` are read, so other JSON files in
    the directory are left alone.
    """
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return []
    results = []
    for path in sorted(results_dir.glob("*__*.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}: not a valid result file ({error.msg})") from error
    return results
