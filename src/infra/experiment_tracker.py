"""Experiment versioning and reproducibility tracker for QLLM.

Records prompt versions, model names, encoding strategies, and run
metadata in a central index CSV.
"""

import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA_DIR


INDEX_PATH = DATA_DIR / "experiments" / "index.csv"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


def _git_hash() -> str:
    """Return the current git commit hash, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _file_hash(path: Path) -> str:
    """Return SHA-256 hex digest of a file's contents."""
    if not path.exists():
        return "missing"
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _prompt_hashes() -> dict[str, str]:
    """Return {prompt_filename: content_hash} for all prompts."""
    if not PROMPTS_DIR.exists():
        return {}
    return {p.name: _file_hash(p) for p in sorted(PROMPTS_DIR.glob("*.txt"))}


def run_metadata(
    model_name: str,
    prompt_file: str | None = None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect versioning / reproducibility metadata for a run."""
    meta: dict[str, Any] = {
        "git_commit": _git_hash(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_name": model_name,
        "quantization": None,
        "prompt_file": prompt_file or "none",
    }
    meta["prompt_hashes"] = _prompt_hashes()
    if extras:
        meta.update(extras)
    return meta


def append_index(record: dict[str, Any]) -> None:
    """Append one run to the experiment index CSV, creating it if needed.

    Parameters
    ----------
    record : dict
        Flat dictionary with at minimum keys: graph_id, family, method,
        model_name, prompt_file, timestamp, git_commit, optimal_energy,
        approximation_ratio, success, duration_seconds.
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(record.keys())

    file_exists = INDEX_PATH.exists()
    with open(INDEX_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def load_index() -> list[dict[str, str]]:
    """Load the experiment index as a list of dicts (all values as strings)."""
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, newline="") as f:
        return list(csv.DictReader(f))
