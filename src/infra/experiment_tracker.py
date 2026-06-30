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

from config import DATA_DIR, LLM_CONFIG


INDEX_PATH = DATA_DIR / "experiments" / "index.csv"
PROMPTS_DIR = Path(__file__).resolve().parent.parent / "llm" / "prompts"


# ---------------------------------------------------------------------------
# Shared index-column helpers (avoids duplicating the schema in every runner)
# ---------------------------------------------------------------------------

INDEX_COLUMNS = [
    "graph_id", "family", "method",
    "model_name", "prompt_file", "prompt_hash",
    "timestamp",
    "k", "num_physical_qubits", "compression_ratio",
    "optimal_energy", "approximation_ratio", "gradient_norm_at_opt",
    "success", "convergence_iters", "duration_seconds",
    "llm_tags", "llm_reasoning",
    "git_commit",
]


def build_index_record(
    record: Any,         # ExperimentRecord
    *,
    method: str,
    git_commit: str,
    model_name: str = "",
    prompt_file: str = "",
    prompt_hash: str = "",
    k: str = "",
    num_physical_qubits: str = "",
    compression_ratio: str = "",
    llm_tags: str = "",
    llm_reasoning: str = "",
) -> dict[str, str]:
    """Build a flat CSV row dict from an ExperimentRecord and extra fields.

    All values are converted to strings to keep the CSV homogeneous.
    """
    return {
        "graph_id": record.graph_id,
        "family": record.family,
        "method": method,
        "model_name": model_name,
        "prompt_file": prompt_file,
        "prompt_hash": prompt_hash,
        "timestamp": record.timestamp,
        "k": k,
        "num_physical_qubits": num_physical_qubits,
        "compression_ratio": compression_ratio,
        "optimal_energy": _fmt(record.optimal_energy),
        "approximation_ratio": _fmt(record.approximation_ratio),
        "gradient_norm_at_opt": _fmt(record.gradient_norm_at_opt),
        "success": str(record.success) if record.success is not None else "",
        "convergence_iters": _fmt(record.convergence_iters),
        "duration_seconds": _fmt(record.duration_seconds),
        "llm_tags": llm_tags,
        "llm_reasoning": llm_reasoning,
        "git_commit": git_commit,
    }


def _fmt(val: Any) -> str:
    """Format a value for CSV — empty string for None."""
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val:.12g}"
    return str(val)


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
        "quantization": LLM_CONFIG.get("quantization", None),
        "prompt_file": prompt_file or "none",
    }
    meta["prompt_hashes"] = _prompt_hashes()
    if extras:
        meta.update(extras)
    return meta


def append_index(record: dict[str, str]) -> None:
    """Append one run to the experiment index CSV, creating it if needed.

    The *record* dict should match the columns in INDEX_COLUMNS.
    Missing columns are filled with ``""``; extra columns are silently
    dropped.
    """
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    row = {col: record.get(col, "") for col in INDEX_COLUMNS}

    file_exists = INDEX_PATH.exists()
    with open(INDEX_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def load_index() -> list[dict[str, str]]:
    """Load the experiment index as a list of dicts (all values as strings)."""
    if not INDEX_PATH.exists():
        return []
    with open(INDEX_PATH, newline="") as f:
        return list(csv.DictReader(f))
