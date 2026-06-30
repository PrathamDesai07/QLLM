"""Data aggregation for QLLM — merge all experiment logs, features, and Pauli stats
into a unified analysis dataset (Parquet format).

Usage
-----
    python -m src.analysis.aggregate_results                  # full aggregation, includes LLM
    python -m src.analysis.aggregate_results --skip-llm       # skip LLM (no GPU needed)
    python -m src.analysis.aggregate_results --status          # print completeness only
"""

import json
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    GRAPHS_DIR,
    FEATURES_DIR,
    EXPERIMENTS_DIR,
    PCE_DIR,
    GRAPH_DEFAULTS,
)


ANALYSIS_DIR = DATA_DIR / "analysis"
RESULT_PATH = ANALYSIS_DIR / "qaoa_pce_results.parquet"

# Methods in our dataset
METHODS = ["baseline_qaoa", "pce_baseline_k1", "pce_baseline_k2", "llm_pce", "rule_pce"]


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_features(graph_id: str) -> dict[str, Any]:
    """Load the feature dict for *graph_id* from disk."""
    path = FEATURES_DIR / f"{graph_id}.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Pauli-string statistics
# ---------------------------------------------------------------------------

def _pauli_stats(pauli_strings: dict[str, str]) -> dict[str, Any]:
    """Compute Pauli statistics from a {variable: pauli_string} dict.

    Returns counts and proportions of X, Y, Z operators across all
    Pauli strings, plus the average number of non-identity operators
    per variable and the fraction of strings using each operator.
    """
    if not pauli_strings:
        return {
            "x_count": 0, "y_count": 0, "z_count": 0,
            "i_count": 0, "total_ops": 0,
            "x_frac": 0.0, "y_frac": 0.0, "z_frac": 0.0,
            "avg_non_id_per_var": 0.0,
            "max_non_id_per_var": 0,
            "num_pauli_strings": 0,
        }

    n_strings = len(pauli_strings)
    x_count = y_count = z_count = i_count = 0
    non_id_per_var: list[int] = []

    for ps in pauli_strings.values():
        xc = ps.count("X")
        yc = ps.count("Y")
        zc = ps.count("Z")
        ic = ps.count("I")
        x_count += xc
        y_count += yc
        z_count += zc
        i_count += ic
        non_id_per_var.append(xc + yc + zc)

    total_ops = x_count + y_count + z_count + i_count
    avg_non_id = float(np.mean(non_id_per_var)) if non_id_per_var else 0.0

    return {
        "x_count": x_count,
        "y_count": y_count,
        "z_count": z_count,
        "i_count": i_count,
        "total_ops": total_ops,
        "x_frac": round(x_count / total_ops, 4) if total_ops else 0.0,
        "y_frac": round(y_count / total_ops, 4) if total_ops else 0.0,
        "z_frac": round(z_count / total_ops, 4) if total_ops else 0.0,
        "avg_non_id_per_var": round(avg_non_id, 4),
        "max_non_id_per_var": max(non_id_per_var) if non_id_per_var else 0,
        "num_pauli_strings": n_strings,
    }


# ---------------------------------------------------------------------------
# Experiment record loader per method
# ---------------------------------------------------------------------------

def _load_experiment_json(graph_id: str, method: str) -> dict[str, Any] | None:
    """Load a single experiment JSON for *graph_id* and *method*."""
    if method == "baseline_qaoa":
        base_dir = EXPERIMENTS_DIR / "baseline_qaoa"
    elif method == "llm_pce":
        base_dir = EXPERIMENTS_DIR / "llm_pce"
    elif method == "pce_baseline_k1":
        base_dir = EXPERIMENTS_DIR / "pce_baseline_k1"
    elif method == "pce_baseline_k2":
        base_dir = EXPERIMENTS_DIR / "pce_baseline_k2"
    elif method == "rule_pce":
        base_dir = EXPERIMENTS_DIR / "rule_pce"
    else:
        return None

    path = base_dir / f"{graph_id}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_pce_encoding(graph_id: str, method: str) -> dict[str, Any] | None:
    """Load the PCE encoding spec from data/pce/ for Pauli statistics.

    For ``llm_pce``, encodings are under ``data/pce/llm/``.
    For manual PCE, they are under ``data/pce/`` (bare graph_id).
    """
    if method == "llm_pce":
        path = PCE_DIR / "llm" / f"{graph_id}.json"
    elif method in ("pce_baseline_k1", "pce_baseline_k2"):
        path = PCE_DIR / f"{graph_id}.json"
    elif method == "rule_pce":
        path = PCE_DIR / "rule" / f"{graph_id}.json"
    else:
        return None
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _build_row(graph_id: str, method: str) -> dict[str, Any] | None:
    """Build one unified row for a single (graph_id, method) pair.

    Returns None if the experiment record is missing.
    """
    record = _load_experiment_json(graph_id, method)
    if record is None:
        return None

    features = load_features(graph_id)
    extra = record.get("extra", {})
    params = record.get("params", {})

    # Determine k value
    if method == "baseline_qaoa":
        k = None
    elif method == "pce_baseline_k1":
        k = 1
    elif method == "pce_baseline_k2":
        k = 2
    else:
        k = extra.get("k")

    # Physical qubits and compression
    if method == "baseline_qaoa":
        num_physical_qubits = record.get("num_nodes")
        compression_ratio = 1.0
    else:
        num_physical_qubits = extra.get("num_physical_qubits")
        raw_cr = extra.get("compression_ratio")
        compression_ratio = float(raw_cr) if raw_cr is not None else None

    # Pauli statistics for PCE methods
    encoding = _load_pce_encoding(graph_id, method)
    if encoding and "pauli_strings" in encoding:
        ps = _pauli_stats(encoding["pauli_strings"])
    else:
        ps = _pauli_stats({})

    # LLM-specific fields
    llm_tags_raw = extra.get("llm_tags", [])
    if isinstance(llm_tags_raw, list):
        llm_tags = ",".join(llm_tags_raw)
    else:
        llm_tags = str(llm_tags_raw) if llm_tags_raw else ""

    llm_reasoning = extra.get("llm_reasoning", "")
    llm_model = extra.get("model_name", "")

    # Optimization efficiency
    nfeval = extra.get("nfeval")

    # Duration
    duration = record.get("duration_seconds")

    # Build row
    row: dict[str, Any] = {
        # Identity
        "graph_id": graph_id,
        "family": features.get("family", record.get("family", "unknown")),
        "num_nodes": features.get("num_nodes", record.get("num_nodes")),
        "num_edges": features.get("num_edges", record.get("num_edges")),

        # Method info
        "method": method,
        "k": k,
        "num_physical_qubits": num_physical_qubits,
        "compression_ratio": compression_ratio,

        # Graph features
        "density": features.get("density"),
        "degree_min": features.get("degree_min"),
        "degree_max": features.get("degree_max"),
        "degree_mean": features.get("degree_mean"),
        "degree_std": features.get("degree_std"),
        "avg_clustering": features.get("avg_clustering"),
        "transitivity": features.get("transitivity"),
        "num_components": features.get("num_components"),
        "largest_component_size": features.get("largest_component_size"),
        "component_size_ratio": features.get("component_size_ratio"),
        "algebraic_connectivity": features.get("algebraic_connectivity"),
        "modularity": features.get("modularity"),

        # Performance metrics
        "optimal_energy": record.get("optimal_energy"),
        "approximation_ratio": record.get("approximation_ratio"),
        "gradient_norm_at_opt": record.get("gradient_norm_at_opt"),
        "success": record.get("success"),
        "convergence_iters": record.get("convergence_iters"),
        "duration_seconds": duration,
        "nfeval": nfeval,

        # Pauli statistics
        **{f"pauli_{k}": v for k, v in ps.items()},

        # LLM metadata (empty for non-LLM methods)
        "llm_tags": llm_tags,
        "llm_model": llm_model,
        "llm_reasoning": llm_reasoning,
    }

    return row


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def _all_graph_ids() -> set[str]:
    """Return the set of every graph_id that appears in any experiment dir."""
    ids: set[str] = set()
    for method_dir_name in ["baseline_qaoa", "pce_baseline_k1", "pce_baseline_k2", "llm_pce", "rule_pce"]:
        d = EXPERIMENTS_DIR / method_dir_name
        if d.exists():
            for f in d.glob("*.json"):
                ids.add(f.stem)
    return ids


def aggregate(skip_llm: bool = False, verbose: bool = True) -> pd.DataFrame:
    """Aggregate all experiment results into a unified DataFrame.

    Parameters
    ----------
    skip_llm : bool
        Skip LLM-guided PCE results (useful if GPU unavailable).
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    pd.DataFrame with one row per (graph_id, method).
    """
    methods = [m for m in METHODS if not (skip_llm and m == "llm_pce")]

    if verbose:
        print(f"Aggregating experiment data over {len(methods)} methods …")

    all_graph_ids = sorted(_all_graph_ids())
    if verbose:
        print(f"  Found {len(all_graph_ids)} unique graph IDs.")

    rows: list[dict[str, Any]] = []
    counts = {m: 0 for m in methods}

    for gid in all_graph_ids:
        for method in methods:
            row = _build_row(gid, method)
            if row is not None:
                rows.append(row)
                counts[method] += 1

    df = pd.DataFrame(rows)

    # Cast types for cleaner data
    for col in ["num_nodes", "num_edges", "degree_min", "degree_max",
                "num_components", "largest_component_size", "nfeval",
                "num_physical_qubits", "pauli_x_count", "pauli_y_count",
                "pauli_z_count", "pauli_i_count", "pauli_total_ops",
                "pauli_num_pauli_strings", "pauli_max_non_id_per_var"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["compression_ratio", "density", "degree_mean", "degree_std",
                "avg_clustering", "transitivity", "component_size_ratio",
                "algebraic_connectivity", "modularity",
                "optimal_energy", "approximation_ratio", "gradient_norm_at_opt",
                "duration_seconds",
                "pauli_x_frac", "pauli_y_frac", "pauli_z_frac",
                "pauli_avg_non_id_per_var"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if verbose:
        print(f"  Aggregated {len(df)} total rows ({len(all_graph_ids)} graphs × {len(methods)} methods).")
        for m in methods:
            print(f"    {m}: {counts[m]} rows")

    return df


def save_aggregate(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save the aggregated DataFrame as Parquet + CSV."""
    out_path = path or RESULT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Parquet
    df.to_parquet(out_path, index=False)

    # CSV companion for easy inspection
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    print(f"  Wrote {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB, {len(df)} rows)")
    print(f"  Wrote {csv_path.name} ({csv_path.stat().st_size / 1024:.1f} KB)")

    return out_path


def status_report() -> None:
    """Print a completeness report for all experiment data."""
    all_ids = sorted(_all_graph_ids())

    print()
    print(f"  Aggregation Status")
    print(f"  {'─' * 40}")
    print(f"  Unique graph IDs:  {len(all_ids)}")

    for method in METHODS:
        if method == "baseline_qaoa":
            d = EXPERIMENTS_DIR / "baseline_qaoa"
        elif method == "llm_pce":
            d = EXPERIMENTS_DIR / "llm_pce"
        elif method == "pce_baseline_k1":
            d = EXPERIMENTS_DIR / "pce_baseline_k1"
        elif method == "pce_baseline_k2":
            d = EXPERIMENTS_DIR / "pce_baseline_k2"
        else:
            continue
        found = {f.stem for f in d.glob("*.json")} if d.exists() else set()
        n_found = len(found & set(all_ids))
        print(f"  {method:20s} {n_found}/{len(all_ids)}")

    # Feature files
    feat_ids = {f.stem for f in FEATURES_DIR.glob("*.json")} if FEATURES_DIR.exists() else set()
    print(f"  {'features':20s} {len(feat_ids & set(all_ids))}/{len(all_ids)}")

    # PCE encodings
    pce_ids = {f.stem for f in PCE_DIR.glob("*.json") if f.is_file()}
    llm_pce_dir = PCE_DIR / "llm"
    if llm_pce_dir.exists():
        pce_ids |= {f.stem for f in llm_pce_dir.glob("*.json")}
    print(f"  {'pce_encodings':20s} {len(pce_ids & set(all_ids))}/{len(all_ids)}")

    print()


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Aggregate QLLM experiment data into analysis dataset"
    )
    parser.add_argument("--status", action="store_true",
                        help="Print completeness report and exit")
    parser.add_argument("--skip-llm", action="store_true",
                        help="Skip LLM-guided PCE results (no GPU needed)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for Parquet file")

    args = parser.parse_args()

    if args.status:
        status_report()
    else:
        df = aggregate(skip_llm=args.skip_llm)
        save_aggregate(df, Path(args.output) if args.output else None)
