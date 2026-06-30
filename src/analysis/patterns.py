"""Pattern mining for QLLM — statistical & ML analysis to discover interpretable
structural patterns in the QAOA/PCE design space.

Mines three categories of patterns:

    1. Feature regime → preferred k (which k gives the best approximation ratio
       depending on graph features).
    2. Feature regime → Pauli layout that avoids vanishing gradients.
    3. Pauli distributions → higher gradient magnitudes / better approximation.

Usage
-----
    python -m src.analysis.patterns                        # mine & save patterns
    python -m src.analysis.patterns --no-llm              # skip LLM rows
    python -m src.analysis.patterns --min-samples 5       # min samples per split
"""

import json
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from config import DATA_DIR
from analysis.aggregate_results import aggregate, RESULT_PATH, load_features


ANALYSIS_DIR = DATA_DIR / "analysis"
PATTERNS_PATH = ANALYSIS_DIR / "patterns.json"


# ── Helper utilities ───────────────────────────────────────────────────

def _load_df() -> pd.DataFrame:
    """Load the aggregated dataset (or build it if missing)."""
    if RESULT_PATH.exists():
        return pd.read_parquet(RESULT_PATH)
    print("  Aggregated dataset not found — building it now.")
    df = aggregate()
    from analysis.aggregate_results import save_aggregate
    save_aggregate(df)
    return df


def _pearson_binned(df: pd.DataFrame, x_col: str, y_col: str,
                    bins: int = 5) -> dict[str, Any]:
    """Bin *x_col* into bins and compute mean ± std of *y_col* per bin.

    Returns a dict with bin_edges, bin_centers, means, stds, and counts.
    """
    if df[x_col].isnull().all() or df[y_col].isnull().all():
        return {"error": f"Column {x_col} or {y_col} is all NaN"}

    clean = df[[x_col, y_col]].dropna()
    if len(clean) < bins:
        return {"error": f"Too few samples ({len(clean)}) for {bins} bins"}

    clean["_bin"] = pd.cut(clean[x_col], bins=bins, labels=False)
    grouped = clean.groupby("_bin")[y_col]

    result = {
        "x_col": x_col,
        "y_col": y_col,
        "bins": bins,
        "bin_ranges": {},
    }
    for bin_id, group in grouped:
        bin_range = f"bin_{bin_id}"
        result["bin_ranges"][bin_range] = {
            "mean": round(float(group.mean()), 4),
            "std": round(float(group.std()), 4),
            "count": int(len(group)),
        }
    return result


def _best_k_per_bin(df: pd.DataFrame, group_col: str,
                    bins: int = 5) -> dict[str, Any]:
    """Bin *group_col* and find which k gives the best approximation ratio
    in each bin."""
    results: dict[str, Any] = {
        "group_col": group_col,
        "bins": bins,
        "bin_assignments": {},
    }

    pce_df = df[df["method"].isin(["pce_baseline_k1", "pce_baseline_k2",
                                    "llm_pce"])].copy()
    pce_df = pce_df[pce_df["k"].notna()].copy()

    if pce_df.empty or pce_df[group_col].isnull().all():
        return {"error": f"No valid data for {group_col}"}

    clean = pce_df[[group_col, "k", "approximation_ratio", "graph_id"]].dropna()
    if len(clean) < bins:
        return {"error": f"Too few samples ({len(clean)}) for {bins} bins"}

    clean["_bin"] = pd.cut(clean[group_col], bins=bins, labels=False)

    for bin_id, group in clean.groupby("_bin"):
        per_k = group.groupby("k")["approximation_ratio"].agg(["mean", "std", "count"])
        best_k = int(per_k["mean"].idxmax())
        results["bin_assignments"][str(bin_id)] = {
            "samples_in_bin": int(len(group)),
            "best_k": best_k,
            "per_k": {
                str(int(k)): {
                    "mean_approx": round(float(r["mean"]), 4),
                    "std_approx": round(float(r["std"]), 4),
                    "count": int(r["count"]),
                }
                for k, r in per_k.iterrows()
            },
        }

    return results


# ── Pattern 1: Feature regime → preferred k ───────────────────────────

def pattern_preferred_k(df: pd.DataFrame) -> dict[str, Any]:
    """For each feature, bin the feature and find which k gives the best
    mean approximation ratio in each bin."""
    features = ["density", "degree_mean", "degree_std", "avg_clustering",
                "transitivity", "algebraic_connectivity", "component_size_ratio"]

    results: dict[str, Any] = {}
    for feat in features:
        if feat not in df.columns:
            continue
        res = _best_k_per_bin(df, feat, bins=4)
        if "error" not in res:
            results[feat] = res

    # Also compute Spearman correlation between k and approximation ratio
    pce_df = df[df["method"].isin(["pce_baseline_k1", "pce_baseline_k2",
                                    "llm_pce"])].dropna(subset=["k", "approximation_ratio"])
    if len(pce_df) > 10:
        rho, p = spearmanr(pce_df["k"], pce_df["approximation_ratio"])
        results["spearman_k_vs_approx"] = {
            "rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n": int(len(pce_df)),
        }

    return results


# ── Pattern 2: Feature regime → gradient behaviour ────────────────────

def _flag_gradient_risk(grad_norm: float, threshold: float = 0.01) -> str:
    """Classify gradient magnitude."""
    if grad_norm < threshold:
        return "vanishing"
    return "healthy"


def pattern_gradient_risk(df: pd.DataFrame) -> dict[str, Any]:
    """Analyse which feature regimes correlate with vanishing / healthy
    gradients."""
    pce_df = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["gradient_norm_at_opt"]
    ).copy()

    if pce_df.empty:
        return {"error": "No PCE data with gradient info"}

    pce_df["gradient_risk"] = pce_df["gradient_norm_at_opt"].apply(
        lambda x: _flag_gradient_risk(x, threshold=0.01)
    )

    results: dict[str, Any] = {
        "gradient_threshold": 0.01,
        "vanishing_count": int((pce_df["gradient_risk"] == "vanishing").sum()),
        "healthy_count": int((pce_df["gradient_risk"] == "healthy").sum()),
        "feature_correlations": {},
    }

    features = ["density", "degree_mean", "degree_std", "avg_clustering",
                "transitivity", "algebraic_connectivity", "num_nodes",
                "component_size_ratio"]

    for feat in features:
        if feat not in pce_df.columns:
            continue
        clean = pce_df[[feat, "gradient_norm_at_opt"]].dropna()
        if len(clean) < 10:
            continue
        rho, p = spearmanr(clean[feat], clean["gradient_norm_at_opt"])
        results["feature_correlations"][feat] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n": int(len(clean)),
        }

    # Gradient risk per method
    method_risk_dict: dict[str, dict] = {}
    for method, srs in pce_df.groupby("method"):
        vanishing = int((srs["gradient_risk"] == "vanishing").sum())
        healthy = int((srs["gradient_risk"] == "healthy").sum())
        method_risk_dict[str(method)] = {"vanishing": vanishing, "healthy": healthy}
    results["risk_per_method"] = method_risk_dict

    # Gradient risk per family
    family_risk_dict: dict[str, dict] = {}
    for family, srs in pce_df.groupby("family"):
        vanishing = int((srs["gradient_risk"] == "vanishing").sum())
        healthy = int((srs["gradient_risk"] == "healthy").sum())
        family_risk_dict[str(family)] = {"vanishing": vanishing, "healthy": healthy}
    results["risk_per_family"] = family_risk_dict

    # Gradient risk by k
    k_risk_dict: dict[str, dict] = {}
    for k_val, srs in pce_df[pce_df["k"].notna()].groupby("k"):
        vanishing = int((srs["gradient_risk"] == "vanishing").sum())
        healthy = int((srs["gradient_risk"] == "healthy").sum())
        k_risk_dict[str(int(k_val))] = {"vanishing": vanishing, "healthy": healthy}
    results["risk_per_k"] = k_risk_dict

    return results


# ── Pattern 3: Pauli distribution → performance ───────────────────────

def pattern_pauli_layout(df: pd.DataFrame) -> dict[str, Any]:
    """Analyse how Pauli operator fractions (X, Y, Z) and average
    non-identity operators per variable correlate with approximation ratio
    and gradient norm."""
    pce_df = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["approximation_ratio", "pauli_x_frac", "pauli_y_frac",
                 "pauli_z_frac", "pauli_avg_non_id_per_var"]
    ).copy()

    if pce_df.empty:
        return {"error": "No PCE data with Pauli stats"}

    results: dict[str, Any] = {}

    # Correlate Pauli fractions with approximation ratio
    pauli_feats = ["pauli_x_frac", "pauli_y_frac", "pauli_z_frac",
                   "pauli_avg_non_id_per_var"]
    for feat in pauli_feats:
        if feat not in pce_df.columns:
            continue
        clean = pce_df[[feat, "approximation_ratio"]].dropna()
        if len(clean) < 10:
            continue
        rho, p = spearmanr(clean[feat], clean["approximation_ratio"])
        results[f"{feat}_vs_approx"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n": int(len(clean)),
            "interpretation": (
                "X-dominant layouts correlate with higher approximation quality"
                if feat == "pauli_x_frac" and rho > 0.1 else
                "Z-dominant layouts correlate with higher approximation quality"
                if feat == "pauli_z_frac" and rho > 0.1 else
                "No strong correlation"
            ),
        }

    # Correlate Pauli fractions with gradient norm
    for feat in pauli_feats:
        if feat not in pce_df.columns:
            continue
        clean = pce_df[[feat, "gradient_norm_at_opt"]].dropna()
        if len(clean) < 10:
            continue
        rho, p = spearmanr(clean[feat], clean["gradient_norm_at_opt"])
        results[f"{feat}_vs_gradient"] = {
            "spearman_rho": round(float(rho), 4),
            "p_value": round(float(p), 6),
            "n": int(len(clean)),
            "interpretation": (
                "Fewer non-identity ops correlates with vanishing gradients"
                if feat == "pauli_avg_non_id_per_var" and abs(rho) > 0.1 else
                "No strong correlation"
            ),
        }

    # Pauli distribution per best-k group
    pce_with_best = pce_df.copy()
    best_k_per_graph: dict[str, int] = {}
    for gid, group in pce_with_best.groupby("graph_id"):
        k_group = group.dropna(subset=["k", "approximation_ratio"])
        if k_group.empty:
            continue
        best = k_group.loc[k_group["approximation_ratio"].idxmax()]
        best_k_per_graph[gid] = int(best["k"])

    pce_with_best["best_k"] = pce_with_best["graph_id"].map(best_k_per_graph)
    # Only keep rows where k matches best_k
    pce_best = pce_with_best[
        pce_with_best["k"].notna() &
        (pce_with_best["k"].astype(int) == pce_with_best["best_k"])
    ].copy()

    if not pce_best.empty:
        pauli_dist: dict[str, dict] = {}
        for k_val, grp in pce_best.groupby("k"):
            k_key = str(int(k_val))
            pauli_dist[k_key] = {
                "mean_x_frac": round(float(grp["pauli_x_frac"].mean()), 4),
                "mean_y_frac": round(float(grp["pauli_y_frac"].mean()), 4),
                "mean_z_frac": round(float(grp["pauli_z_frac"].mean()), 4),
                "mean_avg_non_id": round(float(grp["pauli_avg_non_id_per_var"].mean()), 4),
            }
        results["pauli_distribution_per_best_k"] = pauli_dist

    return results


# ── Pattern 4: Compression ratio vs performance ──────────────────────

def pattern_compression_vs_performance(df: pd.DataFrame) -> dict[str, Any]:
    """Analyse how compression ratio trades off against approximation
    ratio and gradient behaviour."""
    pce_df = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["compression_ratio", "approximation_ratio"]
    ).copy()

    if pce_df.empty:
        return {"error": "No data"}

    results: dict[str, Any] = {}

    # Spearman correlation
    rho, p = spearmanr(pce_df["compression_ratio"], pce_df["approximation_ratio"])
    results["compression_vs_approx"] = {
        "spearman_rho": round(float(rho), 4),
        "p_value": round(float(p), 6),
        "n": int(len(pce_df)),
    }

    rho2, p2 = spearmanr(
        pce_df["compression_ratio"].dropna(),
        pce_df.loc[pce_df["compression_ratio"].notna(), "gradient_norm_at_opt"]
    )
    results["compression_vs_gradient"] = {
        "spearman_rho": round(float(rho2), 4),
        "p_value": round(float(p2), 6),
        "n": int(pce_df["compression_ratio"].notna().sum()),
    }

    # Binned analysis
    results["approx_by_compression_bins"] = _pearson_binned(
        pce_df, "compression_ratio", "approximation_ratio", bins=4
    )

    return results


# ── Pattern 5: Family-level summary ───────────────────────────────────

def pattern_family_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Per-family summary of approximation ratio, gradient, and best method."""
    results: dict[str, Any] = {}

    for family, group in df.groupby("family"):
        per_method = group.groupby("method").agg(
            mean_approx=("approximation_ratio", "mean"),
            std_approx=("approximation_ratio", "std"),
            mean_grad=("gradient_norm_at_opt", "mean"),
            n=("graph_id", "count"),
        ).to_dict("index")

        best_method = max(
            per_method, key=lambda m: per_method[m]["mean_approx"]
        )

        results[family] = {
            "per_method": {
                m: {
                    "mean_approximation_ratio": round(v["mean_approx"], 4),
                    "std_approximation_ratio": round(v["std_approx"], 4),
                    "mean_gradient_norm": round(v["mean_grad"], 4),
                    "n": int(v["n"]),
                }
                for m, v in per_method.items()
            },
            "best_method": best_method,
        }

    return results


# ── Master pattern miner ──────────────────────────────────────────────

def mine_patterns(df: pd.DataFrame | None = None,
                  verbose: bool = True) -> dict[str, Any]:
    """Run all pattern miners on the aggregated dataset.

    Parameters
    ----------
    df : pd.DataFrame | None
        Pre-loaded aggregated DataFrame. If None, loads it.
    verbose : bool
        Print progress.

    Returns
    -------
    dict containing all discovered patterns.
    """
    if df is None:
        df = _load_df()

    patterns: dict[str, Any] = {}

    if verbose:
        print("Mining patterns …")

    # Pattern 1: Feature → preferred k
    if verbose:
        print("  Pattern 1: Feature regime → preferred k")
    patterns["feature_to_preferred_k"] = pattern_preferred_k(df)

    # Pattern 2: Feature → gradient risk
    if verbose:
        print("  Pattern 2: Feature regime → gradient health")
    patterns["feature_to_gradient_risk"] = pattern_gradient_risk(df)

    # Pattern 3: Pauli layout → performance
    if verbose:
        print("  Pattern 3: Pauli layout → performance")
    patterns["pauli_layout_to_performance"] = pattern_pauli_layout(df)

    # Pattern 4: Compression → performance
    if verbose:
        print("  Pattern 4: Compression → performance")
    patterns["compression_vs_performance"] = pattern_compression_vs_performance(df)

    # Pattern 5: Family-level summary
    if verbose:
        print("  Pattern 5: Family-level summary")
    patterns["family_summary"] = pattern_family_summary(df)

    if verbose:
        # Print highlights
        print("\n  Pattern highlights:")
        for k, v in patterns.get("feature_to_preferred_k", {}).items():
            if k == "spearman_k_vs_approx":
                print(f"    Spearman(k, approx_ratio) = {v.get('rho')} (p={v.get('p_value')})")
            elif isinstance(v, dict) and "bin_assignments" in v:
                print(f"    {k}: {len(v['bin_assignments'])} bins analysed")

        grad_m = patterns.get("feature_to_gradient_risk", {})
        if isinstance(grad_m, dict) and "vanishing_count" in grad_m:
            total = grad_m.get("vanishing_count", 0) + grad_m.get("healthy_count", 0)
            print(f"    Gradient: {grad_m.get('vanishing_count')}/{total} vanishing, "
                  f"{grad_m.get('healthy_count')}/{total} healthy")

        for fam, info in patterns.get("family_summary", {}).items():
            if isinstance(info, dict) and "best_method" in info:
                print(f"    {fam}: best method = {info['best_method']}")

    return patterns


def save_patterns(patterns: dict[str, Any],
                  path: Path | None = None) -> Path:
    """Save patterns to JSON."""
    out_path = path or PATTERNS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(patterns, f, indent=2)
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Mine patterns from QLLM aggregated experiment data"
    )
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM rows in analysis")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for patterns JSON")

    args = parser.parse_args()

    df = _load_df()
    if args.no_llm:
        df = df[df["method"] != "llm_pce"]

    patterns = mine_patterns(df)
    save_patterns(patterns, Path(args.output) if args.output else None)
