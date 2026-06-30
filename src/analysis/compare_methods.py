"""Cross-method comparison — compare approximation ratios, compression, and
gradient behaviour across all methods: baseline QAOA, PCE k=1, PCE k=2,
LLM-guided PCE, and rule-based PCE.

Usage
-----
    python -m src.analysis.compare_methods                     # full comparison
    python -m src.analysis.compare_methods --run-rule-pce      # also run rule PCE first
    python -m src.analysis.compare_methods --max-graphs 10     # limit for testing
    python -m src.analysis.compare_methods --plot              # generate comparison plots
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

from config import DATA_DIR, EXPERIMENTS_DIR, GRAPHS_DIR, FEATURES_DIR
from analysis.aggregate_results import (
    aggregate, save_aggregate, RESULT_PATH, _load_experiment_json,
    load_features, _pauli_stats, _load_pce_encoding, _build_row,
)
from pipeline.rule_based_pce import run_rule_pce_batch, RULE_PCE_DIR


ANALYSIS_DIR = DATA_DIR / "analysis"
COMPARISON_PATH = ANALYSIS_DIR / "method_comparison.json"

ALL_METHODS = [
    "baseline_qaoa",
    "pce_baseline_k1",
    "pce_baseline_k2",
    "llm_pce",
    "rule_pce",
]


# ── Ensure rule PCE results exist ─────────────────────────────────────

def ensure_rule_pce_results(
    families: list[str] | None = None,
    max_graphs: int | None = None,
    verbose: bool = True,
) -> set[str]:
    """Run the rule-based PCE pipeline for all graphs that don't have results yet.

    Returns the set of graph_ids with rule_pce results.
    """
    existing = {f.stem for f in RULE_PCE_DIR.glob("*.json")}
    all_graphs = {f.stem for f in GRAPHS_DIR.glob("*.json")}
    needed = all_graphs - existing

    if not needed:
        if verbose:
            print(f"  All {len(all_graphs)} graphs already have rule_pce results.")
        return existing | all_graphs

    if verbose:
        print(f"  Running rule_based_pce on {len(needed)} graphs …")

    results = run_rule_pce_batch(
        families=families,
        max_graphs=max_graphs,
        skip_existing=True,
        verbose=False,
    )

    done = {f.stem for f in RULE_PCE_DIR.glob("*.json")}
    if verbose:
        print(f"  Completed {len(done - existing)} new rule_pce runs.")

    return done


# ── Build unified comparison dataset ──────────────────────────────────

def build_comparison_dataset(
    run_rule_pce_first: bool = False,
    families: list[str] | None = None,
    max_graphs: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Build a unified DataFrame with all methods including rule_pce.

    If *run_rule_pce_first* is True, run the rule-based pipeline
    for any missing graphs first.

    Returns a DataFrame with one row per (graph_id, method).
    """
    # Ensure rule_pce results exist
    if run_rule_pce_first:
        ensure_rule_pce_results(families=families, max_graphs=max_graphs,
                                verbose=verbose)

    all_ids = set()
    for method in ["baseline_qaoa", "pce_baseline_k1", "pce_baseline_k2",
                    "llm_pce", "rule_pce"]:
        d = EXPERIMENTS_DIR / method
        if d.exists():
            all_ids |= {f.stem for f in d.glob("*.json")}

    all_ids = sorted(all_ids)
    if verbose:
        print(f"  Building comparison dataset from {len(all_ids)} graph IDs …")

    rows: list[dict[str, Any]] = []
    counts: dict[str, int] = {m: 0 for m in ALL_METHODS}

    for gid in all_ids:
        for method in ALL_METHODS:
            row = _build_row(gid, method)
            if row is not None:
                rows.append(row)
                counts[method] += 1

    df = pd.DataFrame(rows)
    if verbose:
        print(f"  Total rows: {len(df)}")
        for m in ALL_METHODS:
            print(f"    {m}: {counts[m]}")

    return df


# ── Comparison analysis ──────────────────────────────────────────────

def method_comparison(
    df: pd.DataFrame | None = None,
    run_rule_pce_first: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """Compare all methods across approximation ratio, gradient, and compression.

    Returns a dict with sections:
        method_summary, pairwise_improvements, gradient_analysis,
        compression_analysis, family_breakdown, rule_pce_vs_llm.
    """
    if df is None:
        df = build_comparison_dataset(run_rule_pce_first=run_rule_pce_first,
                                      verbose=verbose)

    results: dict[str, Any] = {}

    # ── Method summary ──────────────────────────────────────────────
    summary = df.groupby("method").agg(
        mean_approx=("approximation_ratio", "mean"),
        std_approx=("approximation_ratio", "std"),
        median_approx=("approximation_ratio", "median"),
        mean_grad=("gradient_norm_at_opt", "mean"),
        median_grad=("gradient_norm_at_opt", "median"),
        mean_compression=("compression_ratio", "mean"),
        count=("graph_id", "count"),
    ).to_dict("index")

    results["method_summary"] = {
        m: {
            "mean_approximation_ratio": round(v["mean_approx"], 4),
            "std_approximation_ratio": round(v["std_approx"], 4),
            "median_approximation_ratio": round(v["median_approx"], 4),
            "mean_gradient_norm": round(v["mean_grad"], 6),
            "median_gradient_norm": round(v["median_grad"], 6),
            "mean_compression_ratio": round(v["mean_compression"], 4),
            "n": int(v["count"]),
        }
        for m, v in summary.items()
    }

    # ── Best method per graph ───────────────────────────────────────
    best_method_per_graph: dict[str, str] = {}
    for gid, group in df.groupby("graph_id"):
        best = group.loc[group["approximation_ratio"].idxmax()]
        best_method_per_graph[gid] = best["method"]

    method_win_rates: dict[str, int] = {}
    for m in ALL_METHODS:
        method_win_rates[m] = sum(1 for v in best_method_per_graph.values()
                                   if v == m)

    results["best_method_win_rates"] = {
        m: {"n_wins": v,
            "pct": round(v / max(len(best_method_per_graph), 1) * 100, 1)}
        for m, v in sorted(method_win_rates.items(), key=lambda x: -x[1])
    }

    # ── Pairwise improvements ───────────────────────────────────────
    pairwise = {}
    for m1 in ALL_METHODS:
        for m2 in ALL_METHODS:
            if m1 >= m2:
                continue
            merged = df[df["method"] == m1].merge(
                df[df["method"] == m2],
                on="graph_id", suffixes=("_1", "_2")
            )
            if merged.empty:
                continue
            delta = merged["approximation_ratio_1"] - merged["approximation_ratio_2"]
            pairwise[f"{m1}_vs_{m2}"] = {
                "n": int(len(delta)),
                "mean_delta": round(float(delta.mean()), 4),
                "std_delta": round(float(delta.std()), 4),
                "median_delta": round(float(delta.median()), 4),
                "pct_improvement": round(
                    (delta > 0).sum() / len(delta) * 100, 1
                ),
            }
    results["pairwise_comparison"] = pairwise

    # ── Gradient analysis ───────────────────────────────────────────
    grad_df = df.dropna(subset=["gradient_norm_at_opt"])

    grad_by_method = {}
    for method, group in grad_df.groupby("method"):
        vanishing = int((group["gradient_norm_at_opt"] < 0.01).sum())
        total = len(group)
        grad_by_method[method] = {
            "total": total,
            "vanishing": vanishing,
            "healthy": total - vanishing,
            "vanishing_pct": round(vanishing / total * 100, 1) if total else 0,
        }
    results["gradient_analysis"] = {
        "threshold": 0.01,
        "per_method": grad_by_method,
    }

    # ── Compression analysis ────────────────────────────────────────
    comp_df = df.dropna(subset=["compression_ratio"])
    comp_by_method = {}
    for method, group in comp_df.groupby("method"):
        if method == "baseline_qaoa":
            continue
        comp_by_method[method] = {
            "mean_compression": round(float(group["compression_ratio"].mean()), 4),
            "min_compression": round(float(group["compression_ratio"].min()), 4),
            "max_compression": round(float(group["compression_ratio"].max()), 4),
            "median_compression": round(float(group["compression_ratio"].median()), 4),
        }
    results["compression_analysis"] = comp_by_method

    # ── Family breakdown ────────────────────────────────────────────
    family_breakdown = {}
    for family, fgroup in df.groupby("family"):
        method_summary = fgroup.groupby("method").agg(
            mean_approx=("approximation_ratio", "mean"),
            std_approx=("approximation_ratio", "std"),
            n=("graph_id", "count"),
        ).to_dict("index")

        best_method = max(method_summary,
                          key=lambda m: method_summary[m]["mean_approx"])

        family_breakdown[family] = {
            "per_method": {
                m: {
                    "mean_approximation_ratio": round(v["mean_approx"], 4),
                    "std_approximation_ratio": round(v["std_approx"], 4),
                    "n": int(v["n"]),
                }
                for m, v in method_summary.items()
            },
            "best_method": best_method,
        }
    results["family_breakdown"] = family_breakdown

    # ── Rule PCE vs LLM comparison ──────────────────────────────────
    if "rule_pce" in df["method"].values and "llm_pce" in df["method"].values:
        rule_llm = df[df["method"].isin(["rule_pce", "llm_pce"])].pivot_table(
            index="graph_id", columns="method",
            values=["approximation_ratio", "gradient_norm_at_opt",
                    "compression_ratio"],
        )
        if not rule_llm.empty and "approximation_ratio" in rule_llm.columns.levels[0]:
            approx_diff = (
                rule_llm["approximation_ratio"]["rule_pce"]
                - rule_llm["approximation_ratio"]["llm_pce"]
            ).dropna()

            grad_diff = (
                rule_llm["gradient_norm_at_opt"]["rule_pce"]
                - rule_llm["gradient_norm_at_opt"]["llm_pce"]
            ).dropna()

            results["rule_pce_vs_llm"] = {
                "approx_ratio_delta": {
                    "mean": round(float(approx_diff.mean()), 4),
                    "std": round(float(approx_diff.std()), 4),
                    "median": round(float(approx_diff.median()), 4),
                    "rule_wins": int((approx_diff > 0).sum()),
                    "llm_wins": int((approx_diff < 0).sum()),
                    "n": int(len(approx_diff)),
                },
                "gradient_norm_delta": {
                    "mean": round(float(grad_diff.mean()), 6),
                    "std": round(float(grad_diff.std()), 6),
                    "median": round(float(grad_diff.median()), 6),
                    "n": int(len(grad_diff)),
                },
            }

    return results


def save_comparison(comparison: dict[str, Any],
                    path: Path | None = None) -> Path:
    """Save comparison results to JSON."""
    out_path = path or COMPARISON_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"  Wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def print_comparison_summary(comparison: dict[str, Any]) -> None:
    """Print a human-readable summary of the comparison."""
    print("\n  Method Comparison Summary")
    print(f"  {'─' * 60}")

    ms = comparison.get("method_summary", {})
    if ms:
        print(f"  {'Method':25s} {'Approx':>8s} {'Grad':>10s} {'Comp':>8s} {'N':>5s}")
        print(f"  {'─' * 60}")
        for method in ALL_METHODS:
            if method in ms:
                v = ms[method]
                print(f"  {method:25s} {v['mean_approximation_ratio']:>8.4f} "
                      f"{v['mean_gradient_norm']:>10.6f} "
                      f"{v['mean_compression_ratio']:>8.2f} "
                      f"{v['n']:>5d}")

    wr = comparison.get("best_method_win_rates", {})
    if wr:
        print(f"\n  Best method win rates:")
        for m, v in wr.items():
            print(f"    {m:25s}: {v['n_wins']:3d}/{sum(x['n_wins'] for x in wr.values())} "
                  f"({v['pct']:.1f}%)")

    grad = comparison.get("gradient_analysis", {}).get("per_method", {})
    if grad:
        print(f"\n  Vanishing gradient rates (grad < 0.01):")
        for m, v in grad.items():
            print(f"    {m:25s}: {v['vanishing']:3d}/{v['total']} "
                  f"({v['vanishing_pct']:.1f}%)")

    rvlm = comparison.get("rule_pce_vs_llm", {})
    if rvlm:
        print(f"\n  Rule PCE vs LLM PCE (delta = rule − llm):")
        print(f"    Approx delta:  mean={rvlm['approx_ratio_delta']['mean']:.4f} "
              f"rule_wins={rvlm['approx_ratio_delta']['rule_wins']}/"
              f"{rvlm['approx_ratio_delta']['n']}")
        print(f"    Grad delta:    mean={rvlm['gradient_norm_delta']['mean']:.6f}")

    print()


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Cross-method comparison for QLLM"
    )
    parser.add_argument("--run-rule-pce", action="store_true",
                        help="Run rule-based PCE for missing graphs first")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Limit graphs for testing")
    parser.add_argument("--plot", action="store_true",
                        help="TODO: generate comparison plots")
    parser.add_argument("--output", type=str, default=None)

    args = parser.parse_args()

    comparison = method_comparison(
        run_rule_pce_first=args.run_rule_pce,
        verbose=True,
    )

    save_comparison(comparison, Path(args.output) if args.output else None)
    print_comparison_summary(comparison)
