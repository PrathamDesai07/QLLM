"""Visualization for QLLM — plots linking graph features, correlation order,
Pauli layouts, and performance metrics.

Produces figures in ``output/plots/``.

Usage
-----
    python -m src.analysis.plots                          # generate all plots
    python -m src.analysis.plots --no-show               # save only (no display)
    python -m src.analysis.plots --format png            # output format
"""

import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import OUTPUT_DIR, PLOTS_DIR
from analysis.aggregate_results import RESULT_PATH, load_features


sns.set_theme(style="whitegrid", palette="muted",
              font_scale=0.9)

# Move legend outside
LEGEND_KWARGS = {"bbox_to_anchor": (1.02, 1), "loc": "upper left"}

# Methods and their display names/colors
METHOD_PALETTE = {
    "baseline_qaoa": ("#6b7280", "Baseline QAOA"),
    "pce_baseline_k1": ("#3b82f6", "PCE k=1"),
    "pce_baseline_k2": ("#f59e0b", "PCE k=2"),
    "llm_pce": ("#10b981", "LLM PCE"),
}


# ── Data loading ───────────────────────────────────────────────────────

def _load() -> pd.DataFrame:
    if not RESULT_PATH.exists():
        raise FileNotFoundError(
            f"Aggregated dataset not found at {RESULT_PATH}. "
            "Run python -m src.analysis.aggregate_results first."
        )
    df = pd.read_parquet(RESULT_PATH)
    df["method_label"] = df["method"].map(
        {k: v[1] for k, v in METHOD_PALETTE.items()}
    )
    return df


# ── Plot 1: Approximation ratio by method and family ─────────────────

def plot_approx_by_method_family(df: pd.DataFrame,
                                  save_dir: Path) -> Path:
    """Box plot: approximation ratio distribution per method, faceted by
    graph family."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

    for ax, (family, group) in zip(axes, df.groupby("family")):
        sns.boxplot(
            data=group, x="method_label", y="approximation_ratio",
            palette=[METHOD_PALETTE[m][0] for m in group["method"].unique()
                     if m in METHOD_PALETTE],
            ax=ax,
        )
        ax.set_title(family.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Approximation Ratio" if ax == axes[0] else "")
        ax.tick_params(axis="x", rotation=30)

    fig.suptitle("Approximation Ratio by Method and Graph Family",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_dir / "approx_by_method_family.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 2: Feature bins → best k heatmap ────────────────────────────

def plot_feature_k_heatmap(patterns: dict[str, Any],
                           save_dir: Path) -> Path:
    """Heatmap showing which k is best for each feature bin."""
    feat_data = patterns.get("feature_to_preferred_k", {})
    if not feat_data:
        raise ValueError("No feature_to_preferred_k data in patterns")

    rows = []
    for feat_name, feat_info in feat_data.items():
        if not isinstance(feat_info, dict) or "bin_assignments" not in feat_info:
            continue
        for bin_key, bin_info in feat_info["bin_assignments"].items():
            rows.append({
                "feature": feat_name.replace("_", " ").title(),
                "bin": int(bin_key),
                "best_k": bin_info["best_k"],
            })

    if not rows:
        return _write_empty_plot(save_dir, "feature_k_heatmap",
                                 "No feature-to-k data")

    hm_df = pd.DataFrame(rows).pivot(
        index="feature", columns="bin", values="best_k"
    )

    fig, ax = plt.subplots(figsize=(8, max(4, len(hm_df) * 0.5)))
    sns.heatmap(hm_df, annot=True, fmt="d", cmap="YlOrRd",
                linewidths=0.5, cbar_kws={"label": "Best k"},
                ax=ax, vmin=1, vmax=2)
    ax.set_title("Preferred Correlation Order (k) by Feature Bin",
                 fontsize=12)
    ax.set_xlabel("Bin (low → high feature value)")
    ax.set_ylabel("")
    fig.tight_layout()
    path = save_dir / "feature_k_heatmap.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 3: Compression ratio vs approximation ratio ─────────────────

def plot_compression_vs_approx(df: pd.DataFrame,
                                save_dir: Path) -> Path:
    """Scatter plot: compression ratio vs approximation ratio, colored by
    method, sized by graph size."""
    pce = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["compression_ratio", "approximation_ratio"]
    ).copy()

    if pce.empty:
        return _write_empty_plot(save_dir, "compression_vs_approx",
                                 "No PCE data")

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ["pce_baseline_k1", "pce_baseline_k2", "llm_pce"]:
        group = pce[pce["method"] == method]
        if group.empty:
            continue
        color, label = METHOD_PALETTE[method]
        ax.scatter(
            group["compression_ratio"], group["approximation_ratio"],
            c=color, label=label, alpha=0.5, s=30,
            edgecolors="none",
        )

    ax.set_xlabel("Compression Ratio (n / qubits)")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Compression vs Approximation Quality", fontsize=12)
    ax.legend(**LEGEND_KWARGS)
    fig.tight_layout()
    path = save_dir / "compression_vs_approx.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 4: Gradient norm vs approximation ratio ─────────────────────

def plot_gradient_vs_approx(df: pd.DataFrame, save_dir: Path) -> Path:
    """Scatter plot: gradient norm at optimum vs approximation ratio."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # All methods
    ax = axes[0]
    for method in ["baseline_qaoa", "pce_baseline_k1", "pce_baseline_k2",
                    "llm_pce"]:
        group = df[df["method"] == method].dropna(
            subset=["gradient_norm_at_opt", "approximation_ratio"]
        )
        if group.empty:
            continue
        color, label = METHOD_PALETTE[method]
        ax.scatter(
            group["gradient_norm_at_opt"], group["approximation_ratio"],
            c=color, label=label, alpha=0.5, s=25,
            edgecolors="none",
        )
    ax.axvline(0.01, color="red", linestyle="--", alpha=0.4,
               label="Vanishing threshold")
    ax.set_xlabel("Gradient Norm at Optimum")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("All Methods", fontsize=11)
    ax.legend(fontsize=8)

    # Zoom on low-gradient region
    ax = axes[1]
    low_grad = df[df["gradient_norm_at_opt"] < 0.1].dropna(
        subset=["gradient_norm_at_opt", "approximation_ratio"]
    )
    for method in ["baseline_qaoa", "pce_baseline_k1", "pce_baseline_k2",
                    "llm_pce"]:
        group = low_grad[low_grad["method"] == method]
        if group.empty:
            continue
        color, label = METHOD_PALETTE[method]
        ax.scatter(
            group["gradient_norm_at_opt"], group["approximation_ratio"],
            c=color, label=label, alpha=0.5, s=25,
            edgecolors="none",
        )
    ax.axvline(0.01, color="red", linestyle="--", alpha=0.4)
    ax.set_xlabel("Gradient Norm at Optimum")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Zoom: gradient norm < 0.1", fontsize=11)
    ax.legend(fontsize=8)

    fig.suptitle("Gradient Behaviour vs Approximation Quality",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = save_dir / "gradient_vs_approx.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 5: Pauli operator fractions per method ──────────────────────

def plot_pauli_fractions(df: pd.DataFrame, save_dir: Path) -> Path:
    """Stacked bar chart: mean X/Y/Z fractions per method."""
    pce = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["pauli_x_frac", "pauli_y_frac", "pauli_z_frac"]
    ).copy()

    if pce.empty:
        return _write_empty_plot(save_dir, "pauli_fractions",
                                 "No Pauli data")

    method_order = ["pce_baseline_k1", "llm_pce", "pce_baseline_k2"]
    means = pce.groupby("method")[
        ["pauli_x_frac", "pauli_y_frac", "pauli_z_frac"]
    ].mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(method_order))
    width = 0.6
    colors = ["#ef4444", "#22c55e", "#3b82f6"]
    labels = ["X fraction", "Y fraction", "Z fraction"]

    bottom = np.zeros(len(method_order))
    for i, (col, color, label) in enumerate(
            zip(["pauli_x_frac", "pauli_y_frac", "pauli_z_frac"],
                colors, labels)):
        vals = [means.loc[m][col] if m in means.index else 0
                for m in method_order]
        ax.bar(x, vals, width, bottom=bottom, color=color, label=label,
               alpha=0.85)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([METHOD_PALETTE[m][1] for m in method_order])
    ax.set_ylabel("Mean Fraction")
    ax.set_title("Pauli Operator Distribution by Encoding Method",
                 fontsize=12)
    ax.legend(**LEGEND_KWARGS)
    fig.tight_layout()
    path = save_dir / "pauli_fractions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 6: Feature correlations heatmap ──────────────────────────────

def plot_feature_correlation_heatmap(df: pd.DataFrame,
                                      save_dir: Path) -> Path:
    """Heatmap: Spearman correlations between graph features and
    performance metrics."""
    feat_cols = ["density", "degree_mean", "degree_std", "avg_clustering",
                 "transitivity", "modularity", "algebraic_connectivity",
                 "component_size_ratio", "num_nodes"]
    metric_cols = ["approximation_ratio", "gradient_norm_at_opt",
                   "compression_ratio"]
    present_feats = [c for c in feat_cols if c in df.columns]
    present_metrics = [c for c in metric_cols if c in df.columns]

    if not present_feats or not present_metrics:
        return _write_empty_plot(save_dir, "feature_correlations",
                                 "Missing feature/metric columns")

    corr = df[present_feats + present_metrics].corr(method="spearman")
    corr_subset = corr.loc[present_feats, present_metrics]

    fig, ax = plt.subplots(figsize=(6, max(4, len(present_feats) * 0.45)))
    sns.heatmap(
        corr_subset, annot=True, fmt=".3f", cmap="RdBu_r",
        center=0, linewidths=0.5, vmin=-0.5, vmax=0.5,
        cbar_kws={"label": "Spearman ρ"}, ax=ax,
    )
    ax.set_title("Feature vs Performance Correlations", fontsize=12)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    path = save_dir / "feature_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 7: Gradient risk per family ──────────────────────────────────

def plot_gradient_risk_by_family(patterns: dict[str, Any],
                                  save_dir: Path) -> Path:
    """Stacked bar chart: vanishing vs healthy gradient per family."""
    risk_data = patterns.get("feature_to_gradient_risk", {})
    if not isinstance(risk_data, dict) or "risk_per_family" not in risk_data:
        return _write_empty_plot(save_dir, "gradient_risk_by_family",
                                 "No gradient risk data")

    families = list(risk_data["risk_per_family"].keys())
    vanishing = [risk_data["risk_per_family"][f]["vanishing"] for f in families]
    healthy = [risk_data["risk_per_family"][f]["healthy"] for f in families]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(families))
    width = 0.5

    ax.bar(x, healthy, width, label="Healthy (grad ≥ 0.01)", color="#22c55e",
           alpha=0.85)
    ax.bar(x, vanishing, width, bottom=healthy,
           label="Vanishing (grad < 0.01)", color="#ef4444", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", " ").title() for f in families])
    ax.set_ylabel("Number of PCE Experiments")
    ax.set_title("Gradient Health by Graph Family", fontsize=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = save_dir / "gradient_risk_by_family.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 8: Method comparison — approximation ratio box ──────────────

def plot_method_comparison(df: pd.DataFrame, save_dir: Path) -> Path:
    """Box plot: approximation ratio by method across all graphs."""
    fig, ax = plt.subplots(figsize=(8, 5))

    order = [m for m in ["baseline_qaoa", "pce_baseline_k1",
                          "pce_baseline_k2", "llm_pce"]
             if m in df["method"].unique()]
    palette = [METHOD_PALETTE[m][0] for m in order]
    labels = [METHOD_PALETTE[m][1] for m in order]

    sns.boxplot(data=df, x="method", y="approximation_ratio",
                order=order, palette=palette, ax=ax)
    sns.stripplot(data=df, x="method", y="approximation_ratio",
                  order=order, color="black", alpha=0.15, size=3, ax=ax)

    ax.set_xticklabels(labels, rotation=20)
    ax.set_xlabel("")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("Overall Method Comparison", fontsize=12)
    fig.tight_layout()
    path = save_dir / "method_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Plot 9: Compression ratio vs gradient norm ───────────────────────

def plot_compression_vs_gradient(df: pd.DataFrame,
                                  save_dir: Path) -> Path:
    """Scatter: compression ratio vs gradient norm."""
    pce = df[df["method"] != "baseline_qaoa"].dropna(
        subset=["compression_ratio", "gradient_norm_at_opt"]
    ).copy()

    if pce.empty:
        return _write_empty_plot(save_dir, "compression_vs_gradient",
                                 "No PCE data")

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ["pce_baseline_k1", "pce_baseline_k2", "llm_pce"]:
        group = pce[pce["method"] == method]
        if group.empty:
            continue
        color, label = METHOD_PALETTE[method]
        ax.scatter(
            group["compression_ratio"], group["gradient_norm_at_opt"],
            c=color, label=label, alpha=0.5, s=30,
            edgecolors="none",
        )

    ax.axhline(0.01, color="red", linestyle="--", alpha=0.4,
               label="Vanishing threshold")
    ax.set_xlabel("Compression Ratio (n / qubits)")
    ax.set_ylabel("Gradient Norm at Optimum")
    ax.set_title("Compression vs Gradient Behaviour", fontsize=12)
    ax.legend(**LEGEND_KWARGS)
    ax.set_yscale("symlog", linthresh=0.001)
    fig.tight_layout()
    path = save_dir / "compression_vs_gradient.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Helper ────────────────────────────────────────────────────────────

def _write_empty_plot(save_dir: Path, name: str,
                      reason: str) -> Path:
    """Write a placeholder image when plot data is unavailable."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"No data available:\n{reason}",
            ha="center", va="center", fontsize=12, color="gray")
    ax.set_title(name.replace("_", " ").title())
    path = save_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# ── Master plot runner ───────────────────────────────────────────────

def generate_all_plots(patterns_path: Path | None = None,
                       save_dir: Path | None = None,
                       verbose: bool = True) -> list[Path]:
    """Generate all analysis plots.

    Parameters
    ----------
    patterns_path : Path | None
        Path to patterns.json. If None, loads from default location.
    save_dir : Path | None
        Directory to save plots. Defaults to ``output/plots/``.
    verbose : bool
        Print progress.

    Returns
    -------
    list of paths to generated plot files.
    """
    save_dir = save_dir or PLOTS_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    df = _load()

    from analysis.patterns import PATTERNS_PATH as default_patterns_path
    patterns_src = patterns_path or default_patterns_path
    if patterns_src.exists():
        import json
        with open(patterns_src) as f:
            patterns = json.load(f)
    else:
        patterns = {}

    plot_funcs = [
        ("Method comparison", lambda: plot_method_comparison(df, save_dir)),
        ("Approx by family", lambda: plot_approx_by_method_family(df, save_dir)),
        ("Compression vs approx", lambda: plot_compression_vs_approx(df, save_dir)),
        ("Gradient vs approx", lambda: plot_gradient_vs_approx(df, save_dir)),
        ("Pauli fractions", lambda: plot_pauli_fractions(df, save_dir)),
        ("Feature correlations", lambda: plot_feature_correlation_heatmap(df, save_dir)),
        ("Compression vs gradient", lambda: plot_compression_vs_gradient(df, save_dir)),
        ("Feature-to-k heatmap", lambda: plot_feature_k_heatmap(patterns, save_dir)),
        ("Gradient risk by family", lambda: plot_gradient_risk_by_family(patterns, save_dir)),
    ]

    paths: list[Path] = []
    for name, func in plot_funcs:
        if verbose:
            print(f"  Plot {len(paths) + 1}: {name} …", end=" ")
        try:
            p = func()
            paths.append(p)
            if verbose:
                print(f"✓ ({p.name})")
        except Exception as e:
            if verbose:
                print(f"✗ ({e})")

    if verbose:
        print(f"\n  Generated {len(paths)} plots in {save_dir}")
    return paths


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate QLLM analysis plots"
    )
    parser.add_argument("--patterns", type=str, default=None,
                        help="Path to patterns.json")
    parser.add_argument("--output", type=str, default=None,
                        help="Plot output directory")
    args = parser.parse_args()

    save_dir = Path(args.output) if args.output else PLOTS_DIR
    patterns_path = Path(args.patterns) if args.patterns else None

    paths = generate_all_plots(
        patterns_path=patterns_path,
        save_dir=save_dir,
    )

    for p in paths:
        print(f"  {p}")
