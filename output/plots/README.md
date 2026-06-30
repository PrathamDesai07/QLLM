# Analysis Plots — Guide

## Files

| File | Description |
|------|-------------|
| `method_comparison.png` | **Overall method comparison.** Box plot of approximation ratios for all 5 methods (baseline QAOA, PCE k=1, PCE k=2, LLM-guided PCE, rule-based PCE) across all 81 graphs. Baseline QAOA shows the highest median (~0.71) and widest spread; PCE k=2 is close behind at ~5× compression. The LLM-guided PCE has the lowest median and tightest spread. |
| `approx_by_method_family.png` | **Family-faceted method comparison.** Four side-by-side box plots, one per graph family (Erdős–Rényi, d-regular, community, bipartite). Shows that Erdős–Rényi and d-regular families benefit most from PCE k=2, while community and bipartite families perform best with uncompressed baseline QAOA. |
| `compression_vs_approx.png` | **Compression vs approximation trade-off.** Scatter plot: each point is one (graph, method) pair, colored by method (PCE k=1 blue, PCE k=2 orange, LLM PCE green). The x-axis is compression ratio (higher = more qubits saved), y-axis is approximation ratio. PCE k=2 achieves the highest compression (~5×) while maintaining competitive approximation ratios. |
| `compression_vs_gradient.png` | **Compression vs gradient behaviour.** Scatter plot showing compression ratio vs gradient norm at optimum, with a dashed red line at the vanishing threshold (0.01). Higher compression ratios correlate with healthier gradient norms (Spearman ρ = +0.38). PCE k=2 (orange) dominates the high-compression, healthy-gradient region. |
| `gradient_vs_approx.png` | **Gradient vs approximation quality.** Two-panel figure: (left) all methods, (right) zoom on gradient norm < 0.1. Dashed red line marks the vanishing threshold. Baseline QAOA (grey) clusters near the vanishing threshold with moderate approximation. PCE k=2 (orange) shows the healthiest gradients. LLM PCE (green) has many points in the vanishing region, indicating potential barren plateau issues. |
| `pauli_fractions.png` | **Pauli operator distribution by method.** Stacked bar chart showing the mean fraction of X (red), Y (green), and Z (blue) operators per method. PCE k=1 uses only X (no Y or Z). LLM PCE and PCE k=2 both use all three operators with a roughly even split. |
| `feature_correlations.png` | **Feature vs performance correlations.** Heatmap of Spearman correlation coefficients between 9 graph features and 3 performance metrics (approximation ratio, gradient norm, compression ratio). Degree_mean and algebraic_connectivity show the strongest positive correlations with gradient norm (ρ = +0.57, +0.48). Density and degree_std also show moderate positive correlations. |
| `feature_k_heatmap.png` | **Preferred correlation order by feature bin.** Heatmap showing which k (1 or 2) gives the best approximation ratio for each feature's value bins (low → high). For most features, lower values favour k=2 and higher values favour k=1. Density, degree_mean, and avg_clustering show the clearest transitions. |
| `gradient_risk_by_family.png` | **Gradient health by graph family.** Stacked bar chart showing the count of vanishing (grad < 0.01, red) vs healthy (green) PCE experiments for each graph family. Erdős–Rényi graphs have the healthiest gradients (most green); community and d-regular graphs have the most vanishing gradients. |

## How to Regenerate

```bash
cd /teamspace/studios/this_studio/QLLM
python -m src.analysis.plots
```
