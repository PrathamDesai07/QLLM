# QLLM Pattern Catalog

> Discovered patterns in the QAOA / Pauli-correlation encoding design space.
> Generated from 81 graph instances × 4 methods (baseline QAOA, PCE k=1, PCE k=2, LLM-guided PCE)
> across 4 graph families (Erdős–Rényi, d-regular, community, bipartite).

---

## Pattern A: Graph family determines optimal method

| Family | Best Method | Mean Approx Ratio |
|--------|------------|-------------------|
| Erdős–Rényi | **PCE k=2** | 0.781 |
| d-regular | **PCE k=2** | 0.684 |
| Community | **Baseline QAOA** | 0.761 |
| Bipartite | **Baseline QAOA** | 0.604 |

**Interpretation:** Dense, irregular graphs (Erdős–Rényi) benefit most from Pauli correlation encoding with k=2, achieving compression ratios of ~5× while _improving_ upon baseline QAOA approximation. Regular and structured graphs (d-regular) also benefit from PCE. Community graphs and bipartite graphs perform best with uncompressed QAOA — the correlation encoding loses too much structure.

---

## Pattern B: Higher graph density and degree favour k=2

For low-density graphs (density < 0.3, bins 0-1), k=2 is the preferred correlation order. For higher-density graphs (density > 0.5, bins 2-3), k=1 becomes competitive.

| Density Bin | Best k | Samples |
|-------------|--------|---------|
| Lowest (bin 0) | k=2 | 87 |
| Low–mid (bin 1) | k=2 | 90 |
| Mid–high (bin 2) | k=1 | 27 |
| Highest (bin 3) | k=1 | 39 |

**Interpretation:** Sparse graphs benefit from higher-order correlations (k=2) to capture the limited edge structure. Dense graphs are already well-represented by single-Pauli encoding (k=1), and the additional qubit overhead of k=2 doesn't yield returns.

---

## Pattern C: Gradient health correlates strongly with node degree

| Feature | Spearman ρ vs Gradient Norm | p-value |
|---------|---------------------------|---------|
| degree_mean | **+0.57** | < 0.001 |
| algebraic_connectivity | **+0.48** | < 0.001 |
| degree_std | **+0.39** | < 0.001 |
| num_nodes | +0.36 | < 0.001 |
| density | +0.33 | < 0.001 |
| component_size_ratio | +0.29 | < 0.001 |
| avg_clustering | +0.19 | 0.003 |
| transitivity | +0.20 | 0.002 |

All features show a positive correlation — denser, more connected graphs produce healthier gradients. **This is the strongest signal in the dataset.** Gradient norm increases with graph complexity, indicating that vanishing gradients are primarily a problem for small, sparse graphs.

---

## Pattern D: PCE k=2 has the healthiest gradients, LLM-PCE the most vanishing

| Method | Vanishing (grad < 0.01) | Healthy |
|--------|-----------------------|---------|
| PCE k=2 | **32** | **49** |
| PCE k=1 | 41 | 40 |
| LLM PCE | **63** | **18** |

**Interpretation:** k=2 encoding consistently produces the healthiest gradient landscape — the additional correlation structure provides more trainable directions. LLM-guided PCE shows the worst gradient health (78% vanishing), suggesting the LLM's Pauli assignments may introduce barren plateau conditions. This warrants further investigation.

---

## Pattern E: Higher compression correlates with healthier gradients

- Compression ratio vs gradient norm: **ρ = +0.38** (p < 0.001)
- Compression ratio vs approximation ratio: **ρ = +0.13** (p = 0.049)

Higher compression ratios (fewer qubits per variable) are associated with _larger_ gradient norms, not smaller. This is good news — compression does not inherently cause vanishing gradients. The trade-off is modest: higher compression gives slightly better gradients and no significant approximation penalty.

---

## Pattern F: Pauli operator choice has limited direct impact

| Metric | X fraction ρ | Y fraction ρ | Z fraction ρ |
|--------|-------------|-------------|-------------|
| vs Approximation ratio | -0.07 | -0.06 | -0.04 |
| vs Gradient norm | **-0.34** | **-0.36** | **-0.34** |

All three Pauli operators (X, Y, Z) correlate _negatively_ with gradient norm at similar magnitudes. The dominant effect is the number of non-identity operators per variable (ρ = -0.32 with gradient), not which specific Pauli is used. **The key insight: sparse Pauli strings (fewer non-identity ops) correlate with vanishing gradients.** Assignments with an average of 1+ non-identity ops per variable produce healthier training.

---

## Pattern G: Best-performing k=2 assignments blend X, Y, and Z

For graphs where k=2 gives the best approximation ratio:
- Mean X fraction: 0.062
- Mean Y fraction: 0.057
- Mean Z fraction: 0.049
- Mean non-id ops per variable: **1.007**

For k=1 in its best regime:
- Mean X fraction: 0.010
- Mean Y fraction: 0.010
- Mean Z fraction: 0.000
- Mean non-id ops per variable: 0.077

Well-performing k=2 encodings use a balanced mix of all three Pauli operators with roughly one non-identity per variable. k=1 best-performers rely almost entirely on X and Y.

---

## Summary of Design Rules

1. **For sparse graphs (density < 0.3):** use PCE k=2
2. **For dense graphs (density > 0.5):** consider k=1 or uncompressed QAOA
3. **For community/bipartite families:** uncompressed QAOA is preferred
4. **For Erdős–Rényi / d-regular:** PCE k=2 gives best results at 5× qubit savings
5. **To avoid vanishing gradients:** ensure Pauli strings have 1+ non-identity ops per variable; prefer k=2 encoding
6. **Paulie operator balance:** mix X, Y, and Z roughly equally — no single operator dominates in best-performing regimes
7. **Compression is safe:** higher compression ratios do not degrade gradient quality or approximation ratio meaningfully
