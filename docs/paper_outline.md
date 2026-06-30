# QLLM: LLM-Guided Pauli-Correlation Encoding for Qubit-Efficient QAOA

## Paper Outline

### Abstract
- Problem: NISQ-era QAOA limited by qubit count; Pauli-correlation encoding (PCE) compresses variables but requires expert design choices
- Approach: LLM-guided and rule-based frameworks for designing PCE strategies, validated across 4 graph families, 3 correlation orders, and 5 methods
- Key results: PCE k=2 achieves up to 5× qubit compression with approximation ratios comparable to uncompressed QAOA; degree_mean and algebraic_connectivity are strong predictors of gradient health (ρ=+0.57, ρ=+0.48); rule-based encoder matches or beats LLM-guided PCE in 63% of cases

### 1. Introduction
- Quantum Approximate Optimization Algorithm (QAOA) for MaxCut
- NISQ hardware constraints and the qubit-efficiency problem
- Pauli-correlation encoding (PCE) as a compression technique
- Challenge: choosing k (correlation order) and Pauli assignments optimally
- Contributions: (1) LLM-guided PCE pipeline, (2) data-driven pattern discovery from 405 experiments, (3) formalized rule engine matching LLM quality

### 2. Background & Related Work
- QAOA formulation (1:1 mapping, circuit depth p)
- Pauli encoding: variable-to-qubit mapping with X/Y/Z operators
- Correlation order k: trade-off between compression (qubits saved) and expressivity
- Prior work on ansatz design, barren plateaus, and learning-based compilation

### 3. Methodology

#### 3.1 Graph Generation & Feature Extraction
- 4 families: Erdős–Rényi, d-regular, community (SBM), bipartite
- 4 sizes: 8, 12, 16, 20 nodes × 5 seeds each = 80 graphs
- 18 extracted features: density, degree statistics, clustering, connectivity, modularity, algebraic connectivity

#### 3.2 PCE Encoding Strategies
- k=1: single-Pauli per variable (up to 3×/qubit)
- k=2: paired-Pauli strings (~3-5× compression)
- k=3: triple-Pauli strings (handles dense modular graphs)
- Hamiltonian construction via Pauli string multiplication

#### 3.3 LLM-Guided Pipeline
- Model: Qwen2.5-32B-Instruct (Q4_K_M, L4 GPU)
- Input: graph features + adjacency; Output: k, Pauli assignments, structural tags
- Prompt-based generation with JSON schema validation
- Pipeline: load graph → call LLM → build circuit → simulate → log

#### 3.4 Rule-Based Engine
- Deterministic rules derived from Phase 3 pattern analysis
- Family-based dispatch, density/degree/modularity thresholds
- Gradient risk assessment and Pauli layout strategy selection
- No LLM required — transparent and interpretable

### 4. Experimental Setup
- 405 total runs (80 graphs × 5 methods + 1 extra graph)
- QAOA p=1, COBYLA optimizer, 200 iterations
- Metrics: approximation ratio, gradient norm at optimum, compression ratio, convergence success
- Simulation: Qiskit statevector/estimator backend

### 5. Results

#### 5.1 Method Comparison
- Baseline QAOA: mean approx 0.698 (49.4% win rate)
- PCE k=2: mean approx 0.692, 4.99× compression (37.0% win rate)
- Rule-based PCE: mean approx 0.678, 3.90× compression, beats LLM PCE in 63% of cases
- LLM PCE: mean approx 0.653, 2.01× compression (6.2% win rate)
- PCE k=1: mean approx 0.666, 2.80× compression

#### 5.2 Compression vs Quality
- k=2: 5× qubit savings with < 1% approximation loss vs baseline
- k=1: 2.8× savings with ~3% loss
- Higher compression correlates with healthier gradients (ρ=+0.38)

#### 5.3 Feature-Gradient Relationships
- Strongest predictors of gradient health: degree_mean (ρ=+0.57), algebraic_connectivity (ρ=+0.48), degree_std (ρ=+0.39)
- Higher-degree, better-connected graphs avoid barren plateaus
- Vanishing gradients: 78% of LLM PCE runs vs 40% of PCE k=2 runs

#### 5.4 Pauli Layout Analysis
- No single Pauli operator strongly correlates with approximation quality
- Sparse Pauli strings (few non-identity ops) correlate with vanishing gradients (ρ=-0.32)
- Best k=2 assignments: ~1 non-identity op per variable with balanced X/Y/Z

#### 5.5 Family-Specific Results
- Erdős–Rényi: PCE k=2 best (0.781) — dense irregular graphs benefit most
- d-regular: PCE k=2 best (0.684) — regular structure compresses well
- Community: Baseline QAOA best (0.761) — community structure lost in compression
- Bipartite: Baseline QAOA best (0.604) — limited structure to exploit

### 6. Pattern Catalog & Design Rules
- Rule A: Sparse graphs → k=2; dense → k=1 (threshold: density < 0.4)
- Rule B: Modular graphs → X/Z-dominant Pauli layout
- Rule C: Low degree → gradient risk → prefer k=2 for more non-identity ops
- Rule D: Erdős–Rényi + d-regular → PCE k=2; community + bipartite → baseline QAOA
- Rule E: Compression correlates with gradient health — compression is safe

### 7. Discussion
- LLM vs Rules: LLM Pauli assignments may create suboptimal training landscapes
- When an LLM helps vs when rules suffice
- Interpretability: rules provide clear scientific insight; LLM provides flexible exploration
- Limitations: MaxCut only, p=1, small graph sizes, simulation-only

### 8. Conclusion & Future Work
- Demonstrated effective qubit compression (up to 5×) via PCE for QAOA
- Identified interpretable patterns in the design space
- Rule engine matches or exceeds LLM-guided PCE
- Future: higher QAOA depth (p>1), additional problem families (MIS, LABS), NISQ deployment

### References
- Farhi et al. (2014) — QAOA
- Hadfield et al. (2019) — Quantum alternating operator ansatz
- McClean et al. (2018) — Barren plateaus
- Wang et al. (2018) — PCE and qubit-efficient encoding
- Qiskit Development Team — Quantum simulation backend

### Figures (planned)
1. Method comparison: box plot of approximation ratios (5 methods)
2. Compression vs approximation scatter (3 PCE methods)
3. Feature correlation heatmap (Spearman ρ matrix)
4. Gradient risk by family (stacked bar)
5. Feature → preferred k heatmap
6. Pauli operator distribution (stacked bar per method)
7. Family-faceted approximation box plots (4 families × 5 methods)
8. LLM reasoning distribution (tag frequency)

### Data Availability
- Code: https://github.com/PrathamDesai07/QLLM
- Dataset: 405 experiment records with features, Pauli stats, and metrics
- Patterns and analysis: https://github.com/PrathamDesai07/QLLM/tree/main/data/analysis
