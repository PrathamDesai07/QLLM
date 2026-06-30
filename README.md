# QLLM — LLM-Guided Pauli-Correlation Encoding for QAOA

QLLM is a research framework that uses LLMs and data-driven rules to discover interpretable structural patterns in the QAOA / Pauli-correlation encoding (PCE) design space. It selects correlation orders, allocates Pauli strings, and shapes QAOA ansätze for qubit-efficient optimization on NISQ hardware.

**Key results across 420 experiment records (5 problem families × up to 5 methods):**
- Up to **5× qubit compression** via PCE with approximation ratios comparable to uncompressed QAOA
- **Rule-based PCE matches or beats LLM-guided PCE in 63% of cases** (mean approx +0.025)
- PCE k=2 produces the **healthiest gradient landscape** (39.5% vanishing vs 77.8% for LLM)
- **degree_mean** and **algebraic_connectivity** are the strongest predictors of gradient health (ρ=+0.57, ρ=+0.48)

---

## Project Structure

```
QLLM/
├── src/
│   ├── config.py                   # Central configuration
│   ├── graphs/
│   │   ├── generate_graphs.py      # Graph family generation (4 families, 8-20 nodes)
│   │   └── features.py             # 18 graph features including modularity
│   ├── qaoa/
│   │   ├── baseline_qaoa.py        # Uncompressed QAOA reference
│   │   └── pce_qaoa_baseline.py    # PCE-encoded QAOA circuits + runner
│   ├── pce/
│   │   └── manual_pce.py           # PCE encodings (k=1, k=2, k=3) + Hamiltonian builder
│   ├── llm/
│   │   ├── schema.py               # LLM input/output dataclasses
│   │   ├── client.py               # LLM inference (Qwen2.5-32B, 4-bit quantized)
│   │   └── prompts/
│   │       └── qaoa_pce_prompt.txt
│   ├── pipeline/
│   │   ├── llm_guided_pce.py       # LLM-guided PCE pipeline (single + batch + parallel)
│   │   └── rule_based_pce.py       # Deterministic rule-based PCE pipeline
│   ├── rules/
│   │   └── encoder_rules.py        # Rule engine (k selection, Pauli layout, gradient risk)
│   ├── analysis/
│   │   ├── aggregate_results.py    # Merge 5 methods into unified Parquet dataset
│   │   ├── patterns.py             # Statistical pattern mining (5 categories)
│   │   ├── plots.py                # 9 publication-ready plots
│   │   └── compare_methods.py      # Cross-method comparison (5 methods, 405 runs)
│   ├── experiments/
│   │   ├── suites/
│   │   │   └── maxcut_suites.py    # MaxCut experiment grid orchestrator
│   │   └── nisq_runs.py            # Noisy NISQ simulation (AerSimulator + noise models)
│   ├── problems/
│   │   ├── labs.py                 # Low Autocorrelation Binary Sequence
│   │   └── budget_constrained.py   # Knapsack-style budget-constrained optimisation
│   └── infra/
│       ├── simulator.py            # Qiskit simulation wrapper (estimator + sampler)
│       ├── logger.py               # ExperimentRecord dataclass + JSON persistence
│       ├── experiment_tracker.py   # Central index CSV + versioning
│       └── hardware_mapping.py     # Device profiles + noise models (Eagle, Heron)
├── data/
│   ├── graphs/                     # 81 graph instances (JSON, node-link format)
│   ├── features/                   # Extracted features (18 per graph)
│   ├── pce/                        # Pauli encoding specs (manual, LLM, rule)
│   ├── experiments/                # Experiment results (420 records across 7 methods)
│   │   ├── index.csv               # Central experiment registry (405+ rows)
│   │   ├── baseline_qaoa/          # Uncompressed QAOA (81 graphs)
│   │   ├── pce_baseline_k1/        # PCE k=1 (81 graphs)
│   │   ├── pce_baseline_k2/        # PCE k=2 (81 graphs)
│   │   ├── llm_pce/                # LLM-guided PCE (81 graphs)
│   │   ├── rule_pce/               # Rule-based PCE (81 graphs)
│   │   ├── labs/                   # LABS problem results
│   │   ├── budget/                 # Budget-constrained problem results
│   │   └── nisq/                   # NISQ noisy simulation results
│   └── analysis/                   # Aggregated datasets, patterns, comparisons
├── docs/
│   ├── api_reference.md            # Function-level API documentation
│   ├── pattern_catalog.md          # Discovered patterns & design rules
│   └── paper_outline.md            # Paper structure for publication
├── output/
│   └── plots/                      # 9 analysis plots (PNG)
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/PrathamDesai07/QLLM
cd QLLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install qiskit-aer    # required for NISQ simulations
```

Create `hf.txt` with your HuggingFace token:

```
hf_your_token_here
```

Configure models, paths, and defaults in `src/config.py`.

---

## LLM Models

### Primary (L4 24 GB)
| Model | HF Link | Role |
|-------|---------|------|
| **Qwen/Qwen2.5-32B-Instruct** (default) | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct) | Main quantum design LLM |
| google/gemma-2-27b-it | [HuggingFace](https://huggingface.co/google/gemma-2-27b-it) | Alternative with stronger coding |

### Utility (T4 16 GB)
| Model | HF Link | Role |
|-------|---------|------|
| **meta-llama/Llama-3.1-8B-Instruct** (default) | [HuggingFace](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | Fast iteration & dev |
| Qwen/Qwen2.5-7B-Instruct | [HuggingFace](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | Smaller Qwen sibling |

---

## Run Instructions

### 1. Graph Generation

```bash
# Generate all 81 graphs (4 families × 4 sizes × 5 seeds + 1 extra)
python -m src.graphs.generate_graphs

# Generate specific families
python -m src.graphs.generate_graphs --families erdos_renyi community --min-nodes 8 --max-nodes 12

# Extract features
python -m src.graphs.features
```

### 2. Baseline QAOA

```bash
# Single graph
python -m src.qaoa.baseline_qaoa --graph-id erdos_renyi_8_0

# All graphs
python -m src.qaoa.baseline_qaoa

# Filter by family
python -m src.qaoa.baseline_qaoa --families erdos_renyi community
```

### 3. Manual PCE Baseline

```bash
# k=1 encoding
python -m src.pce.manual_pce --k 1

# k=2 encoding + QAOA
python -m src.qaoa.pce_qaoa_baseline --k 2

# Single graph
python -m src.qaoa.pce_qaoa_baseline --graph-id erdos_renyi_8_0 --k 2
```

### 4. LLM-Guided PCE Pipeline

```bash
# Single graph (requires L4 GPU with ~24 GB VRAM)
python -m src.pipeline.llm_guided_pce --graph-id erdos_renyi_8_0

# All graphs (sequential)
python -m src.pipeline.llm_guided_pce --all-graphs

# All graphs (parallel batched GPU generation)
python -m src.pipeline.llm_guided_pce --all-graphs --parallel --max-workers 4
```

### 5. Rule-Based PCE Pipeline

```bash
# Single graph (no GPU required)
python -m src.pipeline.rule_based_pce --graph-id erdos_renyi_8_0

# All 81 graphs (~30 min on CPU)
python -m src.pipeline.rule_based_pce --all-graphs

# Check status
python -m src.pipeline.rule_based_pce --status
```

### 6. Rule Engine (standalone)

```bash
# Single graph recommendation
python -m src.rules.encoder_rules --graph-id erdos_renyi_8_0

# All graphs
python -m src.rules.encoder_rules --batch
```

### 7. Experiment Suite

```bash
# Full MaxCut experiment grid (runs baseline, PCE, and LLM)
python -m src.experiments.suites.maxcut_suites

# Status report only
python -m src.experiments.suites.maxcut_suites --status

# Specific families without LLM
python -m src.experiments.suites.maxcut_suites --families erdos_renyi community --no-llm
```

### 8. Analysis Pipeline

```bash
# Aggregation (build Parquet dataset from all experiment files)
python -m src.analysis.aggregate_results
python -m src.analysis.aggregate_results --status
python -m src.analysis.aggregate_results --skip-llm  # skip LLM rows

# Pattern mining
python -m src.analysis.patterns

# Plots
python -m src.analysis.plots

# Cross-method comparison
python -m src.analysis.compare_methods
```

### 9. Alternative Problem Families

```bash
# LABS (Low Autocorrelation Binary Sequence)
python -m src.problems.labs --n 8 --method baseline
python -m src.problems.labs --n 8 --method pce --k 2
python -m src.problems.labs --n 6 --brute-force  # exact optimum
python -m src.problems.labs --batch --n-values 4 6 8 10

# Budget-constrained (knapsack-style)
python -m src.problems.budget_constrained --n 8 --method pce --k 2
python -m src.problems.budget_constrained --batch --n-values 4 6 8
python -m src.problems.budget_constrained --n 6 --show-instance  # view random instance
```

### 10. NISQ Simulation

```bash
# List available device profiles
python -m src.infra.hardware_mapping --list-devices

# Single graph under noise
python -m src.experiments.nisq_runs --graph-id erdos_renyi_8_0 --device ibm_eagle

# Small batch (note: noisy simulation is ~15s/graph/method)
python -m src.experiments.nisq_runs --batch --device ibm_eagle --max-graphs 5

# Aggregate and analyse NISQ results
python -m src.experiments.nisq_runs --analyze
```

---

## Results Summary

### Method Comparison (MaxCut, 81 graphs)

| Method | Mean Approx | Mean Grad | Mean Compression | Vanishing Grad |
|--------|------------|-----------|-----------------|---------------|
| Baseline QAOA | 0.698 | 0.012 | 1.0× | 63.0% |
| PCE k=2 | 0.692 | 0.075 | **5.0×** | **39.5%** |
| Rule-based PCE | 0.678 | 0.032 | 3.9× | 49.4% |
| PCE k=1 | 0.666 | 0.041 | 2.8× | 50.6% |
| LLM-guided PCE | 0.653 | 0.041 | 2.0× | 77.8% |

### Best Method by Graph Family

| Family | Best Method | Mean Approx |
|--------|------------|-------------|
| Erdős–Rényi | **PCE k=2** | 0.781 |
| d-regular | **PCE k=2** | 0.684 |
| Community | **Baseline QAOA** | 0.761 |
| Bipartite | **Baseline QAOA** | 0.604 |

### Gradient Predictors (Spearman ρ)

| Feature | ρ vs Gradient | p-value |
|---------|-------------|---------|
| degree_mean | +0.57 | < 0.001 |
| algebraic_connectivity | +0.48 | < 0.001 |
| degree_std | +0.39 | < 0.001 |

See [docs/pattern_catalog.md](docs/pattern_catalog.md) for the full pattern catalog and [docs/paper_outline.md](docs/paper_outline.md) for the paper structure.

---

## Reproducibility

Every experiment records:
- **Git commit hash** at time of run
- **Model name** and **prompt file** (for LLM runs)
- **Timestamp** and **duration**
- **All parameters** (QAOA depth, optimizer, max iterations)
- **Full results** (energy, approximation ratio, gradient norm, convergence)

The central index at `data/experiments/index.csv` provides a complete, queryable record of all runs.

---

## License

MIT — see [LICENSE](LICENSE).
