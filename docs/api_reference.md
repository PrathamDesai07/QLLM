# QLLM API Reference

Function-level documentation for all public modules.

---

## `src/config.py` — Central Configuration

| Variable | Type | Description |
|----------|------|-------------|
| `PROJECT_ROOT` | `Path` | Root directory of the project |
| `DATA_DIR` | `Path` | `data/` — graphs, features, experiments |
| `GRAPHS_DIR` | `Path` | `data/graphs/` — generated graph instances |
| `FEATURES_DIR` | `Path` | `data/features/` — extracted graph features |
| `PCE_DIR` | `Path` | `data/pce/` — Pauli encoding specs |
| `EXPERIMENTS_DIR` | `Path` | `data/experiments/` — all experiment results |
| `PLOTS_DIR` | `Path` | `output/plots/` — generated figures |
| `BACKEND` | `dict` | Simulator config: name, shots, noise_model |
| `QAOA` | `dict` | Defaults: p=1, optimizer=COBYLA, max_iters=200, seed=42 |
| `GRAPH_DEFAULTS` | `dict` | Families, min/max nodes, step, instances_per_size |
| `PCE` | `dict` | default_k=2, max_k=3 |
| `HF_TOKEN` | `str\|None` | HuggingFace token loaded from `hf.txt` |
| `LLM_CONFIG` | `dict` | Model names, quantization, temperature, max_tokens |

---

## `src/graphs/generate_graphs.py` — Graph Generation

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `generate_erdos_renyi(n, p, seed)` | G(n, p) random graph | `nx.Graph` |
| `generate_d_regular(n, d, seed)` | d-regular random graph | `nx.Graph` |
| `generate_community(n, num_communities, p_in, p_out, seed)` | Stochastic block model with planted partition | `nx.Graph` |
| `generate_bipartite(n_left, n_right, p, seed)` | Bipartite random graph | `nx.Graph` |
| `generate_graph(family, **kwargs)` | Dispatch by family name | `nx.Graph` |
| `save_graph(g, family, params, directory)` | Save as node-link JSON with metadata | `Path` |
| `generate_batch(families, min_nodes, max_nodes, step, instances_per_size, seed_offset, directory)` | Generate all graph instances across the grid | `list[Path]` |

**CLI:** `python -m src.graphs.generate_graphs [--families] [--min-nodes] [--max-nodes] [--step] [--instances] [--seed-offset] [--output-dir]`

---

## `src/graphs/features.py` — Feature Extraction

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `basic_stats(g)` | num_nodes, num_edges, density, degree min/max/mean/std | `dict` |
| `degree_histogram(g, bins=10)` | Binned degree distribution | `dict` |
| `clustering(g)` | Average clustering coefficient, transitivity | `dict` |
| `connectivity(g)` | Connected components analysis | `dict` |
| `algebraic_connectivity(g)` | Fiedler eigenvalue (normalized Laplacian) | `dict` |
| `modularity(g)` | Community structure strength (greedy modularity) | `dict` |
| `extract_all(g)` | Aggregate all extractors into single dict (18 features) | `dict` |
| `features_from_graph_file(path, directory)` | Load graph JSON, extract, save feature JSON | `Path` |
| `batch_extract(graph_dir, feature_dir)` | Extract features for all graphs in directory | `list[Path]` |

**CLI:** `python -m src.graphs.features [--graph-dir] [--output-dir]`

---

## `src/qaoa/baseline_qaoa.py` — Uncompressed QAOA for MaxCut

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `maxcut_hamiltonian(g)` | Build (H_C, |E|) Ising Hamiltonian for MaxCut | `(SparsePauliOp, float)` |
| `maxcut_value_from_energy(energy, num_edges)` | Convert Ising energy to MaxCut value | `float` |
| `brute_force_maxcut(g)` | Exact MaxCut via enumeration (n ≤ 20) | `int` |
| `qaoa_ansatz(g, p)` | Build QAOA circuit with H+Mixer layers | `(QuantumCircuit, list[Parameter])` |
| `run_qaoa(g, p, optimizer, max_iters, seed)` | Run QAOA optimisation | `dict` |
| `run_baseline_qaoa(graph_path, p, optimizer, max_iters, output_dir)` | Full runner with logging | `ExperimentRecord` |
| `run_baseline_batch(graph_dir, output_dir, families, p)` | Batch runner across all graphs | `list[ExperimentRecord]` |

**CLI:** `python -m src.qaoa.baseline_qaoa [--graph-id] [--graph-dir] [--output-dir] [--families] [--p] [--optimizer] [--max-iters]`

---

## `src/qaoa/pce_qaoa_baseline.py` — PCE-Encoded QAOA

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `pce_qaoa_ansatz(hamiltonian, num_physical_qubits, p)` | Build PCE-encoded QAOA ansatz | `(QuantumCircuit, list[Parameter])` |
| `run_pce_qaoa(g, encoding, p, optimizer, max_iters, seed)` | Run PCE-encoded QAOA on a graph | `dict` |
| `run_pce_baseline(graph_path, encoding, encoding_path, k, p, optimizer, max_iters, output_dir)` | Full runner with logging | `ExperimentRecord` |
| `run_pce_batch(graph_dir, output_dir, families, k, p)` | Batch runner | `list[ExperimentRecord]` |

**CLI:** `python -m src.qaoa.pce_qaoa_baseline [--graph-id] [--graph-dir] [--output-dir] [--families] [--k] [--p] [--optimizer] [--max-iters]`

---

## `src/pce/manual_pce.py` — Pauli-Correlation Encoding

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `encode_k1_single_pauli(g, seed)` | k=1: each variable → one Pauli on one qubit | `dict` |
| `encode_k2_paired_pauli(g, seed)` | k=2: each variable → 2-qubit Pauli string | `dict` |
| `encode_k3_triple_pauli(g, seed)` | k=3: each variable → 3-qubit Pauli string | `dict` |
| `encode_graph(g, k, seed)` | Dispatch to encoder by k | `dict` |
| `build_pce_hamiltonian(g, encoding)` | Build PCE cost Hamiltonian from graph edges + encoding | `SparsePauliOp` |
| `save_encoding(encoding, graph_id, directory)` | Save encoding spec as JSON | `Path` |
| `encode_and_save(graph_path, k, seed, directory)` | Load, encode, save in one call | `dict` |

**CLI:** `python -m src.pce.manual_pce [--graph-id] [--graph-dir] [--output-dir] [--k] [--seed]`

---

## `src/llm/schema.py` — LLM Input/Output Schemas

### Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `GraphFeatures` | num_nodes, num_edges, density, degree stats, clustering, connectivity, algebraic_connectivity | Numerical features for LLM input |
| `AdjacencyInfo` | edge_list | Compact edge representation |
| `LLMGraphInput` | graph_id, family, features, adjacency, degree_histogram | Complete LLM input payload |
| `PauliAssignment` | variable, pauli_string, qubits, paulis | One variable's Pauli assignment |
| `LLMOutput` | graph_id, k, num_physical_qubits, pauli_assignments, tags, reasoning, approx_ratio_band | Structured LLM output with validation |

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `build_input_text(inp)` | Format `LLMGraphInput` as structured text for the prompt | `str` |

---

## `src/llm/client.py` — LLM Inference Client

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `get_llm_output(llm_input, model_name, prompt_path, temperature)` | Single graph → LLM output | `LLMOutput` |
| `get_llm_output_with_fallback(llm_input, model_name, prompt_path, temperature)` | LLM call with deterministic fallback | `LLMOutput` |
| `batch_get_llm_output_with_fallback(inputs, model_name, prompt_path, temperature)` | Batched GPU generation | `list[LLMOutput]` |

**Note:** Requires L4 GPU with ~24 GB VRAM for the primary model (Qwen2.5-32B). Falls back to k=1 encoding if LLM is unavailable.

---

## `src/pipeline/llm_guided_pce.py` — LLM-Guided PCE Pipeline

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `load_graph_and_features(graph_id, graph_dir, feature_dir)` | Load graph + features + build LLM input | `(nx.Graph, dict, LLMGraphInput)` |
| `run_llm_pce(graph_id, graph_dir, feature_dir, model_name, prompt_path, qaoa_p, optimizer, max_iters, temperature)` | Full pipeline for one graph | `dict` |
| `run_llm_pce_batch(graph_dir, feature_dir, families, max_graphs, ...)` | Sequential batch runner | `list[dict]` |
| `run_llm_pce_batch_parallel(graph_dir, feature_dir, families, max_graphs, max_workers, ...)` | Multi-threaded parallel batch | `list[dict]` |

**CLI:** `python -m src.pipeline.llm_guided_pce --graph-id ID [--model] [--p] [--parallel] [--all-graphs]`

---

## `src/pipeline/rule_based_pce.py` — Rule-Based PCE Pipeline

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `run_rule_pce(graph_id, graph_dir, feature_dir, qaoa_p, optimizer, max_iters, output_dir, verbose)` | Full rule-based pipeline for one graph | `dict` |
| `run_rule_pce_batch(graph_dir, feature_dir, families, max_graphs, skip_existing, ...)` | Batch runner | `list[dict]` |
| `rule_pce_status()` | Print completeness report | `None` |

**CLI:** `python -m src.pipeline.rule_based_pce --graph-id ID [--all-graphs] [--families] [--status] [--p]`

---

## `src/rules/encoder_rules.py` — Deterministic Rule Engine

### Rules

| Function | Description | Returns |
|----------|-------------|---------|
| `rule_choose_k(features, family)` | Rule A: determine k from density, degree, family | `dict` |
| `rule_choose_pauli_pattern(features, k)` | Rule B: Pauli layout strategy (balanced, XZ-dominant, etc.) | `dict` |
| `rule_avoid_barren_plateaus(features, k)` | Rule C: gradient-risk assessment | `dict` |
| `recommend_encoding(graph_id, features, graph, feature_dir)` | Rule D: full recommendation combining all rules | `dict` |
| `batch_recommend(graph_ids, feature_dir)` | Batch recommendations for all graphs | `list[dict]` |

**Thresholds** (derived from Phase 3 analysis):
- `DENSITY_THRESHOLD = 0.4` — below → k=2, above → k=1
- `DEGREE_MEAN_THRESHOLD = 4.5`
- `BASELINE_PREFERRED_FAMILIES = {"bipartite", "community"}`
- `PCE_K2_PREFERRED_FAMILIES = {"erdos_renyi", "d_regular"}`

**CLI:** `python -m src.rules.encoder_rules --graph-id ID [--batch]`

---

## `src/analysis/aggregate_results.py` — Data Aggregation

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `load_features(graph_id)` | Load feature dict from disk | `dict` |
| `aggregate(skip_llm, verbose)` | Merge all experiments + features + Pauli stats into DataFrame | `pd.DataFrame` |
| `save_aggregate(df, path)` | Save as Parquet + CSV | `Path` |
| `status_report()` | Print completeness report | `None` |

**Output:** `data/analysis/qaoa_pce_results.parquet` (405 rows, 40+ columns)

**CLI:** `python -m src.analysis.aggregate_results [--status] [--skip-llm] [--output PATH]`

---

## `src/analysis/patterns.py` — Pattern Mining

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `pattern_preferred_k(df)` | Feature bin → best k analysis (7 features) | `dict` |
| `pattern_gradient_risk(df)` | Feature → gradient health with Spearman correlations | `dict` |
| `pattern_pauli_layout(df)` | Pauli fractions → performance correlations | `dict` |
| `pattern_compression_vs_performance(df)` | Compression ratio trade-off analysis | `dict` |
| `pattern_family_summary(df)` | Per-family best method | `dict` |
| `mine_patterns(df, verbose)` | Run all 5 pattern miners | `dict` |
| `save_patterns(patterns, path)` | Save to `data/analysis/patterns.json` | `Path` |

**CLI:** `python -m src.analysis.patterns [--no-llm] [--output PATH]`

---

## `src/analysis/plots.py` — Visualization

### Functions

| Function | Description |
|----------|-------------|
| `plot_method_comparison(df, save_dir)` | Box plot: 5 methods × approximation ratio |
| `plot_approx_by_method_family(df, save_dir)` | 4-panel box plot by family |
| `plot_compression_vs_approx(df, save_dir)` | Scatter: compression × approximation |
| `plot_gradient_vs_approx(df, save_dir)` | Scatter: gradient × approximation (full + zoom) |
| `plot_pauli_fractions(df, save_dir)` | Stacked bar: Pauli X/Y/Z distribution per method |
| `plot_feature_correlation_heatmap(df, save_dir)` | Spearman correlation heatmap |
| `plot_compression_vs_gradient(df, save_dir)` | Scatter: compression × gradient |
| `plot_feature_k_heatmap(patterns, save_dir)` | Heatmap: feature bin × best k |
| `plot_gradient_risk_by_family(patterns, save_dir)` | Stacked bar: vanishing/healthy per family |
| `generate_all_plots(patterns_path, save_dir, verbose)` | Generate all 9 plots |

**CLI:** `python -m src.analysis.plots [--patterns PATH] [--output DIR]`

---

## `src/analysis/compare_methods.py` — Cross-Method Comparison

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `ensure_rule_pce_results(families, max_graphs, verbose)` | Run rule-based PCE for missing graphs | `set[str]` |
| `build_comparison_dataset(run_rule_pce_first, families, max_graphs, verbose)` | Build 5-method DataFrame | `pd.DataFrame` |
| `method_comparison(df, run_rule_pce_first, verbose)` | Full comparison analysis | `dict` |
| `save_comparison(comparison, path)` | Save to `data/analysis/method_comparison.json` | `Path` |
| `print_comparison_summary(comparison)` | Human-readable summary | `None` |

**CLI:** `python -m src.analysis.compare_methods [--run-rule-pce] [--max-graphs N] [--output PATH]`

---

## `src/experiments/suites/maxcut_suites.py` — Experiment Suite Orchestrator

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `build_grid(families, min_nodes, max_nodes, step, instances_per_size)` | Build experiment grid configs | `list[dict]` |
| `grid_nodes(grid)` | Count graph instances | `int` |
| `suite_status(families, k_values)` | Check which experiments are complete | `dict` |
| `run_maxcut_suite(families, baseline, pce, llm, ...)` | Run full experiment grid | `dict` |
| `print_suite_status(families, k_values)` | Human-readable status | `None` |

**CLI:** `python -m src.experiments.suites.maxcut_suites [--status] [--families] [--no-baseline] [--no-pce] [--no-llm] [--pce-k] [--max-graphs N] [--llm-parallel]`

---

## `src/experiments/nisq_runs.py` — NISQ Noisy Simulation

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `noisy_expectation(circuit, observable, params, device_name)` | ⟨ψ|H|ψ⟩ under device noise | `float` |
| `run_nisq(graph_id, device_name, p, optimizer, max_iters, seed, verbose)` | Run all methods under noise | `dict` |
| `run_nisq_batch(device_name, families, max_graphs, skip_existing, ...)` | Batch runner | `list[dict]` |
| `analyze_nisq_results()` | Aggregate NISQ results + produce comparison | `dict` |
| `print_nisq_status()` | Completeness report | `None` |

**CLI:** `python -m src.experiments.nisq_runs [--graph-id] [--device] [--batch] [--families] [--max-graphs N] [--status] [--analyze] [--p N]`

---

## `src/infra/simulator.py` — Simulation Backend

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `compute_expectation(circuit, observable, parameter_values, shots)` | ⟨ψ|H|ψ⟩ via StatevectorEstimator | `float` |
| `compute_expectation_batch(circuit, observable, parameter_sets, shots)` | Batch expectation values | `list[float]` |
| `sample_distribution(circuit, parameter_values, shots)` | Measurement outcome distribution | `dict[str, float]` |
| `estimate_gradient(circuit, observable, params, eps)` | Finite-difference gradient | `np.ndarray` |

---

## `src/infra/logger.py` — Experiment Logging

### Classes

| Class | Fields | Description |
|-------|--------|-------------|
| `ExperimentRecord` | graph_id, family, num_nodes, num_edges, method, qaoa_p, optimizer, max_iters, optimal_energy, optimal_params, approximation_ratio, gradient_norm_at_opt, convergence_iters, success, timestamp, duration_seconds, params, extra | Standard experiment dataclass |

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `ExperimentRecord.save(directory)` | Save as JSON in `directory / method / {graph_id}.json` | `Path` |
| `load_experiment(path)` | Load record from JSON | `ExperimentRecord` |
| `timer(fn)` | Decorator: records duration + re-saves | `ExperimentRecord` |

---

## `src/infra/experiment_tracker.py` — Versioning & Reproducibility

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `build_index_record(record, method, git_commit, ...)` | Build flat CSV row from ExperimentRecord | `dict` |
| `run_metadata(model_name, prompt_file, extras)` | Collect versioning metadata | `dict` |
| `append_index(record)` | Append one run to central index CSV | `None` |
| `load_index()` | Load index as list of dicts | `list[dict]` |

**Index columns:** graph_id, family, method, model_name, prompt_file, prompt_hash, timestamp, k, num_physical_qubits, compression_ratio, optimal_energy, approximation_ratio, gradient_norm_at_opt, success, convergence_iters, duration_seconds, llm_tags, llm_reasoning, git_commit

---

## `src/infra/hardware_mapping.py` — NISQ Device Models

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `build_noise_model(device_name)` | Build Aer NoiseModel with thermal + depolarizing + readout errors | `NoiseModel` |
| `list_devices()` | Print device profiles | `None` |

### Devices

| Device | Qubits | T1 | T2 | 1Q err | 2Q err | RO err |
|--------|--------|----|----|--------|--------|--------|
| `ibm_eagle` | 27 | 280µs | 150µs | 0.02% | 0.6% | 1.0% |
| `ibm_heron` | 27 | 350µs | 200µs | 0.015% | 0.4% | 0.8% |
| `ideal` | 100 | — | — | 0 | 0 | 0 |

---

## `src/problems/labs.py` — LABS Problem

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `labs_hamiltonian(n)` | Build LABS Ising Hamiltonian (4-body ZZ terms) | `SparsePauliOp` |
| `brute_force_best_labs(n)` | Exact optimum for n ≤ 18 | `dict` |
| `run_labs_qaoa(n, p, optimizer, max_iters, seed)` | Uncompressed QAOA for LABS | `dict` |
| `run_labs_pce(n, k, p, optimizer, max_iters, seed)` | PCE-encoded QAOA for LABS | `dict` |

**CLI:** `python -m src.problems.labs [--n N] [--method baseline|pce] [--k N] [--p N] [--brute-force] [--batch] [--n-values N [N ...]]`

---

## `src/problems/budget_constrained.py` — Budget-Constrained Optimisation

### Functions

| Function | Description | Returns |
|----------|-------------|---------|
| `generate_instance(n, value_range, cost_range, budget_ratio, seed)` | Generate random instance with items, costs, budget | `dict` |
| `brute_force_optimum(instance)` | Exact optimum for n ≤ 20 | `dict` |
| `budget_hamiltonian(instance)` | Ising Hamiltonian via QUBO→Ising conversion | `SparsePauliOp` |
| `run_budget_qaoa(instance, p, optimizer, max_iters, seed)` | Uncompressed QAOA | `dict` |
| `run_budget_pce(instance, k, p, optimizer, max_iters, seed)` | PCE-encoded QAOA | `dict` |
| `run_budget_rule(instance, p, optimizer, max_iters, seed)` | Rule-based PCE QAOA | `dict` |

**CLI:** `python -m src.problems.budget_constrained [--n N] [--method baseline|pce|rule] [--k N] [--p N] [--seed N] [--batch] [--n-values N [N ...]]`
