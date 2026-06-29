# QLLM — LLM-Guided Pauli-Correlation Encoding for QAOA

QLLM is a research framework that uses LLMs to discover interpretable structural patterns in the QAOA / Pauli-correlation design space. It selects correlation orders, allocates Pauli strings, and shapes QAOA ansätze for qubit-efficient optimization on NISQ hardware.

## Project Structure

```
QLLM/
├── src/
│   ├── config.py               # Central configuration
│   ├── graphs/
│   │   ├── generate_graphs.py   # Graph family generation
│   │   └── features.py          # Graph feature extraction
│   ├── qaoa/
│   │   ├── baseline_qaoa.py     # Uncompressed QAOA reference
│   │   └── pce_qaoa_baseline.py # PCE-based QAOA circuits
│   ├── pce/
│   │   └── manual_pce.py        # Hand-crafted Pauli-correlation encoding
│   ├── llm/
│   │   ├── schema.py            # LLM input/output schemas
│   │   ├── client.py            # LLM client wrapper
│   │   └── prompts/
│   │       └── qaoa_pce_prompt.txt
│   ├── pipeline/
│   │   ├── llm_guided_pce.py    # LLM-guided PCE pipeline
│   │   └── rule_based_pce.py    # Rule-based PCE pipeline
│   ├── rules/
│   │   └── encoder_rules.py     # Design rule engine
│   ├── analysis/
│   │   ├── aggregate_results.py # Experiment data aggregation
│   │   ├── patterns.py          # Pattern mining & analysis
│   │   ├── plots.py             # Visualization
│   │   └── compare_methods.py   # Cross-method comparison
│   ├── experiments/
│   │   ├── suites/
│   │   │   └── maxcut_suites.py
│   │   └── nisq_runs.py
│   ├── problems/
│   │   ├── labs.py
│   │   └── budget_constrained.py
│   └── infra/
│       ├── simulator.py         # Simulation backend wrapper
│       ├── logger.py            # Experiment logging
│       ├── experiment_tracker.py # Versioning & reproducibility
│       └── hardware_mapping.py  # NISQ device mapping
├── data/
│   ├── graphs/                  # Generated graph instances
│   ├── features/                # Graph features
│   ├── pce/                     # Pauli encoding specs
│   └── experiments/             # Experiment results
├── docs/
│   ├── api_reference.md
│   ├── pattern_catalog.md
│   └── paper_outline.md
├── output/
│   └── plots/
├── requirements.txt
└── README.md
```

## LLM Models

### Primary (L4 24 GB)
- **Default:** `Qwen/Qwen2.5-32B-Instruct` — main quantum design LLM
- **Alternative:** `google/gemma-2-27b-it`

### Utility (T4 16 GB)
- **Default:** `meta-llama/Llama-3.1-8B-Instruct` — fast iteration & dev tasks
- **Alternative:** `Qwen/Qwen2.5-7B-Instruct`

## Setup

```bash
git clone https://github.com/your-org/QLLM
cd QLLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Configure models and paths in `src/config.py`.

## Phases

| Phase | Description |
|-------|-------------|
| 1 | Baseline QAOA & PCE infrastructure |
| 2 | LLM interface & orchestration pipeline |
| 3 | Pattern discovery & analysis |
| 4 | Formalized design rules & heuristics |
| 5 | Generalization & NISQ deployment |

## License

MIT — see [LICENSE](LICENSE).
