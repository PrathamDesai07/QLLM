"""Central configuration for QLLM."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
GRAPHS_DIR = DATA_DIR / "graphs"
FEATURES_DIR = DATA_DIR / "features"
PCE_DIR = DATA_DIR / "pce"
EXPERIMENTS_DIR = DATA_DIR / "experiments"
OUTPUT_DIR = PROJECT_ROOT / "output"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Ensure data directories exist
for _dir in [GRAPHS_DIR, FEATURES_DIR, PCE_DIR, EXPERIMENTS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Simulation Backend ─────────────────────────────────────────────────
BACKEND = {
    "name": "qasm_simulator",          # qasm_simulator | statevector_simulator | fake_backend
    "shots": 1024,
    "noise_model": None,               # None or NoiseModel instance
}

# ── QAOA Defaults ──────────────────────────────────────────────────────
QAOA = {
    "p": 1,                            # number of QAOA layers
    "optimizer": "COBYLA",             # COBYLA | SPSA | ADAM | …
    "max_iters": 200,
    "seed": 42,
}

# ── Graph Generation Defaults ──────────────────────────────────────────
GRAPH_DEFAULTS = {
    "families": ["erdos_renyi", "d_regular", "community", "bipartite"],
    "min_nodes": 8,
    "max_nodes": 20,
    "step": 4,
    "instances_per_size": 5,           # random seeds per (family, size)
}

# ── Pauli-Correlation Encoding Defaults ────────────────────────────────
PCE = {
    "default_k": 2,                    # compression order for manual PCE
    "max_k": 4,                        # maximum order to consider
}

# ── HuggingFace Token ──────────────────────────────────────────────────
HF_TOKEN_PATH = PROJECT_ROOT / "hf.txt"

def _load_hf_token() -> str | None:
    """Read the HuggingFace token from hf.txt, or return None."""
    try:
        if HF_TOKEN_PATH.exists():
            raw = HF_TOKEN_PATH.read_text().strip()
            if not raw:
                return None
            # Support both "hf_token: xxx" and bare "hf_xxx" formats
            if ":" in raw:
                raw = raw.split(":", 1)[1].strip()
            return raw if raw.startswith("hf_") else None
    except Exception:
        pass
    return None

HF_TOKEN = _load_hf_token()

# ── LLM Configuration ──────────────────────────────────────────────────
LLM_CONFIG = {
    "primary_model_name": "Qwen/Qwen2.5-32B-Instruct",
    "primary_model_hf_url": "https://huggingface.co/Qwen/Qwen2.5-32B-Instruct",
    "utility_model_name": "meta-llama/Llama-3.1-8B-Instruct",
    "utility_model_hf_url": "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
    "quantization": "q4_k_m",
    "temperature": 0.1,
    "max_tokens": 2048,
    "api_base": None,                  # set to local endpoint (vLLM, llama.cpp, etc.)
}
