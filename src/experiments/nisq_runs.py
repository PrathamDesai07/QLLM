"""NISQ device simulation — noisy simulator runs comparing baseline QAOA with
PCE-encoded circuits under realistic hardware noise models.

Demonstrates the key advantage of PCE for near-term hardware: compressed circuits
use dramatically fewer qubits and gates, reducing exposure to noise and enabling
fits on small, low-topology devices.

Usage
-----
    python -m src.experiments.nisq_runs --status
    python -m src.experiments.nisq_runs --graph-id erdos_renyi_8_0 --device ibm_eagle
    python -m src.experiments.nisq_runs --batch --device ibm_eagle --max-graphs 5
    python -m src.experiments.nisq_runs --analyze
"""

import json
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize
from qiskit_aer import AerSimulator

from config import GRAPHS_DIR, EXPERIMENTS_DIR, QAOA, BACKEND
from infra.experiment_tracker import append_index, _git_hash
from infra.simulator import compute_expectation
from infra.hardware_mapping import build_noise_model, DEVICES
from pce.manual_pce import encode_graph, build_pce_hamiltonian
from qaoa.pce_qaoa_baseline import pce_qaoa_ansatz
from qaoa.baseline_qaoa import maxcut_hamiltonian, qaoa_ansatz, brute_force_maxcut


NISQ_DIR = EXPERIMENTS_DIR / "nisq"
NISQ_DIR.mkdir(parents=True, exist_ok=True)

_SIMULATOR_CACHE: dict[str, AerSimulator] = {}


def _get_simulator(device_name: str) -> AerSimulator:
    """Create or retrieve a cached AerSimulator with the device's noise model."""
    key = device_name
    if key not in _SIMULATOR_CACHE:
        noise_model = build_noise_model(device_name) if device_name != "ideal" else None
        _SIMULATOR_CACHE[key] = AerSimulator(
            noise_model=noise_model,
            method="automatic",
            shots=BACKEND.get("shots", 1024),
            seed_simulator=QAOA["seed"],
        )
    return _SIMULATOR_CACHE[key]


def noisy_expectation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    params: list[float],
    device_name: str = "ibm_eagle",
) -> float:
    """Compute ⟨ψ|H|ψ⟩ under noisy simulation using AerSimulator's
    ``save_expectation_value``.  The circuit is first transpiled to
    the simulator's native basis gates (PauliEvolutionGate is not
    natively supported by AerSimulator)."""
    simulator = _get_simulator(device_name)
    bound = circuit.assign_parameters(params)

    # Transpile to basis gates (AerSimulator doesn't support PauliEvolutionGate)
    basis_gates = ["id", "sx", "x", "rz", "cx", "measure", "reset"]
    t_circ = transpile(bound, basis_gates=basis_gates, optimization_level=0)

    # Attach expectation value computation
    t_circ.save_expectation_value(observable, list(range(t_circ.num_qubits)))

    job = simulator.run(t_circ)
    return float(job.result().data()["expectation_value"])


def run_nisq(
    graph_id: str,
    device_name: str = "ibm_eagle",
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run noisy QAOA on a graph using baseline and PCE methods.

    Compares baseline QAOA, PCE k=1, and PCE k=2 under noise.
    """
    p = p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]
    rng = np.random.default_rng(seed)

    graph_path = GRAPHS_DIR / f"{graph_id}.json"
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})
    family = meta.get("family", "unknown")
    n = g.number_of_nodes()

    device_qubits = DEVICES.get(device_name, {}).get("n_qubits", 0)
    hamiltonian, num_edges = maxcut_hamiltonian(g)
    optimal_maxcut = brute_force_maxcut(g)

    results: dict[str, Any] = {
        "graph_id": graph_id,
        "family": family,
        "num_nodes": n,
        "num_edges": g.number_of_edges(),
        "device_name": device_name,
        "device_qubits": device_qubits,
        "methods": {},
    }

    # ── Baseline QAOA under noise ───────────────────────────────────
    if n <= device_qubits:
        if verbose:
            print(f"  [{graph_id}] Baseline QAOA under {device_name} noise …")
        try:
            circuit, param_list = qaoa_ansatz(g, p)
            x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

            def noisy_obj_fn(params):
                return noisy_expectation(circuit, hamiltonian, params.tolist(),
                                          device_name)

            result = minimize(noisy_obj_fn, x0, method=opt,
                              options={"maxiter": iters, "disp": False})
            noisy_energy = noisy_expectation(
                circuit, hamiltonian, result.x.tolist(), device_name
            )
            ideal_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())
            mc_val = (num_edges - noisy_energy) / 2.0
            ideal_mc = (num_edges - ideal_energy) / 2.0

            results["methods"]["baseline_qaoa"] = {
                "method": "baseline_qaoa",
                "num_physical_qubits": n, "compression_ratio": 1.0,
                "noisy_energy": round(float(noisy_energy), 6),
                "noisy_approximation_ratio": round(
                    mc_val / optimal_maxcut, 6) if optimal_maxcut > 0 else 0,
                "ideal_energy_at_params": round(float(ideal_energy), 6),
                "ideal_approx_at_params": round(
                    ideal_mc / optimal_maxcut, 6) if optimal_maxcut > 0 else 0,
                "success": bool(result.success), "nfeval": getattr(result, "nfev", None),
            }
        except Exception as e:
            if verbose:
                print(f"    ERROR: {e}")
    else:
        if verbose:
            print(f"  [{graph_id}] Baseline QAOA skipped (n={n} > device={device_qubits})")

    # ── PCE methods ─────────────────────────────────────────────────
    for k_val in [1, 2]:
        if verbose:
            print(f"  [{graph_id}] PCE k={k_val} under {device_name} noise …")
        try:
            encoding = encode_graph(g, k=k_val)
            m = encoding["num_physical_qubits"]
            if m > device_qubits:
                if verbose:
                    print(f"    Skipped (m={m} > device={device_qubits})")
                continue

            pce_ham = build_pce_hamiltonian(g, encoding)
            circuit, param_list = pce_qaoa_ansatz(pce_ham, m, p)
            x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

            def noisy_obj_fn_pce(params):
                return noisy_expectation(circuit, pce_ham, params.tolist(), device_name)

            result = minimize(noisy_obj_fn_pce, x0, method=opt,
                              options={"maxiter": iters, "disp": False})
            noisy_energy = noisy_expectation(
                circuit, pce_ham, result.x.tolist(), device_name
            )
            ideal_energy = compute_expectation(circuit, pce_ham, result.x.tolist())
            mc_val = (num_edges - noisy_energy) / 2.0
            ideal_mc = (num_edges - ideal_energy) / 2.0

            results["methods"][f"pce_baseline_k{k_val}"] = {
                "method": f"pce_baseline_k{k_val}", "k": k_val,
                "num_physical_qubits": m,
                "compression_ratio": encoding.get("compression_ratio", round(n / m, 2)),
                "noisy_energy": round(float(noisy_energy), 6),
                "noisy_approximation_ratio": round(
                    mc_val / optimal_maxcut, 6) if optimal_maxcut > 0 else 0,
                "ideal_energy_at_params": round(float(ideal_energy), 6),
                "ideal_approx_at_params": round(
                    ideal_mc / optimal_maxcut, 6) if optimal_maxcut > 0 else 0,
                "success": bool(result.success), "nfeval": getattr(result, "nfev", None),
            }
        except Exception as e:
            if verbose:
                print(f"    ERROR (PCE k={k_val}): {e}")

    # ── Save ────────────────────────────────────────────────────────
    method_dir = NISQ_DIR / device_name
    method_dir.mkdir(parents=True, exist_ok=True)
    path = method_dir / f"{graph_id}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    from datetime import datetime, timezone
    for method_key, method_res in results.get("methods", {}).items():
        append_index({
            "graph_id": graph_id, "family": family,
            "method": f"nisq_{device_name}_{method_key}",
            "model_name": "", "prompt_file": "", "prompt_hash": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "k": str(method_res.get("k", "")),
            "num_physical_qubits": str(method_res.get("num_physical_qubits", "")),
            "compression_ratio": str(method_res.get("compression_ratio", "")),
            "optimal_energy": str(method_res.get("noisy_energy", "")),
            "approximation_ratio": str(method_res.get("noisy_approximation_ratio", "")),
            "gradient_norm_at_opt": "",
            "success": str(method_res.get("success", "")),
            "convergence_iters": "", "duration_seconds": "",
            "llm_tags": "", "llm_reasoning": "",
            "git_commit": _git_hash(),
        })

    return results


# ── Batch ─────────────────────────────────────────────────────────────

def run_nisq_batch(
    device_name: str = "ibm_eagle",
    families: list[str] | None = None,
    max_graphs: int | None = None,
    skip_existing: bool = True,
    **kwargs,
) -> list[dict[str, Any]]:
    """Run NISQ experiments on a batch of graphs."""
    results = []
    existing: set[str] = set()
    nisq_device_dir = NISQ_DIR / device_name
    if skip_existing and nisq_device_dir.exists():
        existing = {f.stem for f in nisq_device_dir.glob("*.json")}

    graph_paths = sorted(GRAPHS_DIR.glob("*.json"))
    if families:
        graph_paths = [p for p in graph_paths
                       if _get_family(p) in families]
    if max_graphs is not None:
        graph_paths = graph_paths[:max_graphs]

    for gpath in graph_paths:
        gid = gpath.stem
        if skip_existing and gid in existing:
            continue
        res = run_nisq(gid, device_name=device_name, **kwargs)
        results.append(res)

    return results


def _get_family(path: Path) -> str:
    try:
        with open(path) as f:
            return json.load(f).get("metadata", {}).get("family", "unknown")
    except Exception:
        return "unknown"


# ── Analysis ──────────────────────────────────────────────────────────

def analyze_nisq_results() -> dict[str, Any]:
    """Aggregate all NISQ results and produce a comparison summary."""
    import pandas as pd

    records: list[dict[str, Any]] = []
    for device_dir in NISQ_DIR.iterdir():
        if not device_dir.is_dir():
            continue
        device_name = device_dir.name
        for f in device_dir.glob("*.json"):
            with open(f) as fh:
                data = json.load(fh)
            for method_key, res in data.get("methods", {}).items():
                records.append({
                    "graph_id": data["graph_id"],
                    "family": data.get("family", "unknown"),
                    "num_nodes": data.get("num_nodes"),
                    "device": device_name,
                    "method": method_key,
                    "k": res.get("k"),
                    "num_physical_qubits": res.get("num_physical_qubits"),
                    "compression_ratio": res.get("compression_ratio"),
                    "noisy_approx": res.get("noisy_approximation_ratio"),
                    "ideal_approx": res.get("ideal_approx_at_params"),
                    "success": res.get("success"),
                })

    if not records:
        return {"error": "No NISQ results found. Run --batch first."}

    df = pd.DataFrame(records)
    df["noise_impact"] = df["noisy_approx"] - df["ideal_approx"]

    analysis: dict[str, Any] = {}
    analysis["total_runs"] = len(df)
    analysis["devices"] = df["device"].unique().tolist()

    summary = df.groupby("method").agg(
        n=("graph_id", "count"),
        mean_noisy_approx=("noisy_approx", "mean"),
        mean_ideal_approx=("ideal_approx", "mean"),
        mean_noise_impact=("noise_impact", "mean"),
        mean_compression=("compression_ratio", "mean"),
    ).to_dict("index")

    analysis["method_summary"] = {
        m: {
            "n": int(v["n"]),
            "mean_noisy_approx": round(float(v["mean_noisy_approx"]), 4),
            "mean_ideal_approx": round(float(v["mean_ideal_approx"]), 4),
            "mean_noise_impact": round(float(v["mean_noise_impact"]), 4),
            "mean_compression": round(float(v["mean_compression"]), 2),
        }
        for m, v in summary.items()
    }

    for m, v in analysis["method_summary"].items():
        v["noise_degradation_pct"] = round(
            abs(v["mean_noise_impact"]) / max(abs(v["mean_ideal_approx"]), 0.001) * 100, 1
        ) if v["mean_ideal_approx"] else 0

    from config import DATA_DIR
    analysis_dir = DATA_DIR / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    path = analysis_dir / "nisq_analysis.json"
    with open(path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"  Wrote {path}")

    return analysis


def print_nisq_status() -> None:
    """Print a completeness report for NISQ experiments."""
    print()
    print(f"  NISQ Experiment Status")
    print(f"  {'─' * 50}")
    all_graphs = {f.stem for f in GRAPHS_DIR.glob("*.json")}
    for device_dir in sorted(NISQ_DIR.iterdir()):
        if not device_dir.is_dir():
            continue
        done = {f.stem for f in device_dir.glob("*.json")}
        print(f"  {device_dir.name:20s}: {len(done & all_graphs):3d}/{len(all_graphs)}")
    total_done = sum(
        len({f.stem for f in d.glob("*.json")} & {f.stem for f in GRAPHS_DIR.glob("*.json")})
        for d in NISQ_DIR.iterdir() if d.is_dir()
    )
    print(f"  {'TOTAL':20s}: {total_done}")
    print()


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="NISQ device simulation for QLLM"
    )
    parser.add_argument("--graph-id", type=str, default=None)
    parser.add_argument("--device", type=str, default="ibm_eagle",
                        choices=list(DEVICES))
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--analyze", action="store_true")
    parser.add_argument("--p", type=int, default=None)

    args = parser.parse_args()

    if args.status:
        print_nisq_status()
    elif args.analyze:
        analyze_nisq_results()
    elif args.graph_id:
        run_nisq(args.graph_id, device_name=args.device, p=args.p)
    elif args.batch:
        run_nisq_batch(
            device_name=args.device,
            families=args.families,
            max_graphs=args.max_graphs,
            p=args.p,
        )
    else:
        parser.print_help()
