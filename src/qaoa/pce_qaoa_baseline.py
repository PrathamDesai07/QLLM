"""PCE-encoded QAOA — QAOA circuits built using a Pauli-correlation encoding."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from config import QAOA
from infra.experiment_tracker import append_index, _git_hash
from infra.logger import ExperimentRecord, timer
from infra.simulator import compute_expectation, estimate_gradient
from pce.manual_pce import build_pce_hamiltonian
from qaoa.baseline_qaoa import brute_force_maxcut


# ── PCE QAOA Ansatz ────────────────────────────────────────────────────

def pce_qaoa_ansatz(
    hamiltonian: SparsePauliOp,
    num_physical_qubits: int,
    p: int,
) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build PCE-encoded QAOA ansatz.

    Parameters
    ----------
    hamiltonian : SparsePauliOp
        The PCE-encoded cost Hamiltonian on *num_physical_qubits* qubits.
    num_physical_qubits : int
        Number of physical qubits (m).
    p : int
        Number of QAOA layers.

    Returns
    -------
    (circuit, param_list)
        *param_list* has 2p Parameters: [gamma_0, beta_0, ...].
    """
    flat_params: list[Parameter] = []
    for i in range(p):
        flat_params.append(Parameter(f"gamma_{i}"))
        flat_params.append(Parameter(f"beta_{i}"))
    params = [(flat_params[2 * i], flat_params[2 * i + 1]) for i in range(p)]

    qc = QuantumCircuit(num_physical_qubits)
    # Initial state |+>^m
    qc.h(range(num_physical_qubits))

    for layer in range(p):
        gamma, beta = params[layer]

        # Cost layer: exp(-i * gamma * H_PCE)
        evol = PauliEvolutionGate(hamiltonian, time=gamma)
        qc.append(evol, range(num_physical_qubits))

        # Mixer layer: exp(-i * beta * sum X)
        qc.rx(2 * beta, range(num_physical_qubits))

    return qc, flat_params


# ── Objective & Runner ─────────────────────────────────────────────────

def _objective_fn(params, circuit, hamiltonian):
    """Return ⟨ψ|H_PCE|ψ⟩ to be minimised."""
    return compute_expectation(circuit, hamiltonian, params.tolist())


def run_pce_qaoa(
    g: nx.Graph,
    encoding: dict,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run PCE-encoded QAOA on graph *g* using the given encoding.

    Returns dict with keys: optimal_energy, optimal_params,
    approximation_ratio, etc.
    """
    p = p or QAOA["p"]
    optimizer = optimizer or QAOA["optimizer"]
    max_iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]
    m = encoding["num_physical_qubits"]

    hamiltonian = build_pce_hamiltonian(g, encoding)
    num_edges = g.number_of_edges()

    circuit, param_list = pce_qaoa_ansatz(hamiltonian, m, p)
    n_params = len(param_list)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=n_params)

    result = minimize(
        _objective_fn,
        x0,
        args=(circuit, hamiltonian),
        method=optimizer,
        options={"maxiter": max_iters, "disp": False},
    )

    final_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())

    # Approx ratio: for PCE, the cut value follows same formula
    # C = (|E| - <H_PCE>) / 2.  Compare to brute-force optimal MaxCut.
    mc_value = (num_edges - final_energy) / 2.0
    optimal_maxcut = brute_force_maxcut(g)
    approx_ratio = mc_value / optimal_maxcut if optimal_maxcut > 0 else 0.0

    grad = estimate_gradient(circuit, hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    return {
        "optimal_energy": float(final_energy),
        "optimal_params": result.x.tolist(),
        "optimal_pce_value": float(mc_value),
        "optimal_maxcut_exact": float(optimal_maxcut),
        "approximation_ratio": round(approx_ratio, 6),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "num_physical_qubits": m,
        "num_edges": num_edges,
        "num_variables": g.number_of_nodes(),
        "compression_ratio": encoding.get("compression_ratio", 1.0),
    }


# ── Full Runner ────────────────────────────────────────────────────────

@timer
def run_pce_baseline(
    graph_path: Path,
    encoding: dict | None = None,
    encoding_path: Path | None = None,
    k: int | None = None,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    output_dir: Path | None = None,
) -> ExperimentRecord:
    """Load graph + encoding, run PCE QAOA, log result."""
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})

    if encoding is None and encoding_path is not None:
        with open(encoding_path) as f:
            encoding = json.load(f)
    elif encoding is None:
        from pce.manual_pce import encode_graph
        encoding = encode_graph(g, k=k)
        encoding["graph_id"] = meta.get("graph_id", graph_path.stem)

    result = run_pce_qaoa(g, encoding, p=p, optimizer=optimizer,
                          max_iters=max_iters)

    record = ExperimentRecord(
        graph_id=encoding.get("graph_id", graph_path.stem),
        family=meta.get("family", "unknown"),
        num_nodes=g.number_of_nodes(),
        num_edges=g.number_of_edges(),
        method="pce_baseline",
        qaoa_p=p or QAOA["p"],
        optimizer=optimizer or QAOA["optimizer"],
        max_iters=max_iters or QAOA["max_iters"],
        optimal_energy=result["optimal_energy"],
        optimal_params=result["optimal_params"],
        approximation_ratio=result["approximation_ratio"],
        gradient_norm_at_opt=result["gradient_norm_at_opt"],
        convergence_iters=result["convergence_iters"],
        success=result["success"],
        params=meta.get("params", {}),
        extra={
            "k": encoding.get("k"),
            "strategy": encoding.get("strategy"),
            "num_physical_qubits": result["num_physical_qubits"],
            "num_variables": result["num_variables"],
            "compression_ratio": result["compression_ratio"],
            "optimal_pce_value": result["optimal_pce_value"],
            "optimal_maxcut_exact": result["optimal_maxcut_exact"],
            "nfeval": result["nfeval"],
        },
    )
    record.save(directory=output_dir)

    # ── Append to central index ─────────────────────────────────────
    k = encoding.get("k")
    m_q = result["num_physical_qubits"]
    comp = result["compression_ratio"]
    append_index({
        "graph_id": record.graph_id,
        "family": record.family,
        "method": "pce_baseline",
        "model_name": "",
        "prompt_file": "",
        "prompt_hash": "",
        "timestamp": record.timestamp,
        "k": str(k) if k is not None else "",
        "num_physical_qubits": str(m_q),
        "compression_ratio": str(comp),
        "optimal_energy": str(record.optimal_energy),
        "approximation_ratio": str(record.approximation_ratio),
        "gradient_norm_at_opt": str(record.gradient_norm_at_opt),
        "success": str(record.success),
        "convergence_iters": str(record.convergence_iters) if record.convergence_iters is not None else "",
        "duration_seconds": str(record.duration_seconds) if record.duration_seconds is not None else "",
        "llm_tags": "",
        "llm_reasoning": "",
        "git_commit": _git_hash(),
    })

    return record


# ── Batch Runner ───────────────────────────────────────────────────────

def run_pce_batch(
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
    families: list[str] | None = None,
    k: int | None = None,
    p: int | None = None,
) -> list[ExperimentRecord]:
    """Run PCE QAOA on all graphs (optionally filtered by family)."""
    from config import EXPERIMENTS_DIR, GRAPHS_DIR, PCE_DIR

    graph_dir = graph_dir or GRAPHS_DIR
    output_dir = output_dir or EXPERIMENTS_DIR

    records = []
    for gpath in sorted(graph_dir.glob("*.json")):
        if families:
            with open(gpath) as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("family") not in families:
                continue
        # generate encoding on the fly
        rec = run_pce_baseline(gpath, k=k, p=p, output_dir=output_dir)
        records.append(rec)
        print(f"  {rec.graph_id}: ratio={rec.approximation_ratio}, "
              f"energy={rec.optimal_energy:.4f}, "
              f"qubits={rec.extra.get('num_physical_qubits')}, "
              f"compression={rec.extra.get('compression_ratio')}x")
    return records


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run PCE-encoded QAOA baseline")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--graph-id", type=str, default=None)
    args = parser.parse_args()

    from config import GRAPHS_DIR
    gdir = Path(args.graph_dir) if args.graph_dir else GRAPHS_DIR
    odir = Path(args.output_dir) if args.output_dir else None

    if args.graph_id:
        rec = run_pce_baseline(
            gdir / f"{args.graph_id}.json",
            k=args.k, p=args.p, optimizer=args.optimizer,
            max_iters=args.max_iters, output_dir=odir,
        )
        print(f"[{rec.graph_id}] ratio={rec.approximation_ratio}, "
              f"energy={rec.optimal_energy:.4f}, "
              f"qubits={rec.extra.get('num_physical_qubits')}, "
              f"duration={rec.duration_seconds:.2f}s")
    else:
        run_pce_batch(
            graph_dir=gdir, output_dir=odir,
            families=args.families, k=args.k, p=args.p,
        )
