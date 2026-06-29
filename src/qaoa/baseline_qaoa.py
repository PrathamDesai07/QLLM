"""Uncompressed baseline QAOA for MaxCut — 1:1 variable-to-qubit mapping."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from config import QAOA, BACKEND
from infra.logger import ExperimentRecord, timer
from infra.simulator import compute_expectation, estimate_gradient


# ── Hamiltonian ────────────────────────────────────────────────────────

def maxcut_hamiltonian(g: nx.Graph) -> tuple[SparsePauliOp, float]:
    """Return (H_C, shift) so C = (|E| - H_C) / 2."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    pauli_terms = []
    coeffs = []
    for i, j in g.edges():
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        pauli_terms.append("".join(label))
        coeffs.append(1.0)
    return SparsePauliOp(pauli_terms, coeffs), m


def maxcut_value_from_energy(energy: float, num_edges: int) -> float:
    """Convert Ising energy <H_C> to MaxCut value C = (|E| - <H_C>) / 2."""
    return (num_edges - energy) / 2.0


def brute_force_maxcut(g: nx.Graph) -> int:
    """Compute optimal MaxCut value via brute force (n ≤ 20)."""
    n = g.number_of_nodes()
    edges = list(g.edges())
    best = 0
    for mask in range(1 << n):
        cut = 0
        for i, j in edges:
            if ((mask >> i) & 1) != ((mask >> j) & 1):
                cut += 1
        if cut > best:
            best = cut
    return best


# ── Ansatz ─────────────────────────────────────────────────────────────

def qaoa_ansatz(g: nx.Graph, p: int) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build the QAOA ansatz for MaxCut on graph *g* with *p* layers.

    Returns (circuit, param_list) where param_list has 2p parameters:
    [gamma_0, beta_0, ..., gamma_{p-1}, beta_{p-1}].
    """
    n = g.number_of_nodes()
    flat_params: list[Parameter] = []
    for i in range(p):
        flat_params.append(Parameter(f"gamma_{i}"))
        flat_params.append(Parameter(f"beta_{i}"))
    # regroup for layer-by-layer access
    params = [(flat_params[2 * i], flat_params[2 * i + 1]) for i in range(p)]

    qc = QuantumCircuit(n)
    # Initial state |+>^n
    qc.h(range(n))

    for layer in range(p):
        gamma = params[layer][0]
        beta = params[layer][1]

        # Cost layer: exp(-i * gamma * H_C)
        for i, j in g.edges():
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)

        # Mixer layer: exp(-i * beta * sum X_i)
        qc.rx(2 * beta, range(n))

    return qc, flat_params


# ── Classical Optimisation ─────────────────────────────────────────────

def _objective_fn(params, circuit, hamiltonian, num_edges=None):
    """Compute -MaxCut value for minimisation."""
    energy = compute_expectation(circuit, hamiltonian, params.tolist())
    if num_edges is not None:
        return -maxcut_value_from_energy(energy, num_edges)
    return energy


def run_qaoa(
    g: nx.Graph,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict:
    """Run baseline uncompressed QAOA on graph *g*.

    Returns a dictionary with keys:
      optimal_energy, optimal_params, optimal_maxcut, approximation_ratio,
      gradient_norm, convergence_iters, nfeval, success.
    """
    p = p or QAOA["p"]
    optimizer = optimizer or QAOA["optimizer"]
    max_iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]

    hamiltonian, num_edges = maxcut_hamiltonian(g)
    optimal_maxcut = brute_force_maxcut(g)

    circuit, param_list = qaoa_ansatz(g, p)
    n_params = len(param_list)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=n_params)

    # minimise
    result = minimize(
        _objective_fn,
        x0,
        args=(circuit, hamiltonian, num_edges),
        method=optimizer,
        options={"maxiter": max_iters, "disp": False},
    )

    energy = compute_expectation(circuit, hamiltonian, result.x.tolist())
    mc_value = maxcut_value_from_energy(energy, num_edges)
    approx_ratio = mc_value / optimal_maxcut if optimal_maxcut > 0 else 0.0

    grad = estimate_gradient(circuit, hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    return {
        "optimal_energy": float(energy),
        "optimal_params": result.x.tolist(),
        "optimal_maxcut": float(mc_value),
        "optimal_maxcut_exact": float(optimal_maxcut),
        "approximation_ratio": round(approx_ratio, 6),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "num_edges": int(num_edges),
    }


# ── Full Runner ────────────────────────────────────────────────────────

@timer
def run_baseline_qaoa(
    graph_path: Path,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    output_dir: Path | None = None,
) -> ExperimentRecord:
    """Load a graph from *graph_path*, run baseline QAOA, log and return record."""
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})

    result = run_qaoa(g, p=p, optimizer=optimizer, max_iters=max_iters)
    num_edges = g.number_of_edges()

    record = ExperimentRecord(
        graph_id=meta.get("graph_id", graph_path.stem),
        family=meta.get("family", "unknown"),
        num_nodes=g.number_of_nodes(),
        num_edges=num_edges,
        method="baseline_qaoa",
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
            "optimal_maxcut": result["optimal_maxcut"],
            "optimal_maxcut_exact": result["optimal_maxcut_exact"],
            "nfeval": result["nfeval"],
        },
    )
    record.save(directory=output_dir)
    return record


# ── Batch Runner ───────────────────────────────────────────────────────

def run_baseline_batch(
    graph_dir: Path | None = None,
    output_dir: Path | None = None,
    families: list[str] | None = None,
    p: int | None = None,
) -> list[ExperimentRecord]:
    """Run baseline QAOA on all graphs (optionally filtered by family)."""
    from config import EXPERIMENTS_DIR, GRAPHS_DIR

    graph_dir = graph_dir or GRAPHS_DIR
    output_dir = output_dir or EXPERIMENTS_DIR

    records = []
    for gpath in sorted(graph_dir.glob("*.json")):
        if families:
            with open(gpath) as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("family") not in families:
                continue
        rec = run_baseline_qaoa(gpath, p=p, output_dir=output_dir)
        records.append(rec)
        print(f"  {rec.graph_id}: ratio={rec.approximation_ratio}, "
              f"energy={rec.optimal_energy:.4f}, success={rec.success}")
    return records


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline QAOA")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--p", type=int, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--graph-id", type=str, default=None)
    args = parser.parse_args()

    if args.graph_id:
        from config import GRAPHS_DIR
        gdir = Path(args.graph_dir) if args.graph_dir else GRAPHS_DIR
        odir = Path(args.output_dir) if args.output_dir else None
        rec = run_baseline_qaoa(
            gdir / f"{args.graph_id}.json",
            p=args.p, optimizer=args.optimizer,
            max_iters=args.max_iters, output_dir=odir,
        )
        print(f"[{rec.graph_id}] ratio={rec.approximation_ratio}, "
              f"energy={rec.optimal_energy:.4f}, "
              f"duration={rec.duration_seconds:.2f}s")
    else:
        run_baseline_batch(
            graph_dir=Path(args.graph_dir) if args.graph_dir else None,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            families=args.families, p=args.p,
        )
