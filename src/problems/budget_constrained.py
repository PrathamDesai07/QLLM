"""Budget-constrained optimisation problems for QAOA with PCE support.

Implements a *knapsack-style* problem: select a subset of items to maximise
total value subject to a budget constraint on total cost.  This is a canonical
constrained binary optimisation problem solvable via QAOA.

The Hamiltonian encodes:
  - Objective: maximise Σ v_i x_i  (x_i ∈ {0, 1})
  - Constraint: Σ c_i x_i ≤ B, enforced via a quadratic penalty
    P · (Σ c_i x_i − B)²

After converting to Ising variables (s_i = 2x_i − 1 ∈ {-1, +1}), the
cost function becomes quadratic in Z operators and can be mapped
directly to a Pauli Hamiltonian.

Functions
---------
- ``budget_hamiltonian`` — Build the uncompressed Ising Hamiltonian.
- ``brute_force_optimum`` — Exact optimum for small instances.
- ``generate_instance`` — Create a random budget-constrained instance.
- ``run_budget_qaoa`` — Run uncompressed QAOA.
- ``run_budget_pce`` — Run PCE-encoded QAOA.
- ``run_budget_rule`` — Run rule-based PCE QAOA.
- CLI for single or batch runs.

Usage
-----
    python -m src.problems.budget_constrained --n 6 --method baseline
    python -m src.problems.budget_constrained --n 8 --method pce --k 2
    python -m src.problems.budget_constrained --batch --n-values 4 6 8
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
from qiskit.circuit import Parameter
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from config import EXPERIMENTS_DIR, QAOA
from infra.experiment_tracker import append_index, _git_hash
from infra.simulator import compute_expectation, estimate_gradient
from pce.manual_pce import encode_graph, _multiply_pauli_strings
from qaoa.pce_qaoa_baseline import pce_qaoa_ansatz


# ── Instance generation ────────────────────────────────────────────────

def generate_instance(
    n: int,
    value_range: tuple[int, int] = (1, 20),
    cost_range: tuple[int, int] = (1, 10),
    budget_ratio: float = 0.5,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate a random budget-constrained optimisation instance.

    Parameters
    ----------
    n : int
        Number of items.
    value_range : (int, int)
        Range of random item values.
    cost_range : (int, int)
        Range of random item costs.
    budget_ratio : float
        Budget as fraction of total cost.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: n, values (list), costs (list), budget (float),
    instance_id (str), penalty_weight (float).
    """
    rng = np.random.default_rng(seed)
    values = rng.integers(value_range[0], value_range[1] + 1, size=n).tolist()
    costs = rng.integers(cost_range[0], cost_range[1] + 1, size=n).tolist()
    total_cost = sum(costs)
    budget = int(total_cost * budget_ratio)

    # Penalty weight: must be large enough to enforce constraint.
    # Set penalty > max_total_value so constraint violation always dominates.
    max_value = sum(values)  # best unconstrained objective
    penalty_weight = float(max_value * 1.5 + 100)

    instance_id = f"budget_n{n}_seed{seed if seed is not None else 0}"

    return {
        "n": n,
        "values": values,
        "costs": costs,
        "budget": budget,
        "total_cost": total_cost,
        "penalty_weight": penalty_weight,
        "instance_id": instance_id,
    }


def brute_force_optimum(instance: dict[str, Any]) -> dict[str, Any]:
    """Brute-force optimal solution for small instances (n ≤ 20).

    Returns dict with best_value, best_selection (list of bool), energy.
    """
    n = instance["n"]
    values = instance["values"]
    costs = instance["costs"]
    budget = instance["budget"]

    best_value = -1
    best_mask = 0

    for mask in range(1 << n):
        total_value = 0
        total_cost = 0
        for i in range(n):
            if (mask >> i) & 1:
                total_value += values[i]
                total_cost += costs[i]
        if total_cost <= budget and total_value > best_value:
            best_value = total_value
            best_mask = mask

    best_selection = [bool((best_mask >> i) & 1) for i in range(n)]

    return {
        "n": n,
        "best_value": best_value,
        "best_selection": best_selection,
        "best_energy": -best_value,  # negative for minimisation
    }


# ── Hamiltonian building ──────────────────────────────────────────────

def _is_to_qubo(values: list[int], costs: list[int],
                budget: int, penalty: float) -> tuple[np.ndarray, np.ndarray]:
    """Convert budget-constrained problem to QUBO matrix.

    maximise   Σ v_i x_i
    subject to Σ c_i x_i ≤ B

    QUBO objective: minimi se  −Σ v_i x_i + P · (Σ c_i x_i − B)²

    The quadratic penalty expands to:
    P · (Σ c_i x_i)² − 2P·B·(Σ c_i x_i) + P·B²

    The QUBO matrix Q (size n×n) and linear vector h are such that:
    f(x) = Σ_i Σ_j Q_ij x_i x_j + Σ_i h_i x_i + const

    Returns (Q, h) for the quartic+linear terms.
    """
    n = len(values)
    Q = np.zeros((n, n), dtype=np.float64)
    h = np.zeros(n, dtype=np.float64)

    for i in range(n):
        h[i] -= values[i]  # objective term
        for j in range(n):
            Q[i, j] += penalty * costs[i] * costs[j]  # quadratic penalty
        h[i] -= 2.0 * penalty * budget * costs[i]  # linear penalty

    return Q, h


def _qubo_to_ising(Q: np.ndarray, h: np.ndarray,
                   n: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Convert QUBO (x_i ∈ {0,1}) to Ising (s_i ∈ {-1,+1}).

    x_i = (1 + s_i) / 2

    Returns (J, h_ising, offset) such that:
    H = Σ_i J_ij s_i s_j + Σ_i h_ising_i s_i + offset
    """
    J = np.zeros((n, n), dtype=np.float64)
    h_ising = np.zeros(n, dtype=np.float64)
    offset = 0.0

    for i in range(n):
        h_ising[i] += h[i] / 2.0
        for j in range(n):
            if i == j:
                h_ising[i] += Q[i, i] / 4.0
                offset += Q[i, i] / 4.0
            else:
                J[i, j] += Q[i, j] / 4.0
                h_ising[i] += Q[i, j] / 4.0
                h_ising[j] += Q[i, j] / 4.0
                offset += Q[i, j] / 4.0
        offset += h[i] / 2.0

    return J, h_ising, offset


def budget_hamiltonian(instance: dict[str, Any]) -> SparsePauliOp:
    """Build the Ising Hamiltonian for a budget-constrained instance.

    The Hamiltonian encodes the penalised objective as a ZZ + Z operator.
    """
    n = instance["n"]
    values = instance["values"]
    costs = instance["costs"]
    budget = instance["budget"]
    penalty = instance["penalty_weight"]

    Q, h_linear = _is_to_qubo(values, costs, budget, penalty)
    J, h_ising, _ = _qubo_to_ising(Q, h_linear, n)

    pauli_terms: list[str] = []
    coeffs: list[float] = []

    # ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                label = ["I"] * n
                label[i] = "Z"
                label[j] = "Z"
                pauli_terms.append("".join(label))
                coeffs.append(J[i, j])

    # Z terms
    for i in range(n):
        if abs(h_ising[i]) > 1e-12:
            label = ["I"] * n
            label[i] = "Z"
            pauli_terms.append("".join(label))
            coeffs.append(h_ising[i])

    if not pauli_terms:
        pauli_terms.append("I" * n)
        coeffs.append(0.0)

    return SparsePauliOp(pauli_terms, coeffs).simplify()


# ── QAOA Ansatz ───────────────────────────────────────────────────────

def _budget_qaoa_ansatz(
    hamiltonian: SparsePauliOp, n_qubits: int, p: int
) -> tuple["QuantumCircuit", list[Parameter]]:
    """Build a QAOA ansatz for the budget-constrained Ising Hamiltonian."""
    flat_params: list[Parameter] = []
    for i in range(p):
        flat_params.append(Parameter(f"gamma_{i}"))
        flat_params.append(Parameter(f"beta_{i}"))

    from qiskit import QuantumCircuit
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))

    for layer in range(p):
        gamma = flat_params[2 * layer]
        beta = flat_params[2 * layer + 1]

        evol = PauliEvolutionGate(hamiltonian, time=gamma)
        qc.append(evol, range(n_qubits))
        qc.rx(2 * beta, range(n_qubits))

    return qc, flat_params


# ── PCE Hamiltonian builder ───────────────────────────────────────────

def _build_pce_budget_hamiltonian(
    instance: dict[str, Any],
    encoding: dict[str, Any],
) -> SparsePauliOp:
    """Build the PCE-encoded budget Hamiltonian.

    Takes the Ising coefficients (J, h) from the uncompressed formulation
    and maps each Z_i operator through the encoding's Pauli strings.
    """
    n = instance["n"]
    values = instance["values"]
    costs = instance["costs"]
    budget = instance["budget"]
    penalty = instance["penalty_weight"]
    var_map = encoding["variable_to_pauli_map"]

    Q, h_linear = _is_to_qubo(values, costs, budget, penalty)
    J, h_ising, _ = _qubo_to_ising(Q, h_linear, n)

    pauli_terms: list[str] = []
    coeffs: list[float] = []

    # ZZ terms: Z_i Z_j → multiply pauli strings assigned to i and j
    for i in range(n):
        for j in range(i + 1, n):
            coeff = J[i, j]
            if abs(coeff) < 1e-12:
                continue
            s_i = var_map[str(i)]["pauli_string"]
            s_j = var_map[str(j)]["pauli_string"]
            result_str, phase = _multiply_pauli_strings(s_i, s_j)
            if phase != 0.0 and not all(c == "I" for c in result_str):
                pauli_terms.append(result_str)
                coeffs.append(coeff * phase)

    # Z terms: Z_i → the pauli string assigned to i
    m = encoding["num_physical_qubits"]
    for i in range(n):
        coeff = h_ising[i]
        if abs(coeff) < 1e-12:
            continue
        ps = var_map[str(i)]["pauli_string"]
        pauli_terms.append(ps)
        coeffs.append(coeff)

    if not pauli_terms:
        pauli_terms.append("I" * m)
        coeffs.append(0.0)

    return SparsePauliOp(pauli_terms, coeffs).simplify()


# ── Runners ───────────────────────────────────────────────────────────

def run_budget_qaoa(
    instance: dict[str, Any],
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run uncompressed QAOA on a budget-constrained instance."""
    n = instance["n"]
    p = p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]

    hamiltonian = budget_hamiltonian(instance)
    circuit, param_list = _budget_qaoa_ansatz(hamiltonian, n, p)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

    def obj_fn(params):
        return compute_expectation(circuit, hamiltonian, params.tolist())

    result = minimize(obj_fn, x0, method=opt,
                      options={"maxiter": iters, "disp": False})

    final_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())
    grad = estimate_gradient(circuit, hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    # Brute force for comparison
    bf = brute_force_optimum(instance)

    return {
        "instance_id": instance["instance_id"],
        "n": n,
        "optimal_energy": float(final_energy),
        "optimal_params": result.x.tolist(),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "method": "budget_baseline",
        "num_physical_qubits": n,
        "compression_ratio": 1.0,
        "brute_force_optimum": bf,
    }


def run_budget_pce(
    instance: dict[str, Any],
    k: int = 2,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run PCE-encoded QAOA on a budget-constrained instance."""
    n = instance["n"]
    p = p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]

    # Chain graph drives PCE variable assignment
    chain = nx.path_graph(n)
    encoding = encode_graph(chain, k=k)
    m = encoding["num_physical_qubits"]

    # Build PCE-encoded Hamiltonian
    pce_hamiltonian = _build_pce_budget_hamiltonian(instance, encoding)
    circuit, param_list = pce_qaoa_ansatz(pce_hamiltonian, m, p)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

    def obj_fn(params):
        return compute_expectation(circuit, pce_hamiltonian, params.tolist())

    result = minimize(obj_fn, x0, method=opt,
                      options={"maxiter": iters, "disp": False})

    final_energy = compute_expectation(circuit, pce_hamiltonian, result.x.tolist())
    grad = estimate_gradient(circuit, pce_hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    bf = brute_force_optimum(instance)

    return {
        "instance_id": instance["instance_id"],
        "n": n,
        "k": k,
        "num_physical_qubits": m,
        "compression_ratio": round(n / m, 2) if m else 1.0,
        "optimal_energy": float(final_energy),
        "optimal_params": result.x.tolist(),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "method": f"budget_pce_k{k}",
        "brute_force_optimum": bf,
    }


def run_budget_rule(
    instance: dict[str, Any],
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run rule-based PCE QAOA on a budget-constrained instance.

    The rule engine (encoder_rules) works on graph features, so we
    build a synthetic graph from the instance and pass it to the
    rules system, then use the recommended k for PCE encoding.
    """
    n = instance["n"]
    from rules.encoder_rules import recommend_encoding

    # Build a fully connected graph representing item relationships
    g = nx.complete_graph(n)
    from graphs.features import extract_all
    features = extract_all(g)
    features["graph_id"] = instance["instance_id"]
    features["family"] = "budget_constrained"

    rec = recommend_encoding(
        instance["instance_id"],
        features=features,
        graph=g,
    )
    k = rec["k_recommendation"]["k"]

    # Now run PCE with recommended k
    return run_budget_pce(instance, k=k, p=p, optimizer=optimizer,
                           max_iters=max_iters, seed=seed)


# ── Logging ───────────────────────────────────────────────────────────

def _log_budget_result(
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> None:
    """Save a budget result as JSON and append to index CSV."""
    output_dir = output_dir or EXPERIMENTS_DIR
    method = result.get("method", "budget_baseline")
    method_dir = output_dir / "budget" / method
    method_dir.mkdir(parents=True, exist_ok=True)

    gid = result.get("instance_id", f"budget_n{result['n']}")
    path = method_dir / f"{gid}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    from datetime import datetime, timezone
    append_index({
        "graph_id": gid,
        "family": "budget_constrained",
        "method": result.get("method", "budget_baseline"),
        "model_name": "",
        "prompt_file": "",
        "prompt_hash": "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "k": str(result.get("k", "")),
        "num_physical_qubits": str(result.get("num_physical_qubits", "")),
        "compression_ratio": str(result.get("compression_ratio", "")),
        "optimal_energy": str(result.get("optimal_energy", "")),
        "approximation_ratio": "",
        "gradient_norm_at_opt": str(result.get("gradient_norm_at_opt", "")),
        "success": str(result.get("success", "")),
        "convergence_iters": str(result.get("convergence_iters", "")),
        "duration_seconds": "",
        "llm_tags": "",
        "llm_reasoning": "",
        "git_commit": _git_hash(),
    })


# ── CLI ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Budget-constrained optimisation for QAOA/PCE"
    )
    parser.add_argument("--n", type=int, default=None, help="Number of items")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "pce", "rule"],
                        help="Encoding method")
    parser.add_argument("--k", type=int, default=None,
                        help="Correlation order for PCE")
    parser.add_argument("--p", type=int, default=None, help="QAOA depth")
    parser.add_argument("--seed", type=int, default=42,
                        help="Instance random seed")
    parser.add_argument("--batch", action="store_true",
                        help="Run multiple sizes")
    parser.add_argument("--n-values", type=int, nargs="*", default=[4, 6, 8],
                        help="Sizes for batch mode")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--show-instance", action="store_true",
                        help="Print instance details and exit")

    args = parser.parse_args()

    if args.batch:
        n_values = args.n_values
        print(f"Running budget batch for n={n_values} method={args.method} ...")
        for n in n_values:
            inst = generate_instance(n, seed=args.seed)
            if args.show_instance:
                print(json.dumps(inst, indent=2))
                continue

            if args.method == "baseline":
                res = run_budget_qaoa(inst, p=args.p)
            elif args.method == "rule":
                res = run_budget_rule(inst, p=args.p)
            else:
                k = args.k or 2
                res = run_budget_pce(inst, k=k, p=args.p)
            _log_budget_result(res, Path(args.output_dir) if args.output_dir else None)

            method = res.get("method", args.method)
            qubits = res.get("num_physical_qubits", n)
            comp = res.get("compression_ratio", 1.0)
            bf_energy = res.get("brute_force_optimum", {}).get("best_energy")
            print(f"  n={n:2d} {method:25s} energy={res['optimal_energy']:.4f} "
                  f"qubits={qubits} comp={comp:.2f}x "
                  f"grad={res['gradient_norm_at_opt']:.6f} "
                  f"bf_opt={bf_energy} ok={res['success']}")

    elif args.n:
        inst = generate_instance(args.n, seed=args.seed)
        if args.show_instance:
            print(json.dumps(inst, indent=2))
            sys.exit(0)

        if args.method == "baseline":
            res = run_budget_qaoa(inst, p=args.p)
        elif args.method == "rule":
            res = run_budget_rule(inst, p=args.p)
        else:
            k = args.k or 2
            res = run_budget_pce(inst, k=k, p=args.p)
        _log_budget_result(res, Path(args.output_dir) if args.output_dir else None)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()
