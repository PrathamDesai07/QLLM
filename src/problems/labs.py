"""Low Autocorrelation Binary Sequence (LABS) problem for QAOA with PCE support.

The LABS problem seeks a binary sequence s_i ∈ {-1, +1} (i=0..N-1) that
minimises the autocorrelation energy:

    E = Σ_{k=1}^{N-1} C_k²,   C_k = Σ_{i=0}^{N-k-1} s_i · s_{i+k}

The *merit factor* F = N² / (2E) is the figure of merit (higher is better).

This module provides:
  - ``labs_hamiltonian`` — Ising Hamiltonian for the LABS problem on N variables.
  - ``brute_force_best_labs`` — exact optimum for small N (≤18).
  - ``run_labs_qaoa`` — run QAOA on a LABS problem instance.
  - ``run_labs_pce`` — run PCE-encoded QAOA on a LABS problem instance.
  - CLI to run single instances or batch.

The LABS Hamiltonian involves 4-body ZZ terms when expanded, making it
a natural candidate for PCE-based qubit compression.

Usage
-----
    python -m src.problems.labs --n 10 --method baseline
    python -m src.problems.labs --n 12 --method pce --k 2
    python -m src.problems.labs --batch --n-values 8 10 12
"""

import itertools
import json
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import networkx as nx
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from config import EXPERIMENTS_DIR, QAOA
from infra.logger import ExperimentRecord
from infra.experiment_tracker import append_index, _git_hash
from infra.simulator import compute_expectation, estimate_gradient
from pce.manual_pce import encode_graph
from qaoa.pce_qaoa_baseline import pce_qaoa_ansatz


# ── LABS Hamiltonian (uncompressed) ────────────────────────────────────

def labs_hamiltonian(n: int) -> SparsePauliOp:
    """Build the LABS Ising Hamiltonian for N variables.

    Variables are s_i ∈ {-1, +1}.  The Hamiltonian is the sum over
    shifts k=1..N-1 of (Σ_i s_i s_{i+k})², mapped to Pauli Z operators.

    For each shift k, the inner sum Σ_i Z_i Z_{i+k} is squared, producing
    4-body ZZ terms: (Z_a Z_{a+k})(Z_b Z_{b+k}) = Z_a Z_b Z_{a+k} Z_{b+k}.
    """
    if n < 2:
        raise ValueError(f"n must be >= 2, got {n}")

    pauli_terms: list[str] = []
    coeffs: list[float] = []

    for k in range(1, n):
        indices = list(range(n - k))
        # For each pair of (i, j) in indices, add Z_i Z_j Z_{i+k} Z_{j+k}
        for idx_a in indices:
            for idx_b in indices:
                label = ["I"] * n
                # Z_i Z_{i+k} * Z_j Z_{j+k} → 4 Z terms
                label[idx_a] = "Z"
                label[idx_b] = "Z"
                label[idx_a + k] = "Z"
                label[idx_b + k] = "Z"
                ps = "".join(label)
                pauli_terms.append(ps)
                coeffs.append(1.0)

    if not pauli_terms:
        pauli_terms.append("I" * n)
        coeffs.append(0.0)

    return SparsePauliOp(pauli_terms, coeffs).simplify()


def brute_force_best_labs(n: int) -> dict[str, Any]:
    """Brute-force optimum for small LABS instances (n ≤ 18).

    Returns dict with best_sequence, energy, merit_factor.
    """
    if n > 18:
        return {"error": f"n={n} too large for brute force (max 18)",
                "energy": None, "merit_factor": None}

    best_energy = float("inf")
    best_seq: list[int] = []

    for mask in range(1 << n):
        s = [1 if (mask >> i) & 1 else -1 for i in range(n)]
        c_sum_sq = 0.0
        for k in range(1, n):
            ck = sum(s[i] * s[i + k] for i in range(n - k))
            c_sum_sq += ck * ck
        if c_sum_sq < best_energy:
            best_energy = c_sum_sq
            best_seq = s

    merit = (n * n) / (2.0 * best_energy) if best_energy > 0 else float("inf")
    return {
        "n": n,
        "best_sequence": best_seq,
        "energy": best_energy,
        "merit_factor": round(merit, 6),
    }


# ── QAOA Ansatz (uncompressed) ────────────────────────────────────────

def _labs_qaoa_ansatz(hamiltonian: SparsePauliOp, n: int, p: int
                      ) -> tuple[QuantumCircuit, list[Parameter]]:
    """Build a QAOA ansatz for the LABS problem with *n* qubits, *p* layers.

    Uses a standard mixer (RX) and the LABS cost Hamiltonian.
    """
    flat_params: list[Parameter] = []
    for i in range(p):
        flat_params.append(Parameter(f"gamma_{i}"))
        flat_params.append(Parameter(f"beta_{i}"))

    qc = QuantumCircuit(n)
    qc.h(range(n))

    for layer in range(p):
        gamma = flat_params[2 * layer]
        beta = flat_params[2 * layer + 1]

        # Cost layer via PauliEvolutionGate
        from qiskit.circuit.library import PauliEvolutionGate
        evol = PauliEvolutionGate(hamiltonian, time=gamma)
        qc.append(evol, range(n))

        # Mixer
        qc.rx(2 * beta, range(n))

    return qc, flat_params


# ── Runner ─────────────────────────────────────────────────────────────

def run_labs_qaoa(
    n: int,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run uncompressed QAOA on a LABS problem instance of size *n*.

    Returns dict with optimisation results and the brute-force optimum
    for comparison (when n ≤ 18).
    """
    p = p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]

    hamiltonian = labs_hamiltonian(n)
    circuit, param_list = _labs_qaoa_ansatz(hamiltonian, n, p)
    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

    def obj_fn(params):
        return compute_expectation(circuit, hamiltonian, params.tolist())

    result = minimize(obj_fn, x0, method=opt,
                      options={"maxiter": iters, "disp": False})

    final_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())
    grad = estimate_gradient(circuit, hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    results: dict[str, Any] = {
        "n": n,
        "optimal_energy": float(final_energy),
        "optimal_params": result.x.tolist(),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "method": "baseline_qaoa",
        "num_physical_qubits": n,
        "compression_ratio": 1.0,
    }

    # Brute-force reference for small n
    if n <= 18:
        bf = brute_force_best_labs(n)
        results["optimal_maxcut_exact"] = bf["energy"]
        results["merit_factor_at_opt"] = bf["merit_factor"]
        if bf["energy"] and bf["energy"] > 0:
            results["optimal_merit_factor"] = round(
                (n * n) / (2.0 * final_energy), 4
            ) if final_energy > 0 else None
    else:
        results["optimal_maxcut_exact"] = None
        results["merit_factor_at_opt"] = None
        results["optimal_merit_factor"] = None

    return results


def run_labs_pce(
    n: int,
    k: int = 2,
    p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Run PCE-encoded QAOA on a LABS problem instance.

    Builds the PCE-encoded LABS Hamiltonian by mapping each variable
    through the encoding's Pauli strings.  For each 4-body term
    Z_i Z_{i+k} Z_j Z_{j+k} in the original LABS Hamiltonian, the
    encoded term is the product of the four corresponding Pauli strings.

    Parameters
    ----------
    n : int
        Number of LABS variables.
    k : int
        Correlation order for PCE encoding.
    p : int | None
        QAOA depth.

    Returns
    -------
    dict with optimisation results.
    """
    p = p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = seed or QAOA["seed"]

    # Build a chain graph and encode it (any connected graph works for
    # assignment generation; the actual Hamiltonian is built independently)
    chain = nx.path_graph(n)
    encoding = encode_graph(chain, k=k)
    m = encoding["num_physical_qubits"]
    var_map = encoding["variable_to_pauli_map"]

    # Build the PCE-encoded LABS Hamiltonian directly from the variable map
    pauli_terms: list[str] = []
    coeffs: list[float] = []

    from pce.manual_pce import _multiply_pauli_strings

    for shift in range(1, n):
        indices = list(range(n - shift))
        for idx_a in indices:
            for idx_b in indices:
                s_a = var_map[str(idx_a)]["pauli_string"]
                s_b = var_map[str(idx_b)]["pauli_string"]
                s_ak = var_map[str(idx_a + shift)]["pauli_string"]
                s_bk = var_map[str(idx_b + shift)]["pauli_string"]

                # Multiply all four together: Z_i Z_{i+k} Z_j Z_{j+k}
                # First multiply adjacent pairs, then the results
                p1, c1 = _multiply_pauli_strings(s_a, s_ak)
                p2, c2 = _multiply_pauli_strings(s_b, s_bk)
                p_final, c_final = _multiply_pauli_strings(p1, p2)

                net_coeff = c1 * c_final
                if net_coeff != 0.0 and not all(c == "I" for c in p_final):
                    pauli_terms.append(p_final)
                    coeffs.append(net_coeff)

    if not pauli_terms:
        pauli_terms.append("I" * m)
        coeffs.append(0.0)

    pce_hamiltonian = SparsePauliOp(pauli_terms, coeffs).simplify()

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

    results: dict[str, Any] = {
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
        "method": f"labs_pce_k{k}",
    }

    if n <= 18:
        bf = brute_force_best_labs(n)
        results["optimal_maxcut_exact"] = bf["energy"]
        results["merit_factor_at_opt"] = bf["merit_factor"]
    else:
        results["optimal_maxcut_exact"] = None
        results["merit_factor_at_opt"] = None

    return results


def _labs_method_name(method: str, k: int | None = None) -> str:
    if method == "baseline":
        return "labs_baseline"
    return f"labs_pce_k{k}" if k else "labs_pce"


# ── Logging helpers ───────────────────────────────────────────────────

def _log_labs_result(
    result: dict[str, Any],
    output_dir: Path | None = None,
) -> None:
    """Save a LABS result as JSON and append to index CSV."""
    output_dir = output_dir or EXPERIMENTS_DIR
    method = result.get("method", "labs_baseline")
    method_dir = output_dir / "labs" / method
    method_dir.mkdir(parents=True, exist_ok=True)

    n = result["n"]
    gid = f"labs_n{n}"

    path = method_dir / f"{gid}.json"
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    # ── Append to central index ─────────────────────────────────────
    from datetime import datetime, timezone
    append_index({
        "graph_id": gid,
        "family": "labs",
        "method": result.get("method", "labs_baseline"),
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


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LABS problem for QAOA/PCE")
    parser.add_argument("--n", type=int, default=None, help="Problem size")
    parser.add_argument("--method", type=str, default="baseline",
                        choices=["baseline", "pce"],
                        help="Encoding method")
    parser.add_argument("--k", type=int, default=None,
                        help="Correlation order for PCE")
    parser.add_argument("--p", type=int, default=None, help="QAOA depth")
    parser.add_argument("--brute-force", action="store_true",
                        help="Show brute-force optimum and exit")
    parser.add_argument("--batch", action="store_true",
                        help="Run multiple sizes")
    parser.add_argument("--n-values", type=int, nargs="*", default=[6, 8, 10],
                        help="Sizes for batch mode")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.brute_force and args.n:
        bf = brute_force_best_labs(args.n)
        print(f"Brute-force optimum for n={args.n}:")
        print(f"  Best energy:      {bf['energy']}")
        print(f"  Merit factor:     {bf['merit_factor']}")
        if "best_sequence" in bf:
            seq_str = "".join("+" if s == 1 else "-" for s in bf["best_sequence"])
            print(f"  Best sequence:    {seq_str}")
        sys.exit(0)

    if args.batch:
        n_values = args.n_values
        print(f"Running LABS batch for n={n_values} method={args.method} ...")
        for n in n_values:
            if args.method == "baseline":
                res = run_labs_qaoa(n, p=args.p)
            else:
                k = args.k or 2
                res = run_labs_pce(n, k=k, p=args.p)
            _log_labs_result(res, Path(args.output_dir) if args.output_dir else None)
            method = res.get("method", args.method)
            qubits = res.get("num_physical_qubits", n)
            comp = res.get("compression_ratio", 1.0)
            print(f"  n={n:2d} {method:20s} energy={res['optimal_energy']:.4f} "
                  f"qubits={qubits} comp={comp:.2f}x "
                  f"grad={res['gradient_norm_at_opt']:.6f} "
                  f"ok={res['success']}")

    elif args.n:
        if args.method == "baseline":
            res = run_labs_qaoa(args.n, p=args.p)
        else:
            k = args.k or 2
            res = run_labs_pce(args.n, k=k, p=args.p)
        _log_labs_result(res, Path(args.output_dir) if args.output_dir else None)
        print(json.dumps(res, indent=2))
    else:
        parser.print_help()
