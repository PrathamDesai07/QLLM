"""Manual Pauli-correlation encoding — hand-crafted mappings of variables to Pauli strings."""

import json
import itertools
from pathlib import Path

import networkx as nx
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from config import PCE, PCE_DIR


# ── Encoding Strategies ────────────────────────────────────────────────

def _qubits_needed(n_vars: int, k: int) -> int:
    """Minimum number of physical qubits needed to encode *n_vars* variables
    with correlation order *k*, assuming 3 distinct Pauli operators per qubit.

    For k=1: each qubit can host up to 3 variables (Pauli X, Y, Z).
    For k>=2: each combination of k qubits has 3^k Pauli strings.
    """
    if k == 1:
        return max(1, (n_vars + 2) // 3)
    # combinations of k qubits from m: C(m, k) * 3^k >= n_vars
    # approximate: m ≈ ceil(cuberoot(n_vars * 6 / 3^k)) for k=2
    # simple: m = ceil(sqrt(2 * n_vars / 9)) for k=2
    if k == 2:
        return max(1, int(np.ceil(np.sqrt(2 * n_vars / 9))))
    # fallback: m = n_vars // 3 (weak encoding)
    return max(1, n_vars // k)


PAULI_OPS = ["X", "Y", "Z"]


def encode_k1_single_pauli(g: nx.Graph, seed: int | None = None) -> dict:
    """k=1 encoding: each variable → one Pauli on one physical qubit.

    Up to 3 variables per qubit (one each of X, Y, Z).  Higher-degree
    variables are placed first to get priority on Pauli choice.
    """
    n = g.number_of_nodes()
    m = _qubits_needed(n, k=1)

    # sort nodes by degree descending
    nodes = sorted(g.nodes(), key=lambda v: g.degree(v), reverse=True)

    pauli_strings: dict[int, str] = {}  # variable_idx -> Pauli string
    variable_to_pauli_map: dict[int, str] = {}
    assignments: dict[int, int] = {}     # qubit -> how many vars assigned
    for q in range(m):
        assignments[q] = 0

    for idx, node in enumerate(nodes):
        q = idx % m
        pauli_idx = assignments[q] % 3
        pauli = PAULI_OPS[pauli_idx]
        assignments[q] += 1

        label = ["I"] * m
        label[q] = pauli
        ps = "".join(label)
        pauli_strings[idx] = ps
        variable_to_pauli_map[node] = {"qubits": [q], "paulis": [pauli], "pauli_string": ps}

    return {
        "k": 1,
        "num_physical_qubits": m,
        "num_variables": n,
        "compression_ratio": round(n / m, 2) if m else 1.0,
        "strategy": "k1_single_pauli",
        "pauli_strings": pauli_strings,
        "variable_to_pauli_map": {str(k): v for k, v in variable_to_pauli_map.items()},
    }


def encode_k2_paired_pauli(g: nx.Graph, seed: int | None = None) -> dict:
    """k=2 encoding: each variable → 2-qubit Pauli string.

    Variables get distinct (qubit_pair, pauli_pair) assignments.
    """
    n = g.number_of_nodes()
    m = _qubits_needed(n, k=2)
    while m < 2 or (m * (m - 1) // 2 * 9) < n:
        m += 1

    nodes = sorted(g.nodes(), key=lambda v: g.degree(v), reverse=True)

    qubit_pairs = list(itertools.combinations(range(m), 2))
    pauli_pairs = list(itertools.product(PAULI_OPS, repeat=2))
    all_assignments = list(itertools.product(qubit_pairs, pauli_pairs))

    single_qubit_opts = [(q,) for q in range(m)]
    all_single = list(itertools.product(single_qubit_opts, [(p,) for p in PAULI_OPS]))
    all_assignments_ext = list(all_assignments) + list(all_single)

    if len(all_assignments_ext) < n:
        # fall back to k=1-style if not enough combinations
        return encode_k1_single_pauli(g, seed=seed)

    pauli_strings: dict[int, str] = {}
    variable_to_pauli_map: dict[int, dict] = {}

    for idx, node in enumerate(nodes):
        qubits, paulis = all_assignments_ext[idx]
        label = ["I"] * m
        for q, p in zip(qubits, paulis):
            label[q] = p
        ps = "".join(label)
        pauli_strings[idx] = ps
        variable_to_pauli_map[node] = {
            "qubits": list(qubits),
            "paulis": list(paulis),
            "pauli_string": ps,
        }

    return {
        "k": 2,
        "num_physical_qubits": m,
        "num_variables": n,
        "compression_ratio": round(n / m, 2) if m else 1.0,
        "strategy": "k2_paired_pauli",
        "pauli_strings": pauli_strings,
        "variable_to_pauli_map": {str(k): v for k, v in variable_to_pauli_map.items()},
    }


# ── Dispatch ───────────────────────────────────────────────────────────

def encode_k3_triple_pauli(g: nx.Graph, seed: int | None = None) -> dict:
    """k=3 encoding: each variable → 3-qubit Pauli string for dense modular graphs.

    Variables get distinct 3-qubit (X/Y/Z) string assignments.
    Falls back to k=2 if insufficient combinations exist.
    """
    n = g.number_of_nodes()
    # m qubits: C(m, 3) * 3^3 >= n → approximate
    m = 3
    while m * (m - 1) * (m - 2) // 6 * 27 < n:
        m += 1
        if m > 10:
            return encode_k2_paired_pauli(g, seed=seed)

    nodes = sorted(g.nodes(), key=lambda v: g.degree(v), reverse=True)

    qubit_triples = list(itertools.combinations(range(m), 3))
    pauli_triples = list(itertools.product(PAULI_OPS, repeat=3))
    all_assignments = list(itertools.product(qubit_triples, pauli_triples))

    # Also add k=2 and k=1 assignments as fallback options
    qubit_pairs = list(itertools.combinations(range(m), 2))
    pauli_pairs = list(itertools.product(PAULI_OPS, repeat=2))
    all_assignments_all = list(all_assignments) + \
        list(itertools.product(qubit_pairs, pauli_pairs)) + \
        list(itertools.product([(q,) for q in range(m)], [(p,) for p in PAULI_OPS]))

    if len(all_assignments_all) < n:
        return encode_k2_paired_pauli(g, seed=seed)

    pauli_strings: dict[int, str] = {}
    variable_to_pauli_map: dict[int, dict] = {}

    for idx, node in enumerate(nodes):
        qubits, paulis = all_assignments_all[idx]
        label = ["I"] * m
        for q, p in zip(qubits, paulis):
            label[q] = p
        ps = "".join(label)
        pauli_strings[idx] = ps
        variable_to_pauli_map[node] = {
            "qubits": list(qubits),
            "paulis": list(paulis),
            "pauli_string": ps,
        }

    compression_ratio = round(n / m, 2) if m else 1.0
    return {
        "k": 3,
        "num_physical_qubits": m,
        "num_variables": n,
        "compression_ratio": compression_ratio,
        "strategy": "k3_triple_pauli",
        "pauli_strings": pauli_strings,
        "variable_to_pauli_map": {str(k): v for k, v in variable_to_pauli_map.items()},
    }


# ── Dispatch ───────────────────────────────────────────────────────────

ENCODERS = {
    1: encode_k1_single_pauli,
    2: encode_k2_paired_pauli,
    3: encode_k3_triple_pauli,
}


def encode_graph(
    g: nx.Graph,
    k: int | None = None,
    seed: int | None = None,
) -> dict:
    """Generate a Pauli-correlation encoding for *g* with correlation order *k*."""
    k = k or PCE["default_k"]
    if k not in ENCODERS:
        raise ValueError(f"Unsupported correlation order k={k}. Supported: {list(ENCODERS)}")
    return ENCODERS[k](g, seed=seed)


# ── Hamiltonian Building ───────────────────────────────────────────────

def _multiply_pauli_strings(pauli_a: str, pauli_b: str) -> tuple[str, float]:
    """Multiply two Pauli strings element-wise, returning (result_string, real_coeff).

    The coefficient is real (0, ±1).  If the product has an imaginary
    phase the coefficient is set to 0 (that edge contributes nothing
    under this encoding).
    """
    # Pauli multiplication tables: phase exponent (0=1, 1=i, 2=-1, 3=-i)
    # and resulting Pauli index for each (first, second) pair.
    _idx = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    _phase = [
        [0, 0, 0, 0],
        [0, 0, 1, 3],
        [0, 3, 0, 1],
        [0, 1, 3, 0],
    ]
    _res = [
        [0, 1, 2, 3],
        [1, 0, 2, 3],
        [2, 3, 0, 1],
        [3, 2, 1, 0],
    ]

    total_phase = 0
    result_chars = []
    for c1, c2 in zip(pauli_a, pauli_b):
        a = _idx[c1]
        b = _idx[c2]
        total_phase = (total_phase + _phase[a][b]) % 4
        result_chars.append("IXYZ"[_res[a][b]])

    if total_phase % 2 == 1:
        return "I" * len(pauli_a), 0.0

    coeff = -1.0 if total_phase == 2 else 1.0
    return "".join(result_chars), coeff


def build_pce_hamiltonian(
    g: nx.Graph, encoding: dict
) -> SparsePauliOp:
    """Build the PCE-encoded cost Hamiltonian from graph edges and encoding.

    For each edge (i,j) in the original graph, the term becomes the
    Hermitian product of the two Pauli strings assigned to variables i and j.
    Edges that would produce an imaginary coefficient are dropped.
    """
    var_map = encoding["variable_to_pauli_map"]

    pauli_terms: list[str] = []
    coeffs: list[float] = []

    for i, j in g.edges():
        si = var_map[str(i)]["pauli_string"]
        sj = var_map[str(j)]["pauli_string"]
        result_str, coeff = _multiply_pauli_strings(si, sj)
        if coeff != 0.0 and not all(c == "I" for c in result_str):
            pauli_terms.append(result_str)
            coeffs.append(coeff)

    if not pauli_terms:
        # fallback: identity term so SparsePauliOp is valid
        m = encoding["num_physical_qubits"]
        pauli_terms.append("I" * m)
        coeffs.append(0.0)

    return SparsePauliOp(pauli_terms, coeffs)


# ── Persistence ────────────────────────────────────────────────────────

def save_encoding(encoding: dict, graph_id: str,
                  directory: Path | None = None) -> Path:
    """Save encoding spec to ``directory / {graph_id}.json``."""
    directory = directory or PCE_DIR
    path = directory / f"{graph_id}.json"
    with open(path, "w") as f:
        json.dump(encoding, f, indent=2)
    return path


def encode_and_save(
    graph_path: Path,
    k: int | None = None,
    seed: int | None = None,
    directory: Path | None = None,
) -> dict:
    """Load a graph, generate an encoding, save it, return the encoding dict."""
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})

    encoding = encode_graph(g, k=k, seed=seed)
    encoding["graph_id"] = meta.get("graph_id", graph_path.stem)
    encoding["family"] = meta.get("family", "unknown")

    save_encoding(encoding, encoding["graph_id"], directory=directory)
    return encoding


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate PCE encoding")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--graph-id", type=str, default=None)
    args = parser.parse_args()

    from config import GRAPHS_DIR
    gdir = Path(args.graph_dir) if args.graph_dir else GRAPHS_DIR
    odir = Path(args.output_dir) if args.output_dir else None
    k = args.k or PCE["default_k"]

    if args.graph_id:
        enc = encode_and_save(gdir / f"{args.graph_id}.json",
                              k=k, seed=args.seed, directory=odir)
        print(f"Encoded {enc['graph_id']}: {enc['num_physical_qubits']} qubits "
              f"(compression {enc['compression_ratio']}x, k={k})")
    else:
        for gpath in sorted(gdir.glob("*.json")):
            enc = encode_and_save(gpath, k=k, seed=args.seed, directory=odir)
            print(f"  {enc['graph_id']}: {enc['num_physical_qubits']} qubits "
                  f"(compression {enc['compression_ratio']}x)")
