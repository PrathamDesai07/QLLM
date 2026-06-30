"""Rule-based PCE pipeline — QAOA using the deterministic rule engine instead of an LLM.

Flow for each graph instance:
  1. Load graph JSON + extracted features.
  2. Run ``rules.encoder_rules`` to obtain k and encoding strategy.
  3. Generate Pauli encoding via ``pce.manual_pce.encode_graph``.
  4. Build the PCE cost Hamiltonian and run QAOA simulation.
  5. Log results to ``data/experiments/rule_pce/`` and the index CSV.

Usage
-----
    python -m src.pipeline.rule_based_pce --graph-id erdos_renyi_8_0
    python -m src.pipeline.rule_based_pce --all-graphs
    python -m src.pipeline.rule_based_pce --status       # check completeness
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import networkx as nx

from config import (
    GRAPHS_DIR,
    FEATURES_DIR,
    PCE_DIR,
    EXPERIMENTS_DIR,
    QAOA,
)
from infra.logger import ExperimentRecord
from infra.experiment_tracker import append_index
from pce.manual_pce import encode_graph
from qaoa.pce_qaoa_baseline import run_pce_qaoa
from rules.encoder_rules import recommend_encoding


logger = logging.getLogger(__name__)


# ── Path helpers ───────────────────────────────────────────────────────

RULE_PCE_DIR = EXPERIMENTS_DIR / "rule_pce"
RULE_PCE_SUMMARY = EXPERIMENTS_DIR / "rule_pce_summary.json"
RULE_ENCODING_DIR = PCE_DIR / "rule"


# ── Pipeline ───────────────────────────────────────────────────────────

def run_rule_pce(
    graph_id: str,
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
    qaoa_p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the rule-based PCE pipeline for one graph.

    Returns dict with keys:
        graph_id, family, recommendation, encoding, qaoa_results, record.
    """
    graph_dir = graph_dir or GRAPHS_DIR
    feature_dir = feature_dir or FEATURES_DIR
    output_dir = output_dir or EXPERIMENTS_DIR
    RULE_PCE_DIR.mkdir(parents=True, exist_ok=True)

    p = qaoa_p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]

    # ── Step 1: Load graph ──────────────────────────────────────────
    graph_path = graph_dir / f"{graph_id}.json"
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})
    family = meta.get("family", "unknown")
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()

    # ── Step 2: Run rules engine to get recommendation ──────────────
    recommendation = recommend_encoding(graph_id, feature_dir=feature_dir)
    k = recommendation["k_recommendation"]["k"]
    gradient_risk = recommendation["gradient_risk"]
    rule_name = recommendation["k_recommendation"]["rule_name"]
    strategy = recommendation["pauli_recommendation"]["strategy"]

    if verbose:
        print(f"  [{graph_id}] rule={rule_name} k={k} "
              f"risk={gradient_risk['risk_level']}")

    # ── Step 3: Generate Pauli encoding ─────────────────────────────
    encoding = encode_graph(g, k=k)
    encoding["graph_id"] = graph_id
    encoding["family"] = family
    encoding["rule_name"] = rule_name
    encoding["rule_strategy"] = strategy
    encoding["gradient_risk"] = gradient_risk["risk_level"]

    # Save encoding for downstream analysis
    RULE_ENCODING_DIR.mkdir(parents=True, exist_ok=True)
    with open(RULE_ENCODING_DIR / f"{graph_id}.json", "w") as f:
        json.dump(encoding, f, indent=2)

    # ── Step 4: Run QAOA ────────────────────────────────────────────
    qaoa_result = run_pce_qaoa(
        g, encoding, p=p, optimizer=opt, max_iters=iters,
    )

    m_q = qaoa_result["num_physical_qubits"]
    approx_ratio = qaoa_result["approximation_ratio"]
    grad_norm = qaoa_result["gradient_norm_at_opt"]
    comp_ratio = qaoa_result["compression_ratio"]

    # ── Step 5: Log result ──────────────────────────────────────────
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    record = ExperimentRecord(
        graph_id=graph_id,
        family=family,
        num_nodes=num_nodes,
        num_edges=num_edges,
        method="rule_pce",
        qaoa_p=p,
        optimizer=opt,
        max_iters=iters,
        optimal_energy=qaoa_result["optimal_energy"],
        optimal_params=qaoa_result["optimal_params"],
        approximation_ratio=approx_ratio,
        gradient_norm_at_opt=grad_norm,
        convergence_iters=qaoa_result["convergence_iters"],
        success=qaoa_result["success"],
        params=recommendation.get("feature_summary", {}),
        extra={
            "k": k,
            "strategy": strategy,
            "rule_name": rule_name,
            "gradient_risk_level": gradient_risk["risk_level"],
            "gradient_risk_reasoning": gradient_risk["reasoning"],
            "num_physical_qubits": m_q,
            "num_variables": num_nodes,
            "compression_ratio": comp_ratio,
            "optimal_pce_value": qaoa_result.get("optimal_pce_value"),
            "optimal_maxcut_exact": qaoa_result.get("optimal_maxcut_exact"),
            "nfeval": qaoa_result.get("nfeval"),
        },
    )
    record.save(directory=output_dir)

    # ── Index CSV ───────────────────────────────────────────────────
    from infra.experiment_tracker import _git_hash
    append_index({
        "graph_id": graph_id,
        "family": family,
        "method": "rule_pce",
        "model_name": "",
        "prompt_file": "",
        "prompt_hash": "",
        "timestamp": timestamp,
        "k": str(k),
        "num_physical_qubits": str(m_q),
        "compression_ratio": str(comp_ratio),
        "optimal_energy": str(record.optimal_energy),
        "approximation_ratio": str(approx_ratio),
        "gradient_norm_at_opt": str(grad_norm),
        "success": str(record.success),
        "convergence_iters": str(record.convergence_iters) if record.convergence_iters is not None else "",
        "duration_seconds": str(record.duration_seconds) if record.duration_seconds is not None else "",
        "llm_tags": "",
        "llm_reasoning": "",
        "git_commit": _git_hash(),
    })

    if verbose:
        print(f"  [{graph_id}] k={k} ratio={approx_ratio:.4f} "
              f"qubits={m_q} comp={comp_ratio:.2f}x grad={grad_norm:.6f}")

    return {
        "graph_id": graph_id,
        "family": family,
        "k": k,
        "recommendation": recommendation,
        "encoding": encoding,
        "qaoa_results": qaoa_result,
        "record": record,
    }


# ── Batch ──────────────────────────────────────────────────────────────

def run_rule_pce_batch(
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
    families: list[str] | None = None,
    max_graphs: int | None = None,
    skip_existing: bool = True,
    **kwargs,
) -> list[dict[str, Any]]:
    """Run rule-based PCE on all graphs (optionally filtered)."""
    graph_dir = graph_dir or GRAPHS_DIR
    feature_dir = feature_dir or FEATURES_DIR
    results = []

    existing: set[str] = set()
    if skip_existing:
        existing = {f.stem for f in RULE_PCE_DIR.glob("*.json")}

    graph_paths = sorted(graph_dir.glob("*.json"))
    if families:
        graph_paths = [p for p in graph_paths
                       if _get_family(p) in families]
    if max_graphs is not None:
        graph_paths = graph_paths[:max_graphs]

    for gpath in graph_paths:
        gid = gpath.stem
        if skip_existing and gid in existing:
            continue
        result = run_rule_pce(gid, graph_dir=graph_dir,
                              feature_dir=feature_dir, **kwargs)
        results.append(result)

    return results


def _get_family(path: Path) -> str:
    """Extract family from a graph JSON file."""
    try:
        with open(path) as f:
            return json.load(f).get("metadata", {}).get("family", "unknown")
    except Exception:
        return "unknown"


def rule_pce_status() -> None:
    """Print completeness report for rule-based PCE experiments."""
    all_graphs = {f.stem for f in GRAPHS_DIR.glob("*.json")}
    done = {f.stem for f in RULE_PCE_DIR.glob("*.json")} if RULE_PCE_DIR.exists() else set()
    print()
    print(f"  Rule-based PCE Status")
    print(f"  {'─' * 40}")
    print(f"  Total graphs:  {len(all_graphs)}")
    print(f"  Completed:     {len(done & all_graphs)}")
    print(f"  Remaining:     {len(all_graphs - done)}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Rule-based PCE pipeline — no LLM required"
    )
    parser.add_argument("--graph-id", type=str, default=None,
                        help="Single graph ID to process")
    parser.add_argument("--all-graphs", action="store_true",
                        help="Process all graphs")
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--p", type=int, default=None, help="QAOA depth")
    parser.add_argument("--no-skip", dest="skip_existing",
                        action="store_false",
                        help="Re-run even if results exist")
    parser.add_argument("--status", action="store_true",
                        help="Print completeness report")
    args = parser.parse_args()

    if args.status:
        rule_pce_status()
    elif args.graph_id:
        run_rule_pce(args.graph_id, qaoa_p=args.p)
    elif args.all_graphs:
        run_rule_pce_batch(
            families=args.families,
            max_graphs=args.max_graphs,
            skip_existing=args.skip_existing,
            qaoa_p=args.p,
        )
    else:
        parser.print_help()
