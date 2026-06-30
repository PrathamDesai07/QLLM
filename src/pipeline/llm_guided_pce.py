"""LLM-guided Pauli-correlation encoding pipeline.

Flow for each graph instance:
  1. Load graph JSON + extracted features.
  2. Build LLM input (``LLMGraphInput``) from features + adjacency.
  3. Call the primary LLM (Qwen2.5-32B-Instruct / Gemma-2-27B) to obtain
     ``LLMOutput`` (k, Pauli assignments, tags, reasoning).
  4. Save the LLM output as a PCE encoding spec under ``data/pce/llm/``.
  5. Build the PCE cost Hamiltonian and run QAOA simulation.
  6. Log results to ``data/experiments/llm_pce/`` and the index CSV.

Command-line usage::

    python -m src.pipeline.llm_guided_pce --graph-id erdos_renyi_8_42
    python -m src.pipeline.llm_guided_pce --all-graphs
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

import networkx as nx

# Ensure src/ is on the path when run as a module or directly
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import (
    GRAPHS_DIR,
    FEATURES_DIR,
    PCE_DIR,
    EXPERIMENTS_DIR,
    LLM_CONFIG,
    QAOA,
)
from infra.logger import ExperimentRecord
from infra.experiment_tracker import run_metadata, append_index
from infra.simulator import compute_expectation, estimate_gradient
from llm.schema import LLMGraphInput, LLMOutput
from llm.client import get_llm_output_with_fallback, batch_get_llm_output_with_fallback, _fallback_output
from pce.manual_pce import build_pce_hamiltonian
from qaoa.pce_qaoa_baseline import pce_qaoa_ansatz
from qaoa.baseline_qaoa import brute_force_maxcut

logger = logging.getLogger(__name__)


# ── Path helpers ───────────────────────────────────────────────────────

LLM_PCE_DIR = PCE_DIR / "llm"
LLM_PCE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_LLM_DIR = EXPERIMENTS_DIR / "llm_pce"
EXPERIMENT_LLM_DIR.mkdir(parents=True, exist_ok=True)


# ── Pipeline steps ─────────────────────────────────────────────────────

def load_graph_and_features(
    graph_id: str,
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
) -> tuple[nx.Graph, dict[str, Any], LLMGraphInput]:
    """Load a graph + its pre-extracted features and build LLM input.

    Parameters
    ----------
    graph_id : str
        Graph identifier (without .json suffix).
    graph_dir : Path, optional
        Directory with graph JSONs.
    feature_dir : Path, optional
        Directory with feature JSONs.

    Returns
    -------
    (graph, feature_dict, llm_input)
    """
    graph_dir = graph_dir or GRAPHS_DIR
    feature_dir = feature_dir or FEATURES_DIR

    # Load graph
    graph_path = graph_dir / f"{graph_id}.json"
    with open(graph_path) as f:
        data = json.load(f)
    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})

    # Build LLM input from feature file if it exists, otherwise from graph
    feat_path = feature_dir / f"{graph_id}.json"
    if feat_path.exists():
        llm_input = LLMGraphInput.from_feature_file(feat_path)
        # Re-populate adjacency from the loaded graph
        from llm.schema import AdjacencyInfo
        llm_input.adjacency = AdjacencyInfo.from_graph(g)
        family = meta.get("family", llm_input.family)
        llm_input.family = family
        llm_input.graph_id = graph_id
    else:
        llm_input = LLMGraphInput.from_graph(
            g, graph_id=graph_id, family=meta.get("family", "unknown"),
        )

    # Load raw feature dict for saving
    if feat_path.exists():
        with open(feat_path) as f:
            feature_dict = json.load(f)
    else:
        from graphs.features import extract_all
        feature_dict = extract_all(g)

    return g, feature_dict, llm_input


def run_llm_pce(
    graph_id: str,
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
    model_name: str | None = None,
    prompt_path: Path | None = None,
    qaoa_p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    temperature: float | None = None,
    output_dir: Path | None = None,
    pce_output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the full LLM-guided PCE pipeline for one graph.

    Returns
    -------
    dict with keys:
        graph_id, family, llm_output (LLMOutput dict), qaoa_results (dict),
        record (ExperimentRecord), metadata (run metadata).
    """
    model_name = model_name or LLM_CONFIG["primary_model_name"]
    output_dir = output_dir or EXPERIMENTS_DIR  # record.save() adds method subdir
    pce_output_dir = pce_output_dir or LLM_PCE_DIR

    g, feature_dict, llm_input = load_graph_and_features(
        graph_id, graph_dir=graph_dir, feature_dir=feature_dir,
    )

    # ── Step 2–3: Call LLM ──────────────────────────────────────────
    llm_output: LLMOutput = get_llm_output_with_fallback(
        llm_input,
        model_name=model_name,
        prompt_path=prompt_path,
        temperature=temperature,
    )

    # ── Step 4: Save LLM output as PCE encoding ─────────────────────
    encoding = llm_output.to_encoding_dict()
    encoding["graph_id"] = graph_id
    encoding["family"] = llm_input.family
    encoding["features"] = feature_dict

    pce_path = pce_output_dir / f"{graph_id}.json"
    with open(pce_path, "w") as f:
        json.dump(encoding, f, indent=2)
    logger.info("Saved LLM PCE encoding to %s", pce_path)

    # ── Step 5: Build Hamiltonian + run QAOA ────────────────────────
    hamiltonian = build_pce_hamiltonian(g, encoding)
    num_physical_qubits = encoding["num_physical_qubits"]

    p = qaoa_p or QAOA["p"]
    opt = optimizer or QAOA["optimizer"]
    iters = max_iters or QAOA["max_iters"]
    seed = QAOA["seed"]

    circuit, param_list = pce_qaoa_ansatz(hamiltonian, num_physical_qubits, p)

    import numpy as np
    from scipy.optimize import minimize

    def obj_fn(params):
        return compute_expectation(circuit, hamiltonian, params.tolist())

    rng = np.random.default_rng(seed)
    x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

    result = minimize(
        obj_fn, x0, method=opt,
        options={"maxiter": iters, "disp": False},
    )

    final_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())

    # Approximation ratio
    num_edges = g.number_of_edges()
    mc_value = (num_edges - final_energy) / 2.0
    optimal_maxcut = brute_force_maxcut(g)
    approx_ratio = mc_value / optimal_maxcut if optimal_maxcut > 0 else 0.0

    grad = estimate_gradient(circuit, hamiltonian, result.x)
    grad_norm = float(np.linalg.norm(grad))

    qaoa_results = {
        "optimal_energy": float(final_energy),
        "optimal_params": result.x.tolist(),
        "optimal_pce_value": float(mc_value),
        "optimal_maxcut_exact": float(optimal_maxcut),
        "approximation_ratio": round(approx_ratio, 6),
        "gradient_norm_at_opt": round(grad_norm, 6),
        "convergence_iters": getattr(result, "nit", None),
        "nfeval": getattr(result, "nfev", None),
        "success": bool(result.success),
        "num_physical_qubits": num_physical_qubits,
        "num_variables": g.number_of_nodes(),
        "compression_ratio": encoding.get("compression_ratio", 1.0),
        "k": encoding["k"],
    }

    # ── Step 6: Log ─────────────────────────────────────────────────
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()

    record = ExperimentRecord(
        graph_id=graph_id,
        family=llm_input.family,
        num_nodes=g.number_of_nodes(),
        num_edges=num_edges,
        method="llm_pce",
        qaoa_p=p,
        optimizer=opt,
        max_iters=iters,
        optimal_energy=qaoa_results["optimal_energy"],
        optimal_params=qaoa_results["optimal_params"],
        approximation_ratio=qaoa_results["approximation_ratio"],
        gradient_norm_at_opt=qaoa_results["gradient_norm_at_opt"],
        convergence_iters=qaoa_results["convergence_iters"],
        success=qaoa_results["success"],
        params=feature_dict,
        extra={
            "k": qaoa_results["k"],
            "strategy": f"llm_k{qaoa_results['k']}",
            "num_physical_qubits": qaoa_results["num_physical_qubits"],
            "num_variables": qaoa_results["num_variables"],
            "compression_ratio": qaoa_results["compression_ratio"],
            "optimal_pce_value": qaoa_results["optimal_pce_value"],
            "optimal_maxcut_exact": qaoa_results["optimal_maxcut_exact"],
            "nfeval": qaoa_results["nfeval"],
            "model_name": model_name,
            "llm_reasoning": llm_output.reasoning,
            "llm_tags": llm_output.tags,
            "llm_approx_band": llm_output.approx_ratio_band,
        },
    )
    record.save(directory=output_dir)

    # ── Append to central index ─────────────────────────────────────
    meta = run_metadata(model_name,
                        prompt_file=str(prompt_path) if prompt_path else "default")
    prompt_hashes = meta.get("prompt_hashes", {})
    prompt_hash = next(iter(prompt_hashes.values()), "")
    append_index({
        "graph_id": graph_id,
        "family": llm_input.family,
        "method": "llm_pce",
        "model_name": model_name,
        "prompt_file": str(prompt_path) if prompt_path else "default",
        "prompt_hash": prompt_hash,
        "timestamp": timestamp,
        "k": str(qaoa_results["k"]),
        "num_physical_qubits": str(num_physical_qubits),
        "compression_ratio": str(qaoa_results["compression_ratio"]),
        "optimal_energy": str(record.optimal_energy),
        "approximation_ratio": str(record.approximation_ratio),
        "gradient_norm_at_opt": str(record.gradient_norm_at_opt),
        "success": str(record.success),
        "convergence_iters": str(record.convergence_iters) if record.convergence_iters is not None else "",
        "duration_seconds": str(record.duration_seconds) if record.duration_seconds is not None else "",
        "llm_tags": "|".join(llm_output.tags),
        "llm_reasoning": llm_output.reasoning[:120] if llm_output.reasoning else "",
        "git_commit": meta["git_commit"],
    })

    return {
        "graph_id": graph_id,
        "family": llm_input.family,
        "llm_output": llm_output,
        "qaoa_results": qaoa_results,
        "record": record,
        "metadata": meta,
    }


# ── Batch ──────────────────────────────────────────────────────────────

def run_llm_pce_batch_parallel(
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
    families: list[str] | None = None,
    max_graphs: int | None = None,
    max_workers: int = 4,
    model_name: str | None = None,
    prompt_path: Path | None = None,
    temperature: float | None = None,
    qaoa_p: int | None = None,
    optimizer: str | None = None,
    max_iters: int | None = None,
    output_dir: Path | None = None,
    pce_output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Multi-threaded parallel pipeline: loads + calls LLM concurrently.

    Step 1: Load all graphs in parallel via ThreadPoolExecutor.
    Step 2: Call LLM on all graphs in parallel via ``batch_get_llm_output``.
    Step 3: Run QAOA simulation sequentially (CPU-bound).
    Step 4: Save results.
    """
    import concurrent.futures

    graph_dir = graph_dir or GRAPHS_DIR
    output_dir = output_dir or EXPERIMENTS_DIR  # record.save() adds method subdir
    pce_output_dir = pce_output_dir or LLM_PCE_DIR
    model_name = model_name or LLM_CONFIG["primary_model_name"]

    # Gather graph paths
    graph_paths = sorted(graph_dir.glob("*.json"))
    if families or max_graphs:
        filtered: list[Path] = []
        for gpath in graph_paths:
            if max_graphs and len(filtered) >= max_graphs:
                break
            if families:
                with open(gpath) as f:
                    meta = json.load(f).get("metadata", {})
                if meta.get("family") not in families:
                    continue
            filtered.append(gpath)
        graph_paths = filtered
    elif max_graphs:
        graph_paths = graph_paths[:max_graphs]

    n_graphs = len(graph_paths)
    logger.info("Parallel batch processing %d graphs with %d workers",
                n_graphs, max_workers)

    # Step 1: Load all graphs in parallel
    def _load_one(gpath: Path) -> tuple[str, nx.Graph, dict[str, Any], LLMGraphInput]:
        gid = gpath.stem
        g, feat, inp = load_graph_and_features(
            gid, graph_dir=graph_dir, feature_dir=feature_dir,
        )
        return gid, g, feat, inp

    loaded: list[tuple[str, nx.Graph, dict[str, Any], LLMGraphInput]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_load_one, gp) for gp in graph_paths]
        for f in concurrent.futures.as_completed(futures):
            loaded.append(f.result())

    # Step 2: Call LLM on all graphs — uses batched GPU generation
    graph_inputs = [item[3] for item in loaded]
    logger.info("Calling LLM on %d graphs (batched GPU generation)...", len(graph_inputs))
    llm_outputs = batch_get_llm_output_with_fallback(
        graph_inputs,
        model_name=model_name,
        prompt_path=prompt_path,
        temperature=temperature,
    )

    results: list[dict[str, Any]] = []

    for (gid, g, feature_dict, llm_input), llm_output in zip(loaded, llm_outputs):
        # ── Save PCE encoding ───────────────────────────────────────
        encoding = llm_output.to_encoding_dict()
        encoding["graph_id"] = gid
        encoding["family"] = llm_input.family
        encoding["features"] = feature_dict

        pce_path = pce_output_dir / f"{gid}.json"
        with open(pce_path, "w") as f:
            json.dump(encoding, f, indent=2)

        # ── Run QAOA ────────────────────────────────────────────────
        hamiltonian = build_pce_hamiltonian(g, encoding)
        m_q = encoding["num_physical_qubits"]

        p = qaoa_p or QAOA["p"]
        opt = optimizer or QAOA["optimizer"]
        iters = max_iters or QAOA["max_iters"]
        seed = QAOA["seed"]

        circuit, param_list = pce_qaoa_ansatz(hamiltonian, m_q, p)

        import numpy as np
        from scipy.optimize import minimize

        def obj_fn(params):
            return compute_expectation(circuit, hamiltonian, params.tolist())

        rng = np.random.default_rng(seed)
        x0 = rng.uniform(0, 2 * np.pi, size=len(param_list))

        result = minimize(
            obj_fn, x0, method=opt,
            options={"maxiter": iters, "disp": False},
        )

        final_energy = compute_expectation(circuit, hamiltonian, result.x.tolist())

        num_edges = g.number_of_edges()
        mc_value = (num_edges - final_energy) / 2.0
        optimal_maxcut = brute_force_maxcut(g)
        approx_ratio = mc_value / optimal_maxcut if optimal_maxcut > 0 else 0.0

        grad = estimate_gradient(circuit, hamiltonian, result.x)
        grad_norm = float(np.linalg.norm(grad))

        qaoa_results = {
            "optimal_energy": float(final_energy),
            "optimal_params": result.x.tolist(),
            "optimal_pce_value": float(mc_value),
            "optimal_maxcut_exact": float(optimal_maxcut),
            "approximation_ratio": round(approx_ratio, 6),
            "gradient_norm_at_opt": round(grad_norm, 6),
            "convergence_iters": getattr(result, "nit", None),
            "nfeval": getattr(result, "nfev", None),
            "success": bool(result.success),
            "num_physical_qubits": m_q,
            "num_variables": g.number_of_nodes(),
            "compression_ratio": encoding.get("compression_ratio", 1.0),
            "k": encoding["k"],
        }

        # ── Log ─────────────────────────────────────────────────────
        from datetime import datetime, timezone
        timestamp = datetime.now(timezone.utc).isoformat()

        record = ExperimentRecord(
            graph_id=gid,
            family=llm_input.family,
            num_nodes=g.number_of_nodes(),
            num_edges=num_edges,
            method="llm_pce",
            qaoa_p=p,
            optimizer=opt,
            max_iters=iters,
            optimal_energy=qaoa_results["optimal_energy"],
            optimal_params=qaoa_results["optimal_params"],
            approximation_ratio=qaoa_results["approximation_ratio"],
            gradient_norm_at_opt=qaoa_results["gradient_norm_at_opt"],
            convergence_iters=qaoa_results["convergence_iters"],
            success=qaoa_results["success"],
            params=feature_dict,
            extra={
                "k": qaoa_results["k"],
                "strategy": f"llm_k{qaoa_results['k']}",
                "num_physical_qubits": qaoa_results["num_physical_qubits"],
                "num_variables": qaoa_results["num_variables"],
                "compression_ratio": qaoa_results["compression_ratio"],
                "optimal_pce_value": qaoa_results["optimal_pce_value"],
                "optimal_maxcut_exact": qaoa_results["optimal_maxcut_exact"],
                "nfeval": qaoa_results["nfeval"],
                "model_name": model_name,
                "llm_reasoning": llm_output.reasoning,
                "llm_tags": llm_output.tags,
                "llm_approx_band": llm_output.approx_ratio_band,
            },
        )
        record.save(directory=output_dir)

        # ── Index ───────────────────────────────────────────────────
        meta = run_metadata(
            model_name,
            prompt_file=str(prompt_path) if prompt_path else "default",
        )
        prompt_hashes = meta.get("prompt_hashes", {})
        prompt_hash = next(iter(prompt_hashes.values()), "")
        append_index({
            "graph_id": gid,
            "family": llm_input.family,
            "method": "llm_pce",
            "model_name": model_name,
            "prompt_file": str(prompt_path) if prompt_path else "default",
            "prompt_hash": prompt_hash,
            "timestamp": timestamp,
            "k": str(qaoa_results["k"]),
            "num_physical_qubits": str(m_q),
            "compression_ratio": str(qaoa_results["compression_ratio"]),
            "optimal_energy": str(record.optimal_energy),
            "approximation_ratio": str(record.approximation_ratio),
            "gradient_norm_at_opt": str(record.gradient_norm_at_opt),
            "success": str(record.success),
            "convergence_iters": str(record.convergence_iters) if record.convergence_iters is not None else "",
            "duration_seconds": str(record.duration_seconds) if record.duration_seconds is not None else "",
            "llm_tags": "|".join(llm_output.tags),
            "llm_reasoning": llm_output.reasoning[:120] if llm_output.reasoning else "",
            "git_commit": meta["git_commit"],
        })

        results.append({
            "graph_id": gid,
            "family": llm_input.family,
            "llm_output": llm_output,
            "qaoa_results": qaoa_results,
            "record": record,
            "metadata": meta,
        })

        k_val = qaoa_results["k"]
        comp = qaoa_results["compression_ratio"]
        tags = llm_output.tags
        logger.info(
            "[%s] k=%d ratio=%.4f qubits=%d comp=%.2fx tags=%s",
            gid, k_val, approx_ratio, m_q, comp, tags,
        )
        print(f"  [{gid}] k={k_val} ratio={approx_ratio:.4f} qubits={m_q} "
              f"comp={comp:.2f}x tags={tags}")

    return results


def run_llm_pce_batch(
    graph_dir: Path | None = None,
    feature_dir: Path | None = None,
    families: list[str] | None = None,
    max_graphs: int | None = None,
    **kwargs,
) -> list[dict[str, Any]]:
    """Run LLM-guided PCE on all graphs in *graph_dir*.

    Optionally filter by *families* and limit to *max_graphs*.
    """
    graph_dir = graph_dir or GRAPHS_DIR
    results = []

    graph_paths = sorted(graph_dir.glob("*.json"))
    if max_graphs is not None:
        graph_paths = graph_paths[:max_graphs]

    for gpath in graph_paths:
        graph_id = gpath.stem

        if families:
            with open(gpath) as f:
                meta = json.load(f).get("metadata", {})
            if meta.get("family") not in families:
                continue

        result = run_llm_pce(
            graph_id,
            graph_dir=graph_dir,
            feature_dir=feature_dir,
            **kwargs,
        )
        results.append(result)
        approx = result["qaoa_results"]["approximation_ratio"]
        qubits = result["qaoa_results"]["num_physical_qubits"]
        comp = result["qaoa_results"]["compression_ratio"]
        k_val = result["qaoa_results"]["k"]
        tags = result["llm_output"].tags
        logger.info(
            "[%s] k=%d ratio=%.4f qubits=%d comp=%.2fx tags=%s",
            graph_id, k_val, approx, qubits, comp, tags,
        )
        print(f"  [{graph_id}] k={k_val} ratio={approx:.4f} qubits={qubits} "
              f"comp={comp:.2f}x tags={tags}")

    return results


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="LLM-guided Pauli-correlation encoding pipeline"
    )
    parser.add_argument("--graph-id", type=str, default=None,
                        help="Single graph ID to process")
    parser.add_argument("--all-graphs", action="store_true",
                        help="Process all graphs in graph_dir")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--feature-dir", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                        help="HF model name (default: primary from config)")
    parser.add_argument("--p", type=int, default=None, help="QAOA depth")
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--families", nargs="*", default=None)
    parser.add_argument("--max-graphs", type=int, default=None)
    parser.add_argument("--parallel", action="store_true",
                        help="Use multi-threaded parallel LLM calls")
    parser.add_argument("--max-workers", type=int, default=4,
                        help="Number of parallel LLM API calls (default: 4)")
    args = parser.parse_args()

    if args.graph_id:
        result = run_llm_pce(
            args.graph_id,
            graph_dir=Path(args.graph_dir) if args.graph_dir else None,
            feature_dir=Path(args.feature_dir) if args.feature_dir else None,
            model_name=args.model,
            qaoa_p=args.p,
            max_iters=args.max_iters,
            temperature=args.temperature,
        )
        q = result["qaoa_results"]
        llm = result["llm_output"]
        print(f"\n=== Results for {args.graph_id} ===")
        print(f"  LLM model:     {args.model or LLM_CONFIG['primary_model_name']}")
        print(f"  k:             {q['k']}")
        print(f"  Physical qubits: {q['num_physical_qubits']}")
        print(f"  Compression:   {q['compression_ratio']}x")
        print(f"  Approx ratio:  {q['approximation_ratio']}")
        print(f"  Tags:          {llm.tags}")
        print(f"  Reasoning:     {llm.reasoning}")
        print(f"  Expected band: {llm.approx_ratio_band}")
    elif args.all_graphs:
        if args.parallel:
            run_llm_pce_batch_parallel(
                graph_dir=Path(args.graph_dir) if args.graph_dir else None,
                feature_dir=Path(args.feature_dir) if args.feature_dir else None,
                families=args.families,
                max_graphs=args.max_graphs,
                max_workers=args.max_workers,
                model_name=args.model,
                qaoa_p=args.p,
                max_iters=args.max_iters,
                temperature=args.temperature,
            )
        else:
            run_llm_pce_batch(
            graph_dir=Path(args.graph_dir) if args.graph_dir else None,
            feature_dir=Path(args.feature_dir) if args.feature_dir else None,
            families=args.families,
            max_graphs=args.max_graphs,
            model_name=args.model,
            qaoa_p=args.p,
            max_iters=args.max_iters,
            temperature=args.temperature,
        )
    else:
        parser.print_help()
