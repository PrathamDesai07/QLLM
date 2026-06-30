"""Experiment suites for MaxCut — grid orchestrator across graph families,
sizes, and encoding methods (baseline QAOA, manual PCE, LLM-guided PCE).

Usage
-----
    python -m src.experiments.suites.maxcut_suites          # full grid
    python -m src.experiments.suites.maxcut_suites --status  # dry-run report
    python -m src.experiments.suites.maxcut_suites --families erdos_renyi community
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from config import (
    GRAPHS_DIR,
    FEATURES_DIR,
    EXPERIMENTS_DIR,
    GRAPH_DEFAULTS,
    QAOA,
    PCE as PCE_CONFIG,
)
from infra.logger import ExperimentRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid definition
# ---------------------------------------------------------------------------

def build_grid(
    families: list[str] | None = None,
    min_nodes: int | None = None,
    max_nodes: int | None = None,
    step: int | None = None,
    instances_per_size: int | None = None,
) -> list[dict[str, Any]]:
    """Build the full experiment grid as a list of (family, n, seed) configs.

    Parameters
    ----------
    families : list[str] | None
        Graph families to include (default: all from config).
    min_nodes, max_nodes, step : int | None
        Node-count range (default: from config).
    instances_per_size : int | None
        Random seeds per (family, size) cell (default: from config).

    Returns
    -------
    list of dicts with keys: family, n, seed, graph_id.
    """
    families = families or list(GRAPH_DEFAULTS["families"])
    min_n = min_nodes or GRAPH_DEFAULTS["min_nodes"]
    max_n = max_nodes or GRAPH_DEFAULTS["max_nodes"]
    step_n = step or GRAPH_DEFAULTS["step"]
    instances = instances_per_size or GRAPH_DEFAULTS["instances_per_size"]

    grid: list[dict[str, Any]] = []
    for family in families:
        for n in range(min_n, max_n + 1, step_n):
            for seed in range(instances):
                gid = f"{family}_{n}_{seed}"
                grid.append({
                    "family": family,
                    "n": n,
                    "seed": seed,
                    "graph_id": gid,
                })
    return grid


def grid_nodes(grid: list[dict[str, Any]]) -> int:
    """Return the number of graph instances defined by the grid."""
    return len(grid)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _parse_summary(summary_path: Path) -> dict[str, dict[str, Any]]:
    """Load a summary JSON into a {graph_id: record} lookup dict."""
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        records = json.load(f)
    return {r["graph_id"]: r for r in records}


def suite_status(
    families: list[str] | None = None,
    k_values: list[int] | None = None,
) -> dict[str, Any]:
    """Report which experiments are complete and which are missing.

    Parameters
    ----------
    families : list[str] | None
        Filter to specific families.
    k_values : list[int] | None
        PCE k-values to check (default: [1]).

    Returns
    -------
    dict with keys:
        grid_size, graphs_exist, graphs_missing,
        baseline_complete, baseline_missing,
        pce_status (list per k value),
        llm_complete, llm_missing.
    """
    if k_values is None:
        k_values = [PCE_CONFIG.get("default_k", 2)]

    grid = build_grid(families=families)
    grid_ids = {g["graph_id"] for g in grid}

    # Existing graph files
    existing_graphs = {f.stem for f in GRAPHS_DIR.glob("*.json")}

    # Baseline QAOA — check individual JSON records (not just summary)
    baseline_done = {f.stem for f in (EXPERIMENTS_DIR / "baseline_qaoa").glob("*.json")}
    # Also check the summary file for any extra entries
    baseline_summary = _parse_summary(EXPERIMENTS_DIR / "baseline_qaoa_summary.json")

    # PCE baseline — check by (graph_id, k) by reading record files
    pce_files = (EXPERIMENTS_DIR / "pce_baseline").glob("*.json")
    pce_done_by_k: dict[int, set[str]] = {k: set() for k in k_values}
    for f in pce_files:
        try:
            with open(f) as fh:
                d = json.load(fh)
            k = d.get("extra", {}).get("k")
            if k in pce_done_by_k:
                pce_done_by_k[k].add(d["graph_id"])
        except (json.JSONDecodeError, KeyError):
            pass

    # LLM PCE
    llm_done = {f.stem for f in (EXPERIMENTS_DIR / "llm_pce").glob("*.json")}

    return {
        "grid_size": len(grid),
        "grid_ids": grid_ids,
        "graphs_exist": existing_graphs & grid_ids,
        "graphs_missing": grid_ids - existing_graphs,
        "baseline_complete": grid_ids & baseline_done,
        "baseline_missing": grid_ids - baseline_done,
        "pce_status": {
            k: {
                "complete": grid_ids & pce_done_by_k.get(k, set()),
                "missing": grid_ids - pce_done_by_k.get(k, set()),
            }
            for k in k_values
        },
        "llm_complete": grid_ids & llm_done,
        "llm_missing": grid_ids - llm_done,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_maxcut_suite(
    families: list[str] | None = None,
    baseline: bool = True,
    pce: bool = True,
    pce_k_values: list[int] | None = None,
    llm: bool = True,
    llm_model: str | None = None,
    llm_parallel: bool = False,
    max_graphs: int | None = None,
    skip_existing: bool = True,
    fail_fast: bool = False,
    baseline_p: int | None = None,
    pce_p: int | None = None,
    llm_p: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full MaxCut experiment suite on the defined grid.

    For each graph in the grid, this orchestrates:

        1. **Baseline QAOA** — uncompressed QAOA via ``run_baseline_batch``.
        2. **Manual PCE** — PCE-encoded QAOA at each ``k`` via ``run_pce_batch``.
        3. **LLM-guided PCE** — LLM-chosen encoding via ``run_llm_pce_batch``.

    The suite first discovers which results already exist and only runs
    missing experiments (idempotent by default).

    Parameters
    ----------
    families : list[str] | None
        Graph families to include (all from config if None).
    baseline : bool
        Run baseline QAOA experiments (default True).
    pce : bool
        Run manual PCE experiments (default True).
    pce_k_values : list[int] | None
        k values for manual PCE (default from config ``default_k``).
    llm : bool
        Run LLM-guided PCE experiments (default True).
    llm_model : str | None
        Model name for LLM (default from config).
    llm_parallel : bool
        Use batched GPU generation (default False).
    max_graphs : int | None
        Limit number of graphs to process (for testing).
    skip_existing : bool
        Skip graphs that already have results (default True).
    fail_fast : bool
        Raise on first error (default False, logs warnings).
    baseline_p, pce_p, llm_p : int | None
        QAOA depth per method (default from config).
    verbose : bool
        Print progress.

    Returns
    -------
    dict with keys:
        grid_size, graphs_processed, baseline_results, pce_results,
        llm_results, errors.
    """
    if pce_k_values is None:
        pce_k_values = [PCE_CONFIG.get("default_k", 2)]

    status = suite_status(families=families, k_values=pce_k_values)
    grid_ids = status["grid_ids"]
    errors: list[str] = []

    # Limit for testing
    if max_graphs is not None and max_graphs < len(grid_ids):
        grid_ids = set(sorted(grid_ids)[:max_graphs])

    # Filter to families that have graphs
    active_families = families or list(GRAPH_DEFAULTS["families"])

    results: dict[str, Any] = {
        "grid_size": len(grid_ids),
        "graphs_processed": 0,
        "baseline_results": [],
        "pce_results": [],
        "llm_results": [],
        "errors": errors,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"  MaxCut Experiment Suite")
        print(f"  Grid: {len(grid_ids)} graph instances")
        print(f"  Families: {active_families}")
        print(f"  Methods: "
              f"{'baseline ' if baseline else ''}"
              f"{'pce ' if pce else ''}"
              f"{'llm ' if llm else ''}")
        print(f"  Skip existing: {skip_existing}")
        print(f"{'='*60}\n")

    # ── Step 1: Ensure graphs exist ──────────────────────────────────
    missing_graphs = status["graphs_missing"]
    if skip_existing:
        missing_graphs = {g for g in missing_graphs if g in grid_ids}
    if missing_graphs:
        if verbose:
            print(f"Generating {len(missing_graphs)} missing graphs …")
        from graphs.generate_graphs import generate_graph, save_graph
        import numpy as np
        rng = np.random.default_rng(42)
        for gid in sorted(missing_graphs):
            parts = gid.rsplit("_", 2)
            family = parts[0]
            n = int(parts[1])
            seed = int(parts[2])
            g = generate_graph(
                family, n=n,
                p=float(rng.uniform(0.3, 0.7)) if family == "erdos_renyi" else None,
                d=max(1, n // 4) if family == "d_regular" else None,
                seed=seed,
            )
            save_graph(g, family, {"graph_id": gid, "family": family,
                                    "params": {"n": n, "seed": seed}},
                       directory=GRAPHS_DIR)
        if verbose:
            print(f"  Generated {len(missing_graphs)} graphs.\n")

    # ── Step 2: Generate features for any graph that lacks them ──────
    existing_features = {f.stem for f in FEATURES_DIR.glob("*.json")}
    missing_features = grid_ids - existing_features
    if missing_features:
        if verbose:
            print(f"Extracting features for {len(missing_features)} graphs …")
        from graphs.features import features_from_graph_file
        for gid in sorted(missing_features):
            features_from_graph_file(GRAPHS_DIR / f"{gid}.json")
        if verbose:
            print(f"  Done.\n")

    # ── Step 3: Baseline QAOA ───────────────────────────────────────
    if baseline:
        baseline_needed = status["baseline_missing"] & grid_ids
        if skip_existing:
            baseline_needed = {g for g in baseline_needed
                               if g in (GRAPHS_DIR / "baseline_qaoa"
                                        if False else grid_ids)}

            # Actually check what's missing at the individual graph level
            existing_baseline = {f.stem for f in
                                  (EXPERIMENTS_DIR / "baseline_qaoa").glob("*.json")}
            baseline_needed = {g for g in grid_ids
                               if g not in existing_baseline}

        if verbose:
            print(f"Baseline QAOA: {len(baseline_needed)} graphs remaining …")
        if baseline_needed:
            from qaoa.baseline_qaoa import run_baseline_batch
            try:
                recs = run_baseline_batch(
                    graph_dir=GRAPHS_DIR,
                    output_dir=EXPERIMENTS_DIR,
                    families=active_families,
                    p=baseline_p,
                )
                results["baseline_results"] = recs
                if verbose:
                    success = sum(1 for r in recs if r.success)
                    print(f"  Baseline complete: {len(recs)} runs, {success} successful")
            except Exception as exc:
                msg = f"Baseline batch failed: {exc}"
                errors.append(msg)
                if verbose:
                    print(f"  ERROR: {msg}")
                if fail_fast:
                    raise
        else:
            if verbose:
                print(f"  All baseline results already exist — skipping.\n")

    # ── Step 4: Manual PCE ──────────────────────────────────────────
    if pce:
        for k in pce_k_values:
            pce_needed = status["pce_status"][k]["missing"] & grid_ids if skip_existing else grid_ids
            # Double-check against actual files
            pce_done = set()
            for f in (EXPERIMENTS_DIR / "pce_baseline").glob("*.json"):
                try:
                    with open(f) as fh:
                        d = json.load(fh)
                    if d.get("extra", {}).get("k") == k:
                        pce_done.add(d["graph_id"])
                except Exception:
                    pass
            pce_needed = {g for g in grid_ids if g not in pce_done}

            if verbose:
                print(f"Manual PCE (k={k}): {len(pce_needed)} graphs remaining …")
            if pce_needed:
                from qaoa.pce_qaoa_baseline import run_pce_batch
                try:
                    recs = run_pce_batch(
                        graph_dir=GRAPHS_DIR,
                        output_dir=EXPERIMENTS_DIR,
                        families=active_families,
                        k=k,
                        p=pce_p,
                    )
                    results["pce_results"].extend(recs)
                    if verbose:
                        success = sum(1 for r in recs if r.success)
                        total_needed = len(pce_needed)
                        print(f"  PCE k={k} complete: {len(recs)} runs, "
                              f"{success} successful")
                except Exception as exc:
                    msg = f"PCE k={k} batch failed: {exc}"
                    errors.append(msg)
                    if verbose:
                        print(f"  ERROR: {msg}")
                    if fail_fast:
                        raise
            else:
                if verbose:
                    print(f"  All PCE k={k} results exist — skipping.\n")

    # ── Step 5: LLM-guided PCE ──────────────────────────────────────
    if llm:
        llm_done = {f.stem for f in (EXPERIMENTS_DIR / "llm_pce").glob("*.json")}
        llm_needed = {g for g in grid_ids if g not in llm_done}

        if verbose:
            print(f"LLM-guided PCE: {len(llm_needed)} graphs remaining …")
        if llm_needed:
            from pipeline.llm_guided_pce import run_llm_pce_batch, run_llm_pce_batch_parallel

            if llm_parallel:
                try:
                    recs = run_llm_pce_batch_parallel(
                        graph_dir=GRAPHS_DIR,
                        feature_dir=FEATURES_DIR,
                        families=active_families,
                        model_name=llm_model,
                        qaoa_p=llm_p,
                    )
                    results["llm_results"] = recs
                    if verbose:
                        print(f"  LLM parallel complete: {len(recs)} graphs")
                except Exception as exc:
                    msg = f"LLM parallel batch failed: {exc}"
                    errors.append(msg)
                    if verbose:
                        print(f"  ERROR: {msg}")
                    if fail_fast:
                        raise
            else:
                try:
                    recs = run_llm_pce_batch(
                        graph_dir=GRAPHS_DIR,
                        feature_dir=FEATURES_DIR,
                        families=active_families,
                        model_name=llm_model,
                        qaoa_p=llm_p,
                    )
                    results["llm_results"] = recs
                    if verbose:
                        print(f"  LLM sequential complete: {len(recs)} graphs")
                except Exception as exc:
                    msg = f"LLM batch failed: {exc}"
                    errors.append(msg)
                    if verbose:
                        print(f"  ERROR: {msg}")
                    if fail_fast:
                        raise
        else:
            if verbose:
                print(f"  All LLM results already exist — skipping.\n")

    # ── Final status ─────────────────────────────────────────────────
    results["graphs_processed"] = len(grid_ids)
    if verbose:
        print(f"\n{'='*60}")
        print(f"  Suite complete — {len(errors)} errors")
        if errors:
            for e in errors:
                print(f"    - {e}")
        # Print summary table
        final_status = suite_status(families=families, k_values=pce_k_values)
        print(f"  Graphs in grid:   {final_status['grid_size']}")
        print(f"  Graphs on disk:   {len(final_status['graphs_exist'])}")
        print(f"  Baseline QAOA:    {len(final_status['baseline_complete'])}/{final_status['grid_size']}")
        for k in pce_k_values:
            done_k = len(final_status['pce_status'][k]['complete'])
            print(f"  Manual PCE (k={k}): {done_k}/{final_status['grid_size']}")
        print(f"  LLM PCE:          {len(final_status['llm_complete'])}/{final_status['grid_size']}")
        print(f"{'='*60}\n")

    return results


def print_suite_status(
    families: list[str] | None = None,
    k_values: list[int] | None = None,
) -> None:
    """Print a human-readable status report for the experiment suite."""
    if k_values is None:
        k_values = [PCE_CONFIG.get("default_k", 2)]
    status = suite_status(families=families, k_values=k_values)

    print()
    print(f"  MaxCut Suite Status")
    print(f"  {'─' * 40}")
    print(f"  Grid size:           {status['grid_size']} graph instances")
    print(f"  Graphs on disk:      {len(status['graphs_exist'])}")
    if status['graphs_missing']:
        print(f"  Graphs to generate:  {len(status['graphs_missing'])}")
        for g in sorted(status['graphs_missing']):
            print(f"    - {g}")
    print()
    print(f"  Baseline QAOA:       {len(status['baseline_complete'])}/{status['grid_size']}")
    if status['baseline_missing']:
        print(f"    Missing: {sorted(status['baseline_missing'])[:5]}{'...' if len(status['baseline_missing']) > 5 else ''}")
    for k in k_values:
        done_k = len(status['pce_status'][k]['complete'])
        missing_k = status['pce_status'][k]['missing']
        print(f"  Manual PCE (k={k}):    {done_k}/{status['grid_size']}")
        if missing_k:
            print(f"    Missing: {sorted(missing_k)[:5]}{'...' if len(missing_k) > 5 else ''}")
    print(f"  LLM PCE:             {len(status['llm_complete'])}/{status['grid_size']}")
    if status['llm_missing']:
        print(f"    Missing: {sorted(status['llm_missing'])[:5]}{'...' if len(status['llm_missing']) > 5 else ''}")
    print()


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="MaxCut experiment suite — run baseline, PCE, and LLM-guided experiments"
    )
    parser.add_argument("--status", action="store_true",
                        help="Print status report and exit")
    parser.add_argument("--families", nargs="*", default=None,
                        help="Graph families (default: all)")
    parser.add_argument("--no-baseline", dest="baseline", action="store_false",
                        help="Skip baseline QAOA")
    parser.add_argument("--no-pce", dest="pce", action="store_false",
                        help="Skip manual PCE")
    parser.add_argument("--no-llm", dest="llm", action="store_false",
                        help="Skip LLM-guided PCE")
    parser.add_argument("--pce-k", type=int, nargs="*", default=None,
                        help="PCE k values (default from config)")
    parser.add_argument("--max-graphs", type=int, default=None,
                        help="Limit number of graphs")
    parser.add_argument("--llm-parallel", action="store_true",
                        help="Use batched GPU generation for LLM")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop on first error")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="Override LLM model name")

    args = parser.parse_args()

    if args.status:
        print_suite_status(families=args.families, k_values=args.pce_k)
    else:
        run_maxcut_suite(
            families=args.families,
            baseline=args.baseline,
            pce=args.pce,
            llm=args.llm,
            pce_k_values=args.pce_k,
            max_graphs=args.max_graphs,
            llm_parallel=args.llm_parallel,
            fail_fast=args.fail_fast,
            llm_model=args.llm_model,
        )
