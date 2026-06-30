"""Graph generation for QLLM — canonical graph families with metadata."""

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

from config import GRAPHS_DIR, GRAPH_DEFAULTS


# ── Generators ─────────────────────────────────────────────────────────

def generate_erdos_renyi(n: int, p: float, seed: int | None = None) -> nx.Graph:
    """Erdős–Rényi G(n, p) random graph."""
    return nx.erdos_renyi_graph(n, p, seed=seed)


def generate_d_regular(n: int, d: int, seed: int | None = None) -> nx.Graph:
    """d-regular random graph (n * d must be even)."""
    if n * d % 2 != 0:
        raise ValueError(f"n * d must be even, got n={n}, d={d}")
    return nx.random_regular_graph(d, n, seed=seed)


def generate_community(
    n: int,
    num_communities: int,
    p_in: float = 0.8,
    p_out: float = 0.05,
    seed: int | None = None,
) -> nx.Graph:
    """Community-structured graph via planted partition (stochastic block model)."""
    sizes = [n // num_communities] * num_communities
    # distribute remainder
    for i in range(n % num_communities):
        sizes[i] += 1
    probs = [[p_in if i == j else p_out for j in range(num_communities)]
             for i in range(num_communities)]
    return nx.stochastic_block_model(sizes, probs, seed=seed)


def generate_bipartite(
    n_left: int, n_right: int, p: float, seed: int | None = None
) -> nx.Graph:
    """Bipartite random graph."""
    return nx.bipartite.random_graph(n_left, n_right, p, seed=seed)


# ── Dispatch ───────────────────────────────────────────────────────────

GENERATORS = {
    "erdos_renyi": generate_erdos_renyi,
    "d_regular": generate_d_regular,
    "community": generate_community,
    "bipartite": generate_bipartite,
}


def generate_graph(family: str, **kwargs) -> nx.Graph:
    """Generate a graph by family name.  Raises KeyError on unknown family."""
    if family not in GENERATORS:
        raise KeyError(
            f"Unknown graph family '{family}'. "
            f"Available: {list(GENERATORS)}"
        )
    return GENERATORS[family](**kwargs)


# ── Persistence ────────────────────────────────────────────────────────

def graph_metadata(g: nx.Graph, family: str, params: dict) -> dict:
    """Build a metadata dict for a graph instance."""
    return {
        "graph_id": f"{family}_{g.number_of_nodes()}_{params.get('seed', 0)}",
        "family": family,
        "params": params,
        "num_nodes": g.number_of_nodes(),
        "num_edges": g.number_of_edges(),
        "density": nx.density(g),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _json_default(obj: object) -> object:
    """JSON encoder fallback for non-serializable types (set, numpy, etc.)."""
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    raise TypeError(f"Type {type(obj)} not serializable")


def save_graph(g: nx.Graph, family: str, params: dict,
               directory: Path | None = None) -> Path:
    """Save graph as JSON (node-link format) alongside metadata."""
    directory = directory or GRAPHS_DIR
    directory.mkdir(parents=True, exist_ok=True)

    meta = graph_metadata(g, family, params)
    out = nx.node_link_data(g)
    out["metadata"] = meta

    path = directory / f"{meta['graph_id']}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=_json_default)
    return path


# ── Batch Generation ───────────────────────────────────────────────────

def generate_batch(
    families: list[str] | None = None,
    min_nodes: int | None = None,
    max_nodes: int | None = None,
    step: int | None = None,
    instances_per_size: int | None = None,
    seed_offset: int = 0,
    directory: Path | None = None,
) -> list[Path]:
    """Generate a batch of graphs across families and sizes."""
    families = families or GRAPH_DEFAULTS["families"]
    min_nodes = min_nodes or GRAPH_DEFAULTS["min_nodes"]
    max_nodes = max_nodes or GRAPH_DEFAULTS["max_nodes"]
    step = step or GRAPH_DEFAULTS["step"]
    instances_per_size = instances_per_size or GRAPH_DEFAULTS["instances_per_size"]

    paths: list[Path] = []
    for family in families:
        for n in range(min_nodes, max_nodes + 1, step):
            for inst in range(instances_per_size):
                seed = seed_offset + inst
                rng = random.Random(seed)

                if family == "erdos_renyi":
                    p = rng.uniform(0.3, 0.7)
                    g = generate_erdos_renyi(n, p, seed=seed)
                    params = {"n": n, "p": round(p, 3), "seed": seed}
                elif family == "d_regular":
                    d = min(max(2, n // 4), n - 1)
                    if n * d % 2 != 0:
                        d += 1
                    g = generate_d_regular(n, d, seed=seed)
                    params = {"n": n, "d": d, "seed": seed}
                elif family == "community":
                    nc = max(2, n // 5)
                    g = generate_community(n, nc, seed=seed)
                    params = {"n": n, "num_communities": nc, "seed": seed}
                elif family == "bipartite":
                    n_left = n // 2
                    n_right = n - n_left
                    p = rng.uniform(0.3, 0.7)
                    g = generate_bipartite(n_left, n_right, p, seed=seed)
                    params = {"n_left": n_left, "n_right": n_right,
                              "p": round(p, 3), "seed": seed}
                else:
                    continue

                path = save_graph(g, family, params, directory=directory)
                paths.append(path)

    return paths


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate graphs for QLLM")
    parser.add_argument("--families", nargs="*",
                        default=GRAPH_DEFAULTS["families"])
    parser.add_argument("--min-nodes", type=int,
                        default=GRAPH_DEFAULTS["min_nodes"])
    parser.add_argument("--max-nodes", type=int,
                        default=GRAPH_DEFAULTS["max_nodes"])
    parser.add_argument("--step", type=int, default=GRAPH_DEFAULTS["step"])
    parser.add_argument("--instances", type=int,
                        default=GRAPH_DEFAULTS["instances_per_size"])
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()
    paths = generate_batch(
        families=args.families,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        step=args.step,
        instances_per_size=args.instances,
        seed_offset=args.seed_offset,
        directory=Path(args.output_dir) if args.output_dir else None,
    )
    print(f"Generated {len(paths)} graphs:")
    for p in paths:
        print(f"  {p}")
