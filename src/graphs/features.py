"""Graph feature extraction for QLLM."""

import json
from pathlib import Path

import networkx as nx
import numpy as np

from config import FEATURES_DIR


# ── Feature Extractors ─────────────────────────────────────────────────

def basic_stats(g: nx.Graph) -> dict:
    """Basic structural properties."""
    n = g.number_of_nodes()
    m = g.number_of_edges()
    density = nx.density(g)
    degrees = [d for _, d in g.degree()]
    return {
        "num_nodes": n,
        "num_edges": m,
        "density": round(density, 6),
        "degree_min": int(min(degrees)),
        "degree_max": int(max(degrees)),
        "degree_mean": round(float(np.mean(degrees)), 4),
        "degree_std": round(float(np.std(degrees)), 4),
    }


def degree_histogram(g: nx.Graph, bins: int = 10) -> dict:
    """Binned degree distribution."""
    degrees = [d for _, d in g.degree()]
    hist, edges = np.histogram(degrees, bins=bins)
    return {
        "degree_histogram": hist.tolist(),
        "degree_bin_edges": edges.tolist(),
    }


def clustering(g: nx.Graph) -> dict:
    """Average clustering coefficient and global transitivity."""
    return {
        "avg_clustering": round(float(nx.average_clustering(g)), 6),
        "transitivity": round(float(nx.transitivity(g)), 6),
    }


def connectivity(g: nx.Graph) -> dict:
    """Connected components analysis."""
    components = list(nx.connected_components(g))
    sizes = sorted([len(c) for c in components], reverse=True)
    return {
        "num_components": len(components),
        "largest_component_size": sizes[0] if sizes else 0,
        "component_size_ratio": round(sizes[0] / g.number_of_nodes(), 6)
        if g.number_of_nodes() else 0,
    }


def algebraic_connectivity(g: nx.Graph) -> dict:
    """Fiedler eigenvalue — measure of graph connectivity."""
    try:
        # laplacian spectrum, second smallest eigenvalue
        # use the normalized laplacian for better conditioning
        L = nx.normalized_laplacian_matrix(g).toarray()
        eigenvalues = sorted(np.linalg.eigvalsh(L))
        val = round(float(eigenvalues[1]), 6) if len(eigenvalues) > 1 else 0.0
    except Exception:
        val = None
    return {"algebraic_connectivity": val}


def extract_all(g: nx.Graph) -> dict:
    """Aggregate all feature extractors into a single dict."""
    features = {}
    features.update(basic_stats(g))
    features.update(degree_histogram(g))
    features.update(clustering(g))
    features.update(connectivity(g))
    features.update(algebraic_connectivity(g))
    return features


# ── Persistence ────────────────────────────────────────────────────────

def features_from_graph_file(
    path: Path, directory: Path | None = None
) -> Path:
    """Load a graph JSON, extract features, save feature JSON."""
    directory = directory or FEATURES_DIR
    directory.mkdir(parents=True, exist_ok=True)

    with open(path) as f:
        data = json.load(f)

    g = nx.node_link_graph(data)
    meta = data.get("metadata", {})
    graph_id = meta.get("graph_id", path.stem)

    features = extract_all(g)
    features["graph_id"] = graph_id
    features["family"] = meta.get("family", "unknown")
    features["params"] = meta.get("params", {})

    out_path = directory / f"{graph_id}.json"
    with open(out_path, "w") as f:
        json.dump(features, f, indent=2)
    return out_path


def batch_extract(graph_dir: Path | None = None,
                  feature_dir: Path | None = None) -> list[Path]:
    """Extract features for every graph in graph_dir."""
    graph_dir = graph_dir or FEATURES_DIR.parent / "graphs"
    feature_dir = feature_dir or FEATURES_DIR
    paths: list[Path] = []
    for gpath in sorted(graph_dir.glob("*.json")):
        out = features_from_graph_file(gpath, directory=feature_dir)
        paths.append(out)
    return paths


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract graph features")
    parser.add_argument("--graph-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    gdir = Path(args.graph_dir) if args.graph_dir else None
    odir = Path(args.output_dir) if args.output_dir else None
    paths = batch_extract(gdir, odir)
    print(f"Extracted features for {len(paths)} graphs:")
    for p in paths:
        print(f"  {p}")
