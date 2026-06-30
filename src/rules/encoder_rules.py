"""Deterministic rule engine for Pauli-correlation encoding.

Translates patterns discovered in Phase 3 into explicit, interpretable
design rules that can be applied without an LLM.

Rules are derived from the quantitative analysis in ``data/analysis/patterns.json``.

Rule categories
---------------
Rule A — Correlation order selection (``rule_choose_k``)
    Determines k based on graph features (density, degree, family).

Rule B — Pauli layout strategy (``rule_choose_pauli_pattern``)
    Determines which Pauli operator distribution to use (balanced X/Y/Z,
    X/Z-dominant, etc.) based on graph structure.

Rule C — Gradient-risk avoidance (``rule_avoid_barren_plateaus``)
    Flags configurations likely to produce vanishing gradients.

Rule D — Full recommendation (``recommend_encoding``)
    Combines all rules into a complete encoding recommendation dict
    that can be passed to ``pce.manual_pce.encode_graph`` (or the
    rule-based pipeline) to build the actual Pauli strings.

Usage
-----
    python -m src.rules.encoder_rules --graph-id erdos_renyi_8_0
    python -m src.rules.encoder_rules --batch       # all graphs
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

from config import FEATURES_DIR, DATA_DIR
from graphs.features import extract_all


PATTERNS_PATH = DATA_DIR / "analysis" / "patterns.json"


# ── Rule thresholds (derived from patterns.json analysis) ──────────────

# Density breakpoints from Phase 3 pattern analysis:
#   bins 0-1 (low density) → k=2 is best
#   bins 2-3 (high density) → k=1 is competitive
DENSITY_THRESHOLD = 0.4   # below this → prefer k=2, above → prefer k=1

# Degree mean breakpoints
DEGREE_MEAN_THRESHOLD = 4.5  # below → k=2, above → k=1

# Families where baseline QAOA beats PCE
BASELINE_PREFERRED_FAMILIES = {"bipartite", "community"}

# Families where PCE k=2 is the best method
PCE_K2_PREFERRED_FAMILIES = {"erdos_renyi", "d_regular"}

# Category-specific: families where k=3 might help (dense modular graphs)
# From Phase 3: modular graphs with high density may need higher-order
K3_CANDIDATE_DENSITY = 0.5
K3_CANDIDATE_MODULARITY = 0.4

# Gradient thresholds
VANISHING_GRADIENT_THRESHOLD = 0.01

# Minimum average non-identity operators per variable to avoid barren plateaus
MIN_NON_ID_PER_VAR = 0.3


# ── Rule A: Choose correlation order k ────────────────────────────────

def rule_choose_k(features: dict[str, Any],
                  family: str | None = None) -> dict[str, Any]:
    """Determine the recommended correlation order k from graph features.

    Returns
    -------
    dict with keys:
        k (int), confidence (str: high|medium|low),
        reasoning (str), rule_name (str)
    """
    density = features.get("density")
    degree_mean = features.get("degree_mean")
    transitivity = features.get("transitivity")
    modularity_val = features.get("modularity")

    # Rule 1: Family-based override
    if family and family in PCE_K2_PREFERRED_FAMILIES:
        return {
            "k": 2,
            "confidence": "high",
            "reasoning": f"Family '{family}' has best empirical results with k=2",
            "rule_name": "family_k2",
        }

    if family and family in BASELINE_PREFERRED_FAMILIES:
        return {
            "k": 1,
            "confidence": "medium",
            "reasoning": f"Family '{family}' performs best with uncompressed QAOA; "
                         f"k=1 as fallback PCE",
            "rule_name": "family_baseline_fallback",
        }

    # Rule 2: Density-based
    if density is not None:
        if density < DENSITY_THRESHOLD:
            # Sparse → k=2 captures more edge structure
            reason = f"Low density ({density:.3f} < {DENSITY_THRESHOLD}) favours k=2"
            return {"k": 2, "confidence": "high",
                    "reasoning": reason, "rule_name": "density_low_k2"}
        else:
            # Dense → k=1 sufficient
            reason = f"High density ({density:.3f} >= {DENSITY_THRESHOLD}) favours k=1"
            # But check if k=3 would help for very dense modular graphs
            if (modularity_val is not None and modularity_val > K3_CANDIDATE_MODULARITY
                    and density > K3_CANDIDATE_DENSITY and transitivity is not None
                    and transitivity > 0.3):
                return {
                    "k": 3,
                    "confidence": "low",
                    "reasoning": f"Dense modular graph (d={density:.3f}, "
                                 f"modularity={modularity_val:.3f}) may benefit "
                                 f"from k=3",
                    "rule_name": "dense_modular_k3",
                }
            return {"k": 1, "confidence": "medium",
                    "reasoning": reason, "rule_name": "density_high_k1"}

    # Rule 3: Degree-based fallback
    if degree_mean is not None:
        if degree_mean < DEGREE_MEAN_THRESHOLD:
            return {"k": 2, "confidence": "medium",
                    "reasoning": f"Low mean degree ({degree_mean:.2f}) favours k=2",
                    "rule_name": "degree_low_k2"}
        else:
            return {"k": 1, "confidence": "low",
                    "reasoning": f"High mean degree ({degree_mean:.2f}) favours k=1",
                    "rule_name": "degree_high_k1"}

    # Default
    return {"k": 2, "confidence": "low",
            "reasoning": "Default fallback: k=2",
            "rule_name": "default_k2"}


# ── Rule B: Choose Pauli layout pattern ──────────────────────────────

def rule_choose_pauli_pattern(features: dict[str, Any],
                              k: int) -> dict[str, Any]:
    """Determine the Pauli operator distribution strategy.

    Returns
    -------
    dict with keys:
        strategy (str), description (str), use_x, use_y, use_z (bool),
        bias_weight (float), reasoning (str)
    """
    density = features.get("density", 0.5)
    modularity_val = features.get("modularity", 0.0)
    degree_mean = features.get("degree_mean", 3.0)

    if k == 3:
        # k=3: use all three Paulis evenly for maximum expressivity
        return {
            "strategy": "balanced_xyz",
            "description": "Even X/Y/Z distribution for high-order encoding",
            "use_x": True, "use_y": True, "use_z": True,
            "bias_weight": 1.0,
            "reasoning": "k=3 encoding uses all Pauli operators evenly",
        }

    if k == 2:
        if modularity_val is not None and modularity_val > 0.3:
            # Modular graphs: use X and Z more (less Y) — Pattern B style
            return {
                "strategy": "xz_dominant",
                "description": "X/Z dominant with reduced Y for modular graphs",
                "use_x": True, "use_y": False, "use_z": True,
                "bias_weight": 1.0,
                "reasoning": f"Modular graph (modularity={modularity_val:.3f}) "
                             f"benefits from X/Z-dominant layout",
            }
        elif density is not None and density < 0.25:
            # Very sparse: use balanced mix — Pattern A style
            return {
                "strategy": "balanced_xyz",
                "description": "Even X/Y/Z mix for sparse graphs",
                "use_x": True, "use_y": True, "use_z": True,
                "bias_weight": 1.0,
                "reasoning": "Sparse graph benefits from balanced Pauli mix",
            }
        else:
            # Default k=2: balanced with slight X preference
            return {
                "strategy": "slight_x_bias",
                "description": "Balanced with slight X bias",
                "use_x": True, "use_y": True, "use_z": True,
                "bias_weight": 1.2,  # X gets 20% more assignments
                "reasoning": "Default k=2 layout with slight X preference",
            }

    # k=1: use X and Z (avoid Y for simplicity)
    if modularity_val is not None and modularity_val > 0.3:
        # Modular dense: X-only for simplicity
        return {
            "strategy": "x_only",
            "description": "X-only layout for modular graphs at k=1",
            "use_x": True, "use_y": False, "use_z": False,
            "bias_weight": 1.0,
            "reasoning": "Modular graph benefits from single-Pauli (X) encoding",
        }
    else:
        # Default k=1: X and Z
        return {
            "strategy": "xz_mix",
            "description": "X and Z mix for k=1 encoding",
            "use_x": True, "use_y": False, "use_z": True,
            "bias_weight": 1.0,
            "reasoning": "X/Z mix for k=1 encoding on non-modular graph",
        }


# ── Rule C: Gradient-risk avoidance ──────────────────────────────────

def rule_avoid_barren_plateaus(features: dict[str, Any],
                               k: int) -> dict[str, Any]:
    """Check if the configuration risks vanishing gradients.

    Returns
    -------
    dict with keys:
        risk_level (str: low|medium|high),
        min_ops_per_var (int), reasoning (str)
    """
    degree_mean = features.get("degree_mean", 3.0)
    density = features.get("density", 0.3)
    num_nodes = features.get("num_nodes", 8)

    risk_factors: list[str] = []
    risk_score = 0.0

    # Low degree → higher gradient risk (ρ = +0.57 with gradient norm)
    if degree_mean is not None and degree_mean < 3.0:
        risk_factors.append(f"Low degree_mean ({degree_mean:.2f})")
        risk_score += 0.3
    elif degree_mean is not None and degree_mean > 6.0:
        risk_score -= 0.2  # healthy

    # Small graphs → higher risk (ρ = +0.36)
    if num_nodes is not None and num_nodes <= 8:
        risk_factors.append(f"Small graph (n={num_nodes})")
        risk_score += 0.2

    # Low density → higher risk (ρ = +0.33)
    if density is not None and density < 0.2:
        risk_factors.append(f"Low density ({density:.3f})")
        risk_score += 0.2

    # Higher k → more non-identity ops → healthier (ρ = -0.32)
    # k=1 encodings have fewer non-identity ops on average
    if k == 1:
        risk_score += 0.2

    k_ops = {1: 1, 2: 2, 3: 3}
    min_ops = k_ops.get(k, 2)

    # Determine risk level
    if risk_score >= 0.5:
        risk_level = "high"
    elif risk_score >= 0.2:
        risk_level = "medium"
    else:
        risk_level = "low"

    reasoning_parts = risk_factors if risk_factors else ["No significant risk factors"]
    reasoning = "; ".join(reasoning_parts) + f" (risk score: {risk_score:.2f})"

    return {
        "risk_level": risk_level,
        "min_ops_per_var": min_ops,
        "risk_score": round(risk_score, 2),
        "reasoning": reasoning,
    }


# ── Rule D: Full recommendation ──────────────────────────────────────

def recommend_encoding(graph_id: str,
                        features: dict[str, Any] | None = None,
                        graph: nx.Graph | None = None,
                        feature_dir: Path | None = None) -> dict[str, Any]:
    """Generate a complete deterministic encoding recommendation.

    Loads features if not provided, runs all rules, and returns a
    recommendation dict compatible with the PCE pipeline.

    Parameters
    ----------
    graph_id : str
    features : dict | None
        Pre-loaded feature dict. If None, loaded from disk.
    graph : nx.Graph | None
        The graph object. If None and features is provided, still works.
    feature_dir : Path | None
        Directory containing feature JSONs.

    Returns
    -------
    dict with keys:
        graph_id, family, features (summary),
        k_recommendation (from rule_choose_k),
        pauli_recommendation (from rule_choose_pauli_pattern),
        gradient_risk (from rule_avoid_barren_plateaus),
        encoding (complete encoding dict for PCE pipeline)
    """
    if features is None:
        if feature_dir is None:
            feature_dir = FEATURES_DIR
        feat_path = feature_dir / f"{graph_id}.json"
        if feat_path.exists():
            with open(feat_path) as f:
                features = json.load(f)
        else:
            # Load graph and extract features
            from config import GRAPHS_DIR
            graph_path = GRAPHS_DIR / f"{graph_id}.json"
            with open(graph_path) as f:
                data = json.load(f)
            graph = nx.node_link_graph(data)
            features = extract_all(graph)
            features["graph_id"] = graph_id
            features["family"] = data.get("metadata", {}).get("family", "unknown")
            features["params"] = data.get("metadata", {}).get("params", {})

    family = features.get("family", "unknown")

    # Run all rules
    k_rule = rule_choose_k(features, family=family)
    k = k_rule["k"]

    pauli_rule = rule_choose_pauli_pattern(features, k)
    gradient_rule = rule_avoid_barren_plateaus(features, k)

    # Build encoding strategy info
    encoding_strategy = {
        "k": k,
        "strategy": pauli_rule["strategy"],
        "num_variables": features.get("num_nodes", 0),
        "rule_name": k_rule["rule_name"],
        "pauli_description": pauli_rule["description"],
        "use_x": pauli_rule["use_x"],
        "use_y": pauli_rule["use_y"],
        "use_z": pauli_rule["use_z"],
        "bias_weight": pauli_rule.get("bias_weight", 1.0),
    }

    return {
        "graph_id": graph_id,
        "family": family,
        "feature_summary": {
            "num_nodes": features.get("num_nodes"),
            "num_edges": features.get("num_edges"),
            "density": features.get("density"),
            "degree_mean": features.get("degree_mean"),
            "avg_clustering": features.get("avg_clustering"),
            "modularity": features.get("modularity"),
        },
        "k_recommendation": k_rule,
        "pauli_recommendation": pauli_rule,
        "gradient_risk": gradient_rule,
        "encoding_strategy": encoding_strategy,
    }


# ── Batch ──────────────────────────────────────────────────────────────

def batch_recommend(graph_ids: list[str] | None = None,
                    feature_dir: Path | None = None) -> list[dict[str, Any]]:
    """Recommend encodings for all (or given) graph IDs."""
    if graph_ids is None:
        feature_dir = feature_dir or FEATURES_DIR
        graph_ids = sorted(f.stem for f in feature_dir.glob("*.json"))
    return [recommend_encoding(gid, feature_dir=feature_dir) for gid in graph_ids]


# ── CLI ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="QLLM deterministic rule engine for PCE encoding"
    )
    parser.add_argument("--graph-id", type=str, default=None,
                        help="Single graph ID")
    parser.add_argument("--batch", action="store_true",
                        help="Run on all graphs")
    parser.add_argument("--feature-dir", type=str, default=None)
    args = parser.parse_args()

    if args.graph_id:
        rec = recommend_encoding(
            args.graph_id,
            feature_dir=Path(args.feature_dir) if args.feature_dir else None,
        )
        print(json.dumps(rec, indent=2))
    elif args.batch:
        recs = batch_recommend(
            feature_dir=Path(args.feature_dir) if args.feature_dir else None,
        )
        print(f"Recommended encodings for {len(recs)} graphs:")
        for r in recs:
            k = r["k_recommendation"]
            g = r["gradient_risk"]
            print(f"  {r['graph_id']:25s} k={k['k']} [{k['rule_name']:25s}] "
                  f"risk={g['risk_level']} [{g['risk_score']:.2f}]")
    else:
        parser.print_help()
