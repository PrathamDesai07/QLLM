"""LLM input/output schemas for QLLM's Pauli-correlation encoding pipeline."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import networkx as nx

from graphs.features import extract_all


# ── Input Schema ───────────────────────────────────────────────────────

@dataclass
class GraphFeatures:
    """Numerical features describing a graph instance."""
    num_nodes: int
    num_edges: int
    density: float
    degree_min: int
    degree_max: int
    degree_mean: float
    degree_std: float
    avg_clustering: float
    transitivity: float
    num_components: int
    largest_component_size: int
    component_size_ratio: float
    algebraic_connectivity: float | None = None

    @classmethod
    def from_feature_dict(cls, d: dict) -> "GraphFeatures":
        """Build from a features dict produced by ``graphs.features.extract_all``."""
        return cls(
            num_nodes=d["num_nodes"],
            num_edges=d["num_edges"],
            density=d["density"],
            degree_min=d["degree_min"],
            degree_max=d["degree_max"],
            degree_mean=d["degree_mean"],
            degree_std=d["degree_std"],
            avg_clustering=d["avg_clustering"],
            transitivity=d["transitivity"],
            num_components=d["num_components"],
            largest_component_size=d["largest_component_size"],
            component_size_ratio=d["component_size_ratio"],
            algebraic_connectivity=d.get("algebraic_connectivity"),
        )


@dataclass
class AdjacencyInfo:
    """Compact representation of a graph's adjacency structure.

    ``edge_list`` stores every edge as (source, target) with source < target.
    This is the exact edge set, not randomised or sampled.
    """
    edge_list: list[tuple[int, int]]

    @classmethod
    def from_graph(cls, g: nx.Graph) -> "AdjacencyInfo":
        edges = sorted((min(u, v), max(u, v)) for u, v in g.edges())
        return cls(edge_list=edges)

    def to_labels(self, max_nodes: int) -> list[str]:
        """Return a human-readable adjacency representation for the LLM prompt."""
        return [f"{u}-{v}" for u, v in self.edge_list]


@dataclass
class LLMGraphInput:
    """Complete input to the LLM for a single graph instance."""
    graph_id: str
    family: str
    features: GraphFeatures
    adjacency: AdjacencyInfo
    # Optional degree histogram (binned) for richer structural context
    degree_histogram: list[int] | None = None
    degree_bin_edges: list[float] | None = None

    @classmethod
    def from_graph(
        cls,
        g: nx.Graph,
        graph_id: str,
        family: str = "unknown",
        params: dict | None = None,
    ) -> "LLMGraphInput":
        """Build input schema directly from a networkx graph object."""
        feature_dict = extract_all(g)
        if params:
            feature_dict["params"] = params
        return cls(
            graph_id=graph_id,
            family=family,
            features=GraphFeatures.from_feature_dict(feature_dict),
            adjacency=AdjacencyInfo.from_graph(g),
            degree_histogram=feature_dict.get("degree_histogram"),
            degree_bin_edges=feature_dict.get("degree_bin_edges"),
        )

    @classmethod
    def from_graph_file(cls, path: Path) -> "LLMGraphInput":
        """Build from a saved graph JSON file (node-link format with metadata)."""
        with open(path) as f:
            data = json.load(f)
        g = nx.node_link_graph(data)
        meta = data.get("metadata", {})
        return cls.from_graph(
            g,
            graph_id=meta.get("graph_id", path.stem),
            family=meta.get("family", "unknown"),
            params=meta.get("params"),
        )

    @classmethod
    def from_feature_file(cls, feature_path: Path) -> "LLMGraphInput":
        """Build from a saved features JSON file.

        Note: adjacency will be empty — use from_graph_file if edges are needed.
        """
        with open(feature_path) as f:
            d = json.load(f)
        return cls(
            graph_id=d["graph_id"],
            family=d.get("family", "unknown"),
            features=GraphFeatures.from_feature_dict(d),
            adjacency=AdjacencyInfo(edge_list=[]),
            degree_histogram=d.get("degree_histogram"),
            degree_bin_edges=d.get("degree_bin_edges"),
        )


# ── Output Schema ──────────────────────────────────────────────────────

# Valid classification tags that the LLM may assign.
VALID_TAGS = frozenset({
    "low_order_sufficient",
    "needs_higher_order",
    "gradient_risk",
})


@dataclass
class PauliAssignment:
    """Pauli string assigned to one graph variable (node)."""
    variable: int
    pauli_string: str          # e.g. "XIZ", "ZYX", "IIZ"
    qubits: list[int]          # indices of non-identity qubits
    paulis: list[str]          # Pauli operators on those qubits


@dataclass
class LLMOutput:
    """Structured output from the LLM for Pauli-correlation encoding."""
    graph_id: str
    k: int                         # compression order
    num_physical_qubits: int
    pauli_assignments: list[PauliAssignment]
    tags: list[str] = field(default_factory=list)
    reasoning: str = ""            # LLM's free-text explanation
    approx_ratio_band: str = ""    # e.g. "0.7-0.8"

    def __post_init__(self):
        if not (1 <= self.k <= 5):
            raise ValueError(f"k must be between 1 and 5, got {self.k}")
        if self.num_physical_qubits < 1:
            raise ValueError(f"num_physical_qubits must be >= 1, got {self.num_physical_qubits}")
        for tag in self.tags:
            if tag not in VALID_TAGS:
                raise ValueError(f"Invalid tag '{tag}'. Valid: {sorted(VALID_TAGS)}")
        # validate Pauli string length matches num_physical_qubits
        for pa in self.pauli_assignments:
            if len(pa.pauli_string) != self.num_physical_qubits:
                raise ValueError(
                    f"Pauli string '{pa.pauli_string}' has length {len(pa.pauli_string)}, "
                    f"expected {self.num_physical_qubits}"
                )
            for ch in pa.pauli_string:
                if ch not in "IXYZ":
                    raise ValueError(f"Invalid Pauli character '{ch}' in '{pa.pauli_string}'")

    def to_encoding_dict(self) -> dict[str, Any]:
        """Convert to a PCE encoding dict compatible with ``pce.manual_pce``."""
        variable_to_pauli_map: dict[str, dict] = {}
        for pa in self.pauli_assignments:
            variable_to_pauli_map[str(pa.variable)] = {
                "qubits": pa.qubits,
                "paulis": pa.paulis,
                "pauli_string": pa.pauli_string,
            }
        return {
            "k": self.k,
            "num_physical_qubits": self.num_physical_qubits,
            "num_variables": len(self.pauli_assignments),
            "compression_ratio": round(
                len(self.pauli_assignments) / self.num_physical_qubits, 2
            ) if self.num_physical_qubits else 1.0,
            "strategy": f"llm_k{self.k}",
            "pauli_strings": {
                str(i): pa.pauli_string
                for i, pa in enumerate(self.pauli_assignments)
            },
            "variable_to_pauli_map": variable_to_pauli_map,
            "tags": self.tags,
            "reasoning": self.reasoning,
            "approx_ratio_band": self.approx_ratio_band,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), indent=indent)

    @classmethod
    def from_json(cls, s: str) -> "LLMOutput":
        """Deserialize from JSON string."""
        d = json.loads(s)
        d["pauli_assignments"] = [
            PauliAssignment(**pa) for pa in d["pauli_assignments"]
        ]
        return cls(**d)

    def save(self, path: Path) -> Path:
        """Save output as JSON."""
        with open(path, "w") as f:
            f.write(self.to_json())
        return path

    @classmethod
    def load(cls, path: Path) -> "LLMOutput":
        """Load output from JSON file."""
        return cls.from_json(path.read_text())


# ── Prompt Helpers ─────────────────────────────────────────────────────

def build_input_text(inp: LLMGraphInput) -> str:
    """Build a structured text representation of the graph for the LLM prompt."""
    lines = [
        f"Graph ID: {inp.graph_id}",
        f"Family: {inp.family}",
        f"Nodes: {inp.features.num_nodes}, Edges: {inp.features.num_edges}",
        f"Density: {inp.features.density}",
        f"Degree range: {inp.features.degree_min} - {inp.features.degree_max}",
        f"Degree mean: {inp.features.degree_mean}, std: {inp.features.degree_std}",
        f"Avg clustering: {inp.features.avg_clustering}",
        f"Transitivity: {inp.features.transitivity}",
        f"Components: {inp.features.num_components} (largest: {inp.features.largest_component_size})",
        f"Algebraic connectivity: {inp.features.algebraic_connectivity}",
        "Edges:",
    ]
    for u, v in inp.adjacency.edge_list:
        lines.append(f"  {u} - {v}")
    return "\n".join(lines)
