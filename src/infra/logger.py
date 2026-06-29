"""Experiment logging utilities — structured JSON logs."""

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import EXPERIMENTS_DIR


@dataclass
class ExperimentRecord:
    """Standard experiment record for QLLM runs."""
    graph_id: str
    family: str
    num_nodes: int
    num_edges: int

    # Method info
    method: str = "baseline_qaoa"          # baseline_qaoa | pce_baseline | llm_pce | rule_pce
    qaoa_p: int = 1
    optimizer: str = "COBYLA"
    max_iters: int = 200

    # Results
    optimal_energy: float | None = None
    optimal_params: list[float] | None = None
    approximation_ratio: float | None = None
    gradient_norm_at_opt: float | None = None
    convergence_iters: int | None = None
    success: bool | None = None

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    duration_seconds: float | None = None
    params: dict[str, Any] = field(default_factory=dict)  # extra graph/run params
    extra: dict[str, Any] = field(default_factory=dict)    # any additional data

    def save(self, directory: Path | None = None) -> Path:
        """Save record as JSON under `directory / {graph_id}.json`."""
        directory = directory or EXPERIMENTS_DIR
        method_dir = directory / self.method
        method_dir.mkdir(parents=True, exist_ok=True)
        path = method_dir / f"{self.graph_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        return path


def load_experiment(path: Path) -> ExperimentRecord:
    """Load an experiment record from a JSON file."""
    with open(path) as f:
        data = json.load(f)
    return ExperimentRecord(**data)


def timer(fn):
    """Decorator: records duration + saves the record after duration is set."""

    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        record = fn(*args, **kwargs)
        record.duration_seconds = round(time.perf_counter() - t0, 4)
        # Re-save with duration included (the function may have saved already)
        if hasattr(record, "save"):
            record.save()
        return record
    return wrapper
