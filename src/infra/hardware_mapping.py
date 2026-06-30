"""Hardware mapping for NISQ devices — topology, noise profiles, and transpilation.

Provides realistic device models based on IBM Quantum hardware for simulating
QAOA/PCE circuits under device-appropriate noise.

Usage
-----
    python -m src.infra.hardware_mapping --list-devices
"""

import sys
from pathlib import Path
from typing import Any

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    depolarizing_error,
    thermal_relaxation_error,
    ReadoutError,
)


# ── Device Registry ────────────────────────────────────────────────────

DEVICES: dict[str, dict[str, Any]] = {}


def _register(name: str, n_qubits: int,
              t1_us: float, t2_us: float,
              gate1q_err: float, gate2q_err: float,
              readout_err: float,
              description: str) -> None:
    DEVICES[name] = {
        "name": name,
        "n_qubits": n_qubits,
        "t1_us": t1_us,
        "t2_us": t2_us,
        "gate1q_err": gate1q_err,
        "gate2q_err": gate2q_err,
        "readout_err": readout_err,
        "description": description,
    }


_register("ibm_eagle", 27,
          t1_us=280.0, t2_us=150.0,
          gate1q_err=0.0002, gate2q_err=0.006,
          readout_err=0.01,
          description="IBM Eagle-like (27 qubits, heavy-hex, typical noise)")

_register("ibm_heron", 27,
          t1_us=350.0, t2_us=200.0,
          gate1q_err=0.00015, gate2q_err=0.004,
          readout_err=0.008,
          description="IBM Heron-like (improved coherence)")

_register("ideal", 100,
          t1_us=1e9, t2_us=1e9,
          gate1q_err=0.0, gate2q_err=0.0, readout_err=0.0,
          description="Noiseless ideal simulator")


# ── Noise Model Builder ───────────────────────────────────────────────

def _gate_duration_ns(gate_name: str) -> float:
    durations = {
        "id": 35.0, "sx": 35.0, "x": 35.0,
        "rz": 0.0, "cx": 300.0,
        "rx": 35.0, "h": 35.0,
        "measure": 1000.0, "reset": 1000.0,
    }
    return durations.get(gate_name, 100.0)


def build_noise_model(device_name: str = "ibm_eagle") -> NoiseModel:
    """Build a Qiskit Aer NoiseModel for a given device."""
    if device_name not in DEVICES:
        raise ValueError(f"Unknown device '{device_name}'. "
                         f"Available: {list(DEVICES)}")
    cfg = DEVICES[device_name]
    nm = NoiseModel()
    t1 = cfg["t1_us"] * 1e-6
    t2 = cfg["t2_us"] * 1e-6
    nq = cfg["n_qubits"]

    for q in range(nq):
        therm = thermal_relaxation_error(t1, t2, _gate_duration_ns("sx") * 1e-9)
        depol = depolarizing_error(cfg["gate1q_err"], 1)
        nm.add_quantum_error(therm.compose(depol), ["sx", "x", "h", "rx"], [q])

    for q in range(nq - 1):
        gd = _gate_duration_ns("cx") * 1e-9
        t2q = thermal_relaxation_error(t1, t2, gd).expand(
               thermal_relaxation_error(t1, t2, gd))
        d2q = depolarizing_error(cfg["gate2q_err"], 2)
        nm.add_quantum_error(t2q.compose(d2q), ["cx"], [q, q + 1])

    for q in range(nq):
        nm.add_readout_error(ReadoutError(
            [[1 - cfg["readout_err"], cfg["readout_err"]],
             [cfg["readout_err"], 1 - cfg["readout_err"]]]), [q])

    for q in range(nq):
        nm.add_quantum_error(
            thermal_relaxation_error(t1, t2, _gate_duration_ns("id") * 1e-9),
            ["id"], [q])

    return nm


def list_devices() -> None:
    print(f"  {'Device':25s} {'Qubits':>7s} {'T1 (µs)':>9s} {'T2 (µs)':>9s} "
          f"{'1Q err':>9s} {'2Q err':>9s} {'RO err':>7s}")
    print(f"  {'─' * 76}")
    for name, c in sorted(DEVICES.items()):
        print(f"  {name:25s} {c['n_qubits']:>7d} {c['t1_us']:>9.1f} "
              f"{c['t2_us']:>9.1f} {c['gate1q_err']:>9.5f} "
              f"{c['gate2q_err']:>9.5f} {c['readout_err']:>7.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Hardware mapping for QLLM")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()
    if args.list_devices:
        list_devices()
    else:
        parser.print_help()
