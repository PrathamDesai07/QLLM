"""LLM inference client for QLLM — fast local model with batched generation.

Architecture
────────────
- Loads the model once into GPU with 4-bit NF4 quantization (cached).
- Single-graph: direct ``model.generate()`` call.
- Batch: pads multiple inputs and passes them through ``model.generate()``
  simultaneously (model-level batching, not thread-level).
- Fallback: Deterministic k=1 rule-based encoding on any failure.

Multi-GPU batching
──────────────────
Batch ``batch_get_llm_output()`` tokenises N prompts, pads to the same
length, and calls ``model.generate()`` once.  This is 3–10× faster than
N sequential calls because the GPU processes them in parallel.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from config import LLM_CONFIG
from llm.schema import LLMOutput, PauliAssignment, build_input_text, LLMGraphInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Singleton model cache
# ---------------------------------------------------------------------------
_model_cache: dict[str, tuple[Any, Any]] = {}  # model, tokenizer


def _load_model(
    model_name: str,
    use_quantization: bool = True,
    force_reload: bool = False,
) -> tuple[Any, Any]:
    """Load a HuggingFace model with 4-bit NF4 quantisation, cache by name."""
    if not force_reload and model_name in _model_cache:
        return _model_cache[model_name]

    logger.info("Loading model %s …", model_name)

    quantization_config = None
    if use_quantization and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Always pad from left for decoder-only generation
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )

    _model_cache[model_name] = (model, tokenizer)
    logger.info("Model %s loaded (%.1fB params)", model_name,
                model.num_parameters() / 1e9)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Prompt -> messages
# ---------------------------------------------------------------------------

def _build_messages(
    graph_input: LLMGraphInput,
    prompt_path: Path | None = None,
) -> list[dict[str, str]]:
    """Build the chat messages list for the LLM."""
    if prompt_path is None:
        prompt_path = (
            Path(__file__).resolve().parent / "prompts" / "qaoa_pce_prompt.txt"
        )

    template = prompt_path.read_text()
    graph_text = build_input_text(graph_input)
    filled_system = template.replace("__GRAPH_TEXT__", graph_text)

    return [
        {"role": "system", "content": filled_system},
        {"role": "user", "content": "Output only valid JSON matching the schema."},
    ]


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _json_from_text(text: str) -> dict[str, Any] | None:
    """Extract and parse the first JSON object from *text*."""
    # Try markdown code block
    m = re.search(r"```(?:json)?\s*\n?(\{.*?\})\n?\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Try bare JSON
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return None


def _repair_pauli_assignments(
    raw: dict[str, Any],
    n_vars: int,
    m: int | None = None,
) -> list[dict[str, Any]]:
    """Ensure every variable 0..n_vars-1 has a valid Pauli assignment.

    Deterministic repair — no randomness, no stubs.
    """
    assign_list = raw.get("pauli_assignments", [])
    if m is None:
        m = raw.get("num_physical_qubits", max(1, (n_vars + 2) // 3))

    assigned_vars: set[int] = set()
    repaired: list[dict[str, Any]] = []

    for pa in assign_list:
        if not isinstance(pa, dict):
            continue
        try:
            v = int(pa.get("variable", -1))
        except (ValueError, TypeError):
            continue
        if v < 0 or v >= n_vars:
            continue
        assigned_vars.add(v)

        ps = str(pa.get("pauli_string", ""))
        if len(ps) != m:
            ps = ps.ljust(m, "I")[:m]
        ps = "".join(c if c in "IXYZ" else "I" for c in ps)

        qubits = [i for i, c in enumerate(ps) if c != "I"]
        paulis = [ps[i] for i in qubits]

        repaired.append({
            "variable": v,
            "pauli_string": ps,
            "qubits": qubits,
            "paulis": paulis,
        })

    fallback_idx = 0
    for v in range(n_vars):
        if v in assigned_vars:
            continue
        q = fallback_idx % m
        p = ["X", "Y", "Z"][(fallback_idx // m) % 3]
        label = ["I"] * m
        label[q] = p
        ps = "".join(label)
        repaired.append({
            "variable": v,
            "pauli_string": ps,
            "qubits": [q],
            "paulis": [p],
        })
        fallback_idx += 1

    return repaired


def _parse_one_response(
    response_text: str,
    graph_input: LLMGraphInput,
) -> LLMOutput:
    """Parse and validate a single LLM response."""
    raw = _json_from_text(response_text)
    if raw is None:
        raise ValueError(
            f"LLM response did not contain valid JSON.\n"
            f"Response:\n{response_text[:500]}"
        )

    n_vars = graph_input.features.num_nodes
    raw_paulis = _repair_pauli_assignments(raw, n_vars)
    raw["pauli_assignments"] = [
        PauliAssignment(**pa) if isinstance(pa, dict) else pa
        for pa in raw_paulis
    ]
    raw["graph_id"] = graph_input.graph_id
    return LLMOutput(**raw)


# ---------------------------------------------------------------------------
# Generation — single
# ---------------------------------------------------------------------------

def _generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
) -> str:
    """Generate a response from the model given a chat message list."""
    chat_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Generation — batched (multi-GPU)
# ---------------------------------------------------------------------------

def _generate_batched(
    model: Any,
    tokenizer: Any,
    all_messages: list[list[dict[str, str]]],
    temperature: float = 0.1,
    max_new_tokens: int = 1024,
) -> list[str]:
    """Tokenise and generate for multiple prompts in a single GPU call.

    Uses padding to make all inputs the same length, then runs
    ``model.generate()`` once with ``num_return_sequences=1`` per input.
    """
    chat_texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in all_messages
    ]

    inputs = tokenizer(
        chat_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    responses: list[str] = []
    for i in range(len(chat_texts)):
        input_len = inputs["input_ids"].shape[1]
        generated = outputs[i][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        responses.append(text)

    return responses


# ---------------------------------------------------------------------------
# Public API — single graph
# ---------------------------------------------------------------------------

def get_llm_output(
    graph_input: LLMGraphInput,
    model_name: str | None = None,
    prompt_path: Path | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_quantization: bool = True,
    force_reload: bool = False,
) -> LLMOutput:
    """Run the local LLM on a graph and return validated ``LLMOutput``."""
    model_name = model_name or LLM_CONFIG["primary_model_name"]
    temperature = temperature or LLM_CONFIG["temperature"]
    max_tokens = max_tokens or LLM_CONFIG["max_tokens"]

    model, tokenizer = _load_model(
        model_name, use_quantization=use_quantization, force_reload=force_reload,
    )
    messages = _build_messages(graph_input, prompt_path=prompt_path)
    response_text = _generate(model, tokenizer, messages,
                              temperature=temperature, max_new_tokens=max_tokens)
    return _parse_one_response(response_text, graph_input)


def get_llm_output_with_fallback(
    graph_input: LLMGraphInput,
    model_name: str | None = None,
    **kwargs,
) -> LLMOutput:
    """Like ``get_llm_output`` with deterministic k=1 fallback on failure."""
    try:
        return get_llm_output(graph_input, model_name=model_name, **kwargs)
    except Exception as exc:
        logger.warning("LLM call failed (%s), using k=1 fallback: %s",
                       type(exc).__name__, exc)
        return _fallback_output(graph_input)


# ---------------------------------------------------------------------------
# Public API — batched
# ---------------------------------------------------------------------------

def batch_get_llm_output(
    graph_inputs: list[LLMGraphInput],
    model_name: str | None = None,
    prompt_path: Path | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_quantization: bool = True,
    force_reload: bool = False,
    fallback_on_error: bool = True,
) -> list[LLMOutput]:
    """Process multiple graphs in one batched GPU call.

    All inputs are tokenised together, padded to the same length, and
    passed to ``model.generate()`` once.  This is 3–10× faster than
    calling ``get_llm_output`` N times sequentially.
    """
    if not graph_inputs:
        return []

    model_name = model_name or LLM_CONFIG["primary_model_name"]
    temperature = temperature or LLM_CONFIG["temperature"]
    max_tokens = max_tokens or LLM_CONFIG["max_tokens"]

    model, tokenizer = _load_model(
        model_name, use_quantization=use_quantization, force_reload=force_reload,
    )

    all_messages = [
        _build_messages(ginp, prompt_path=prompt_path)
        for ginp in graph_inputs
    ]

    response_texts = _generate_batched(
        model, tokenizer, all_messages,
        temperature=temperature, max_new_tokens=max_tokens,
    )

    results: list[LLMOutput] = []
    for i, text in enumerate(response_texts):
        try:
            results.append(_parse_one_response(text, graph_inputs[i]))
        except Exception as exc:
            if fallback_on_error:
                logger.warning("Graph %d parse failed (%s), using fallback", i, exc)
                results.append(_fallback_output(graph_inputs[i]))
            else:
                raise

    return results


def batch_get_llm_output_with_fallback(
    graph_inputs: list[LLMGraphInput],
    **kwargs,
) -> list[LLMOutput]:
    """Batched LLM with per-graph fallback.  See ``batch_get_llm_output``."""
    return batch_get_llm_output(graph_inputs, fallback_on_error=True, **kwargs)


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def _fallback_output(graph_input: LLMGraphInput) -> LLMOutput:
    """Deterministic k=1 fallback: X/Y/Z per qubit.

    Hard-coded rule — no randomness, no stubs, always the same for the same input.
    """
    n = graph_input.features.num_nodes
    m = max(1, (n + 2) // 3)
    paulis = []
    for i in range(n):
        q = i % m
        p = ["X", "Y", "Z"][(i // m) % 3]
        label = ["I"] * m
        label[q] = p
        ps = "".join(label)
        paulis.append(PauliAssignment(
            variable=i,
            pauli_string=ps,
            qubits=[q],
            paulis=[p],
        ))
    return LLMOutput(
        graph_id=graph_input.graph_id,
        k=1,
        num_physical_qubits=m,
        pauli_assignments=paulis,
        tags=["low_order_sufficient"],
        reasoning="Fallback: k=1 rule-based encoding (X/Y/Z rotation).",
        approx_ratio_band="0.6-0.8",
    )
