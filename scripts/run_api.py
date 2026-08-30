"""Resumable cross-provider runner for a frozen FG9 subset.

The runner never reads gold. It records transport failures, schema fallbacks and
format-only re-asks separately so that provider protocol failures cannot be
mistaken for model capability failures.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


RUNNER_VERSION = "FWB-FG9-API-RUNNER-v10.1-openai-chat-wire-null-safe"
PROMPT_VERSION = "FWB-FG9-PROMPT-v3-exact-set"
SCHEMA_VERSION = "FWB-FG9-RESPONSE-SCHEMA-v7-empty-choice-set"

CANONICAL_MODEL_ALIASES = {
    "claude-haiku-4-5": {"claude-haiku-4-5-20251001"},
}


def model_identity_matches(requested: str, reported: str | None) -> bool:
    if reported is None:
        return False
    return reported == requested or reported in CANONICAL_MODEL_ALIASES.get(requested, set())


def completed_qa_ids(rows: list[dict], provider: str, model: str) -> set[str]:
    return {
        row["qa_id"]
        for row in rows
        if row.get("status") in {"ok", "model_mismatch"}
        and (provider != "openai"
             or model_identity_matches(model, row.get("response_model")))
    }


TASK_FIELDS = {
    "L1-1": {"event_class": ["fire", "no_fire", "ventilation_disturbance", "sensor_fault"],
             "cause_region": [*[f"R{i}" for i in range(1, 9)], "global", "none"],
             "dominant_signal": ["temperature", "soot", "visibility", "flow", "none"],
             "secondary_signal": ["temperature", "soot", "visibility", "flow", "none"],
             "affected_region_count": list(range(9))},
    "L1-2": {"temperature_trend": ["up", "stable", "down"], "smoke_trend": ["up", "stable", "down"],
             "visibility_trend": ["up", "stable", "down"]},
    "L1-3": {"consistency": ["consistent", "inconsistent"],
             "first_violation_bin": ["none", "early", "middle", "late"],
             "violation_type": ["none", "reverse", "jump", "disappearance", "direction_flip", "sensor_conflict"]},
    "L2-1": {"source_region": [f"R{i}" for i in range(1, 9)],
             "stage": ["incipient", "growth", "developed", "decay"]},
    "L2-2": {"risk_region": [f"R{i}" for i in range(1, 9)],
             "risk_level": ["low", "moderate", "high", "critical"],
             "risk_driver": ["temperature", "visibility", "soot"],
             "runner_up_region": [f"R{i}" for i in range(1, 9)],
             "stability_phase": ["early", "middle", "late", "unstable"],
             "winner_switch_count_bin": ["zero", "one", "multiple"],
             "winner_support_count": [f"v{i}" for i in range(16)],
             "runner_up_support_count": [f"v{i}" for i in range(16)],
             "winner_switch_count": [f"v{i}" for i in range(16)]},
    "L2-3": {"mechanism": ["buoyant_plume", "ceiling_jet", "smoke_layer", "longitudinal_ventilation",
                             "backlayering", "extraction_dominated"],
             "flow_direction": ["vertical", "downstream", "upstream", "bidirectional", "mixed"],
             "control_regime": ["buoyancy", "ventilation", "extraction", "mixed"]},
    "L3-2": {"region": [*[f"R{i}" for i in range(1, 9)], "none"],
             "time_bin": ["none", "0_10s", "10_30s", "30_60s", "after_60s"],
             "trigger_variable": ["none", "temperature", "visibility", "soot"]},
    "L3-3": {"direction": ["A", "B", "same"],
             "magnitude_bin": ["negligible", "small", "medium", "large"],
             "earliest_affected_target": ["none", "temperature", "smoke", "visibility", "risk_region"]},
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_confidential_config(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([A-Z][A-Z0-9_]+)=(.+)$", line.strip())
        if match:
            values[match.group(1)] = match.group(2).strip()
    return values


def enum_value_schema(values: list[object]) -> dict:
    if values and all(isinstance(value, str) for value in values):
        value_type = "string"
    elif values and all(isinstance(value, int) and not isinstance(value, bool) for value in values):
        value_type = "integer"
    else:
        raise ValueError(f"enum values must have one supported scalar type: {values!r}")
    return {"type": value_type, "enum": values}


def enum_object(fields: dict[str, list[object]]) -> dict:
    return {
        "type": "object", "additionalProperties": False, "required": list(fields),
        "properties": {key: enum_value_schema(values) for key, values in fields.items()},
    }


def full_prediction_schema(task: str) -> dict:
    if task != "L3-1":
        return enum_object(TASK_FIELDS[task])
    trend = enum_object({"temperature_trend": ["up", "stable", "down"],
                         "smoke_trend": ["up", "stable", "down"],
                         "visibility_trend": ["up", "stable", "down"]})
    return {
        "type": "object", "additionalProperties": False, "required": ["horizons"],
        "properties": {"horizons": {
            "type": "object", "additionalProperties": False, "required": ["10s", "30s", "60s"],
            "properties": {key: trend for key in ("10s", "30s", "60s")},
        }},
    }


def schema_subset(schema: dict, fields: list[str]) -> dict:
    selected: dict = {"type": "object", "additionalProperties": False, "required": [], "properties": {}}
    for field in fields:
        source = schema
        target = selected
        for part in field.split("."):
            if source.get("type") != "object" or part not in source.get("properties", {}):
                raise KeyError(f"unknown prediction field: {field}")
            source = source["properties"][part]
            if part not in target["properties"]:
                target["required"].append(part)
                target["properties"][part] = (
                    {"type": "object", "additionalProperties": False,
                     "required": [], "properties": {}}
                    if source.get("type") == "object" else source
                )
            target = target["properties"][part]
    return selected


def prediction_schema(row: dict) -> dict:
    schema = full_prediction_schema(row["task"])
    fields = row.get("answer_fields")
    return schema_subset(schema, fields) if fields else schema


def response_schema(row: dict) -> dict:
    if row["question_type"] == "choice":
        labels = [option["label"] for option in row.get("options", [])]
        if not labels or len(labels) != len(set(labels)):
            raise ValueError(f"invalid choice labels for {row.get('qa_id', '<unknown>')}")
        return {
            "type": "object", "additionalProperties": False, "required": ["choices", "confidence"],
            "properties": {"choices": {"type": "array", "items": {"type": "string", "enum": labels},
                                        "minItems": 0, "maxItems": len(labels), "uniqueItems": True},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
        }
    return {
        "type": "object", "additionalProperties": False,
        "required": ["conclusion", "prediction", "evidence", "mechanism", "confidence"],
        "properties": {
            "conclusion": {"type": "string"}, "prediction": prediction_schema(row),
            "evidence": {"type": "array", "items": {"type": "string"}},
            "mechanism": {"type": "string"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
    }


def strip_paths(value: Any) -> Any:
    if isinstance(value, list):
        return [strip_paths(item) for item in value]
    if isinstance(value, dict):
        return {key: strip_paths(item) for key, item in value.items() if key not in {"path", "asset_path"}}
    return value


def asset_entries(value: Any) -> list[dict]:
    output = []
    if isinstance(value, list):
        for item in value:
            output.extend(asset_entries(item))
    elif isinstance(value, dict):
        if "path" in value or "asset_path" in value:
            output.append(value)
        for key, item in value.items():
            if key not in {"path", "asset_path"}:
                output.extend(asset_entries(item))
    return output


def resolve_asset(entry: dict, source_root: Path) -> Path:
    relative = entry.get("path", entry.get("asset_path"))
    path = source_root / str(relative)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prompt_text(row: dict, task_contract: str | None = None) -> str:
    material = strip_paths(row["material"])
    parts = []
    if task_contract:
        parts.extend(["Frozen task contract (shared across all items):", task_contract])
    parts.extend([f"Task: {row['task']}", row["question"],
                  "Observation metadata/data:",
                  json.dumps(material, ensure_ascii=False, separators=(",", ":"))])
    if row["question_type"] == "choice":
        parts.extend(["Candidate hypotheses:", json.dumps(row["options"], ensure_ascii=False, separators=(",", ":")),
                      'Select every and only true claim. Return only JSON: {"choices":["A","C"],"confidence":0..1}.'])
    else:
        parts.append("Return only the required JSON report. Evidence must contain 1-3 concise observed facts; mechanism must explain the physical link.")
    return "\n".join(parts)


def encode_images(row: dict, source_root: Path) -> list[tuple[str, str]]:
    images = []
    for entry in asset_entries(row["material"]):
        path = resolve_asset(entry, source_root)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        images.append((mime, base64.b64encode(path.read_bytes()).decode("ascii")))
    return images


def extract_text(provider: str, response: dict) -> str:
    if provider in {"openai", "grok"}:
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        chunks = []
        for item in response.get("output", []):
            for content in item.get("content", []):
                if isinstance(content.get("text"), str):
                    chunks.append(content["text"])
        if chunks:
            return "\n".join(chunks)
        choices = response.get("choices", [])
        return (choices[0].get("message", {}).get("content") or "") if choices else ""
    if provider == "anthropic":
        return "\n".join(item.get("text", "") for item in response.get("content", []) if item.get("type") == "text")
    if provider == "gemini":
        candidates = response.get("candidates", [])
        if not candidates:
            return ""
        return "\n".join(part.get("text", "") for part in candidates[0].get("content", {}).get("parts", []))
    choices = response.get("choices", [])
    return (choices[0].get("message", {}).get("content") or "") if choices else ""


def usage_record(provider: str, response: dict) -> dict:
    if provider == "gemini":
        return response.get("usageMetadata", {})
    return response.get("usage", {})


def normalized_usage(provider: str, response: dict) -> dict:
    usage = usage_record(provider, response)
    if provider == "gemini":
        return {
            "input_tokens": usage.get("promptTokenCount"),
            "output_tokens": usage.get("candidatesTokenCount"),
            "cached_tokens": usage.get("cachedContentTokenCount"),
            "reasoning_tokens": usage.get("thoughtsTokenCount"),
            "total_tokens": usage.get("totalTokenCount"),
        }
    input_details = usage.get("input_tokens_details") or usage.get("prompt_tokens_details") or {}
    output_details = usage.get("output_tokens_details") or usage.get("completion_tokens_details") or {}
    return {
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens")),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens")),
        "cached_tokens": usage.get("cache_read_input_tokens", input_details.get("cached_tokens")),
        "reasoning_tokens": output_details.get("reasoning_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def provider_cost(response: dict) -> dict:
    usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
    candidates = (
        (response.get("cost"), "response.cost"),
        (response.get("total_cost"), "response.total_cost"),
        (usage.get("cost"), "response.usage.cost"),
        (usage.get("total_cost"), "response.usage.total_cost"),
    )
    for value, source in candidates:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            currency = response.get("currency") or usage.get("currency")
            return {"amount": value, "currency": currency, "source": source}
    return {"amount": None, "currency": None, "source": "not_returned_by_provider"}


def response_model(provider: str, response: dict) -> str | None:
    if provider == "gemini":
        return response.get("modelVersion")
    return response.get("model")


def file_digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def classify_exception(exc: Exception) -> str:
    text = str(exc).lower()
    if isinstance(exc, KeyError):
        return "tooling"
    if isinstance(exc, FileNotFoundError):
        return "capability"
    if any(term in text for term in (
        "datainspectionfailed", "data_inspection_failed", "inappropriate content",
        "content safety", "content policy", "safety filter",
    )):
        return "content_safety"
    if any(term in text for term in ("billing", "insufficient credit", "insufficient balance", "http 402")):
        return "billing"
    if "timeout" in text or isinstance(exc, requests.Timeout):
        return "timeout"
    if "proxy" in text:
        return "proxy"
    if "http 429" in text or "rate limit" in text:
        return "rate_limit"
    if re.search(r"http (?:4|5)\d\d", text):
        return "http"
    if any(term in text for term in ("connection", "transport")):
        return "transport"
    return "capability"


def write_run_manifest(path: Path, metadata: dict, records: list[dict]) -> None:
    status_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    usage_totals = {key: 0 for key in ("input_tokens", "output_tokens", "cached_tokens", "reasoning_tokens", "total_tokens")}
    unknown_usage = {key: 0 for key in usage_totals}
    costs = []
    for record in records:
        status = str(record.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        classification = record.get("error_classification")
        if classification:
            error_counts[classification] = error_counts.get(classification, 0) + 1
        for key in usage_totals:
            value = record.get("usage_normalized", {}).get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                usage_totals[key] += value
            else:
                unknown_usage[key] += 1
        cost = record.get("provider_cost", {})
        if cost.get("amount") is not None:
            costs.append(cost)
    manifest = {
        **metadata,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "record_count": len(records),
        "status_counts": status_counts,
        "error_classification_counts": error_counts,
        "usage_totals": usage_totals,
        "usage_missing_record_counts": unknown_usage,
        "provider_cost_records": costs,
        "provider_cost_note": "No monetary amount is inferred when the provider response omits billing data.",
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def build_request(provider: str, model: str, system: str, prompt: str, images: list[tuple[str, str]],
                  schema: dict, mode: str, max_tokens: int, reasoning_effort: str,
                  wire_api: str = "responses") -> tuple[str, dict, dict]:
    if provider in {"openai", "grok"}:
        if provider == "openai" and wire_api == "chat_completions":
            content = [{"type": "text", "text": prompt}]
            content.extend({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                           for mime, data in images)
            payload = {
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            }
            if mode == "json_schema":
                payload["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "fwb_fg9_report", "strict": True, "schema": schema},
                }
            elif mode == "json_object":
                payload["response_format"] = {"type": "json_object"}
            return "/chat/completions", {
                "Authorization": "Bearer {key}", "Content-Type": "application/json",
            }, payload
        content = [{"type": "input_text", "text": prompt}]
        content.extend({"type": "input_image", "image_url": f"data:{mime};base64,{data}"} for mime, data in images)
        payload = {"model": model, "input": [{"role": "system", "content": [{"type": "input_text", "text": system}]},
                                               {"role": "user", "content": content}],
                   "max_output_tokens": max_tokens, "store": False, "reasoning": {"effort": reasoning_effort}}
        if mode == "json_schema":
            payload["text"] = {"format": {"type": "json_schema", "name": "fwb_fg9_report", "schema": schema, "strict": True}}
        elif mode == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        path = "/responses" if provider == "grok" else "/responses"
        return path, {"Authorization": "Bearer {key}", "Content-Type": "application/json"}, payload
    if provider == "anthropic":
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image", "source": {"type": "base64", "media_type": mime, "data": data}}
                       for mime, data in images)
        payload = {"model": model, "system": system, "messages": [{"role": "user", "content": content}],
                   "max_tokens": max_tokens, "temperature": 0}
        if mode == "json_schema":
            payload["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return "/v1/messages", {"x-api-key": "{key}", "anthropic-version": "2023-06-01", "Content-Type": "application/json"}, payload
    if provider == "gemini":
        parts = [{"text": system + "\n\n" + prompt}]
        parts.extend({"inlineData": {"mimeType": mime, "data": data}} for mime, data in images)
        generation = {"temperature": 0, "maxOutputTokens": max_tokens, "responseMimeType": "application/json"}
        if mode == "json_schema":
            generation["responseJsonSchema"] = schema
        payload = {"contents": [{"role": "user", "parts": parts}], "generationConfig": generation}
        return f"/v1beta/models/{model}:generateContent", {"x-goog-api-key": "{key}",
                                                              "Content-Type": "application/json"}, payload
    if provider == "gemini_openai_compat":
        content = [{"type": "text", "text": prompt}]
        content.extend({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
                       for mime, data in images)
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
            "temperature": 0,
            "max_tokens": max_tokens,
        }
        if mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "fwb_fg9_report", "strict": True, "schema": schema},
            }
        elif mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return "/chat/completions", {"Authorization": "Bearer {key}", "Content-Type": "application/json"}, payload
    content = [{"type": "text", "text": prompt}]
    content.extend({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}} for mime, data in images)
    payload = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
               "temperature": 0, "max_tokens": max_tokens, "enable_thinking": False}
    if mode == "json_schema":
        payload["response_format"] = {"type": "json_schema", "json_schema": {"name": "fwb_fg9_report", "strict": True, "schema": schema}}
    elif mode == "json_object":
        payload["response_format"] = {"type": "json_object"}
    return "/chat/completions", {"Authorization": "Bearer {key}", "Content-Type": "application/json"}, payload


def parse_json(text: str) -> dict | None:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            try:
                value = json.loads(text[start:end + 1])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                pass
    return None


def validation_errors(value: Any, schema: dict, path: str = "<root>") -> list[str]:
    errors: list[str] = []
    expected_type = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
    }.get(expected_type, True)
    if not type_ok:
        return [f"{path}: expected {expected_type}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: value {value!r} not in enum")
    if expected_type == "number":
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: value below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: value above maximum")
    elif expected_type == "object":
        required = set(schema.get("required", []))
        missing = sorted(required - set(value))
        if missing:
            errors.append(f"{path}: missing required keys {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                errors.append(f"{path}: additional keys {extra}")
        for key, child_schema in properties.items():
            if key in value:
                errors.extend(validation_errors(value[key], child_schema, f"{path}.{key}"))
    elif expected_type == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            errors.append(f"{path}: fewer than minItems")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            errors.append(f"{path}: more than maxItems")
        if schema.get("uniqueItems") and len({json.dumps(item, sort_keys=True) for item in value}) != len(value):
            errors.append(f"{path}: duplicate items")
        item_schema = schema.get("items", {})
        for index, item in enumerate(value):
            errors.extend(validation_errors(item, item_schema, f"{path}[{index}]"))
    return errors


def post_with_retry(session: requests.Session, url: str, headers: dict, payload: dict,
                    timeout: float, retries: int) -> tuple[dict, int, float]:
    started = time.perf_counter()
    last_error = None
    for attempt in range(retries + 1):
        try:
            response = session.post(url, headers=headers, json=payload, timeout=timeout)
            if response.status_code < 400:
                return response.json(), attempt, time.perf_counter() - started
            error = RuntimeError(f"HTTP {response.status_code}: {response.text[:800]}")
            if response.status_code not in {408, 409, 429, 500, 502, 503, 504}:
                raise error
            last_error = error
        except (requests.Timeout, requests.ConnectionError, ValueError) as exc:
            last_error = exc
        if attempt < retries:
            time.sleep(min(12.0, 1.5 * 2 ** attempt) + random.random() * 0.5)
    raise RuntimeError(f"transport exhausted after {retries + 1} attempts: {last_error}")


def provider_config(provider: str, values: dict[str, str]) -> tuple[str, str, str]:
    if provider == "openai":
        return values["OPENAI_BASE_URL"], values["OPENAI_API_KEY"], values["OPENAI_MODEL"]
    if provider == "grok":
        return values["GROK_BASE_URL"], values["GROK_API_KEY"], values["GROK_MODEL"]
    if provider == "anthropic":
        return values["ANTHROPIC_BASE_URL"], values["ANTHROPIC_AUTH_TOKEN"], values["ANTHROPIC_MODEL"]
    if provider == "gemini":
        return values["GOOGLE_GEMINI_BASE_URL"], values["GEMINI_API_KEY"], values["GEMINI_MODEL"]
    if provider == "gemini_openai_compat":
        return values["GEMINI_OPENAI_BASE_URL"], values["GEMINI_API_KEY"], values["GEMINI_MODEL"]
    return values["BAILIAN_BASE_URL"], values["DASHSCOPE_API_KEY"], "qwen3.5-plus-2026-02-15"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        choices=("openai", "grok", "anthropic", "gemini", "gemini_openai_compat", "bailian"),
        required=True,
    )
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--task-contract", type=Path,
        help="Optional frozen task contract prepended identically to every selected item.",
    )
    parser.add_argument(
        "--confidential-config", type=Path,
        help="Optional KEY=VALUE file. Process environment variables take precedence.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model")
    parser.add_argument("--max-requests", type=int)
    parser.add_argument("--track", choices=("S", "I"))
    parser.add_argument("--question-type", choices=("choice", "open"))
    parser.add_argument("--task")
    parser.add_argument("--subtrack")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--wire-api", choices=("responses", "chat_completions"), default="responses",
        help="OpenAI-compatible transport. chat_completions is recorded and only valid for --provider openai.",
    )
    parser.add_argument("--format-reasks", type=int, default=1)
    parser.add_argument(
        "--format-mode", choices=("auto", "json_schema", "json_object", "prompt_only"),
        default="auto", help="Use auto fallback or a recorded single response-format mode.",
    )
    args = parser.parse_args()
    if args.provider != "openai" and args.wire_api != "responses":
        parser.error("--wire-api chat_completions is only valid with --provider openai")

    values = load_confidential_config(args.confidential_config) if args.confidential_config else {}
    values.update(os.environ)
    base_url, key, default_model = provider_config(args.provider, values)
    model = args.model or default_model
    rows = load_jsonl(args.questions)
    if args.track:
        rows = [row for row in rows if row["track"] == args.track]
    if args.question_type:
        rows = [row for row in rows if row["question_type"] == args.question_type]
    if args.task:
        rows = [row for row in rows if row["task"] == args.task]
    if args.subtrack:
        rows = [row for row in rows if row["subtrack"] == args.subtrack]
    existing = load_jsonl(args.output) if args.output.exists() else []
    completed = completed_qa_ids(existing, args.provider, model)
    pending = [row for row in rows if row["qa_id"] not in completed]
    if args.max_requests is not None:
        pending = pending[:args.max_requests]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    task_contract = args.task_contract.read_text(encoding="utf-8").strip() if args.task_contract else None
    manifest_metadata = {
        "schema_version": "FWB-FG9-API-RUN-MANIFEST-v1",
        "runner_version": RUNNER_VERSION,
        "prompt_version": PROMPT_VERSION,
        "response_schema_version": SCHEMA_VERSION,
        "provider": args.provider,
        "requested_model": model,
        "questions": str(args.questions.resolve()),
        "questions_sha256": file_digest(args.questions),
        "source_root": str(args.source_root.resolve()),
        "task_contract": str(args.task_contract.resolve()) if args.task_contract else None,
        "task_contract_sha256": file_digest(args.task_contract) if args.task_contract else None,
        "filters": {key: value for key, value in {
            "track": args.track, "question_type": args.question_type,
            "task": args.task, "subtrack": args.subtrack,
        }.items() if value is not None},
        "selected_question_count": len(rows),
        "pending_question_count_at_start": len(pending),
        "selected_qa_ids_sha256": hashlib.sha256(
            "\n".join(row["qa_id"] for row in rows).encode()
        ).hexdigest(),
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_tokens,
        "transport_retry_limit": args.retries,
        "format_reask_limit": args.format_reasks,
        "requested_format_mode": args.format_mode,
        "wire_api": args.wire_api,
    }
    all_records = list(existing)
    write_run_manifest(manifest_path, manifest_metadata, all_records)
    system = ("You are evaluating a fire-physics benchmark. Use only the supplied observation. "
              "Return one JSON object matching the requested schema, without markdown or commentary.")
    session = requests.Session()
    with args.output.open("a", encoding="utf-8") as stream:
        for position, row in enumerate(pending, 1):
            record = {"qa_id": row["qa_id"], "provider": args.provider,
                      "requested_model": model, "model": model,
                      "runner_version": RUNNER_VERSION, "prompt_version": PROMPT_VERSION,
                      "response_schema_version": SCHEMA_VERSION,
                      "question_type": row["question_type"], "task": row["task"], "status": "error",
                      "started_at_utc": datetime.now(timezone.utc).isoformat(),
                      "attempts": [], "format_reasks": 0}
            final_raw_response = None
            try:
                images = encode_images(row, args.source_root)
                prompt = prompt_text(row, task_contract)
                schema = response_schema(row)
                prompt += "\nRequired JSON Schema:\n" + json.dumps(schema, separators=(",", ":"))
                record["request_fingerprint"] = {
                    "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                    "schema_sha256": hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest(),
                    "image_count": len(images),
                }
                parsed = None
                schema_valid = False
                raw_text = ""
                received_response = False
                modes = (("json_schema", "json_object", "prompt_only")
                         if args.format_mode == "auto" else (args.format_mode,))
                for mode in modes:
                    path, headers, payload = build_request(args.provider, model, system, prompt, images, schema, mode,
                                                           args.max_tokens, args.reasoning_effort, args.wire_api)
                    url = base_url.rstrip("/") + path
                    headers = {name: value.replace("{key}", key) for name, value in headers.items()}
                    try:
                        response, transport_retries, latency = post_with_retry(
                            session, url, headers, payload, args.timeout, args.retries)
                    except RuntimeError as exc:
                        record["attempts"].append({
                            "mode": mode, "status": "request_error",
                            "error": str(exc)[:1000],
                            "error_classification": classify_exception(exc),
                        })
                        if "HTTP 400" in str(exc) or "HTTP 404" in str(exc) or "HTTP 422" in str(exc):
                            continue
                        raise
                    raw_text = extract_text(args.provider, response)
                    final_raw_response = response
                    received_response = True
                    parsed = parse_json(raw_text)
                    schema_issues = validation_errors(parsed, schema)
                    schema_valid = not schema_issues
                    record["attempts"].append({"mode": mode, "status": "response", "latency_s": latency,
                                               "transport_retries": transport_retries,
                                               "usage": usage_record(args.provider, response),
                                               "usage_normalized": normalized_usage(args.provider, response),
                                               "provider_cost": provider_cost(response),
                                               "response_model": response_model(args.provider, response),
                                               "raw_response": response,
                                               "parsed_json": parsed is not None, "schema_valid": schema_valid,
                                               "schema_errors": schema_issues[:12]})
                    record["format_mode"] = mode
                    record["usage"] = usage_record(args.provider, response)
                    record["usage_normalized"] = normalized_usage(args.provider, response)
                    record["provider_cost"] = provider_cost(response)
                    record["response_model"] = response_model(args.provider, response)
                    if schema_valid:
                        break
                if not received_response:
                    raise RuntimeError("all response-format modes were rejected before model inference")
                for _ in range(args.format_reasks):
                    if schema_valid:
                        break
                    record["format_reasks"] += 1
                    reask = prompt + "\nYour previous output was not valid JSON. Return only one valid JSON object now."
                    path, headers, payload = build_request(args.provider, model, system, reask, images, schema,
                                                           "json_object", args.max_tokens, args.reasoning_effort,
                                                           args.wire_api)
                    headers = {name: value.replace("{key}", key) for name, value in headers.items()}
                    response, transport_retries, latency = post_with_retry(
                        session, base_url.rstrip("/") + path, headers, payload, args.timeout, args.retries)
                    final_raw_response = response
                    raw_text = extract_text(args.provider, response)
                    parsed = parse_json(raw_text)
                    schema_issues = validation_errors(parsed, schema)
                    schema_valid = not schema_issues
                    record["attempts"].append({"mode": "format_reask", "status": "response", "latency_s": latency,
                                               "transport_retries": transport_retries,
                                               "usage": usage_record(args.provider, response),
                                               "usage_normalized": normalized_usage(args.provider, response),
                                               "provider_cost": provider_cost(response),
                                               "response_model": response_model(args.provider, response),
                                               "raw_response": response,
                                               "parsed_json": parsed is not None, "schema_valid": schema_valid,
                                               "schema_errors": schema_issues[:12]})
                    record["format_mode"] = "format_reask"
                    record["usage"] = usage_record(args.provider, response)
                    record["usage_normalized"] = normalized_usage(args.provider, response)
                    record["provider_cost"] = provider_cost(response)
                    record["response_model"] = response_model(args.provider, response)
                record["response"] = parsed if parsed is not None else raw_text
                record["raw_text"] = raw_text
                record["raw_response"] = final_raw_response
                record["status"] = "ok" if schema_valid else ("schema_error" if parsed is not None else "format_error")
                if not schema_valid:
                    record["error_classification"] = "format" if parsed is not None else "parse"
                reported_model = record.get("response_model")
                if (args.provider == "openai" and reported_model is not None
                        and not model_identity_matches(model, reported_model)):
                    record["status"] = "model_mismatch"
                    record["error_classification"] = "model_identity"
            except Exception as exc:  # Each item is checkpointed; one provider failure does not discard the gate.
                record["error_type"] = type(exc).__name__
                record["error"] = str(exc)[:1200]
                attempt_classes = {
                    attempt["error_classification"]
                    for attempt in record["attempts"]
                    if attempt.get("error_classification")
                }
                record["error_classification"] = (
                    next(iter(attempt_classes)) if len(attempt_classes) == 1
                    else classify_exception(exc)
                )
            record["attempt_count"] = len(record["attempts"])
            record["transport_retries"] = sum(
                attempt.get("transport_retries", 0) for attempt in record["attempts"]
            )
            record["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
            stream.flush()
            all_records.append(record)
            write_run_manifest(manifest_path, manifest_metadata, all_records)
            print(json.dumps({"finished": position, "selected": len(pending), "qa_id": row["qa_id"],
                              "status": record["status"], "provider": args.provider, "model": model}))


if __name__ == "__main__":
    main()
