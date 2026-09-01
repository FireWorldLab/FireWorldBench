"""Score FG9 choice/open predictions with format, ability, evidence and calibration separated."""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUIRED_OPEN_KEYS = {"conclusion", "prediction", "evidence", "mechanism", "confidence"}
TOKEN_PATTERN = re.compile(r"[a-z0-9_]+(?:\.[0-9]+)?", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"(?<![a-z0-9_])[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[-+]?\d+)?", re.IGNORECASE)
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "in", "is", "it",
    "of", "on", "or", "that", "the", "this", "to", "under", "while", "with",
}
ALIASES = {
    "thermal": "temperature", "heat": "temperature", "hot": "temperature",
    "smoke": "soot", "smoky": "soot", "obscuration": "visibility",
    "airflow": "flow", "velocity": "flow", "movement": "flow",
    "extract": "extraction", "extracted": "extraction", "exhaust": "extraction",
    "buoyant": "buoyancy", "plume": "buoyancy",
}

# Provider wording varies substantially even for the same physical explanation.
# These concepts form a small, frozen ontology for deterministic auxiliary
# evidence/mechanism scoring; exact task labels remain scored separately above.
CONCEPT_PATTERNS = {
    "fire_source": (r"\bfire\b", r"\bflame\w*\b", r"\bcombust\w*\b", r"\bignition\b", r"\bhrrpua\b"),
    "heat": (r"\btemperature\b", r"\bthermal\b", r"\bheat\w*\b", r"\bhot\w*\b"),
    "smoke": (r"\bsmoke\b", r"\bsoot\b", r"\bobscur\w*\b"),
    "visibility": (r"\bvisibility\b", r"\bvisible\b"),
    "flow": (r"\bflow\b", r"\bairflow\b", r"\bvelocity\b", r"\btransport\b", r"\badvection\b"),
    "ventilation": (r"\bventilat\w*\b", r"\blongitudinal\s+flow\b"),
    "extraction": (r"\bextract\w*\b", r"\bexhaust\w*\b"),
    "buoyancy": (r"\bbuoyan\w*\b", r"\bconvect\w*\b", r"\bplume\b"),
    "ceiling_jet": (r"\bceiling\s+jet\b",),
    "backlayering": (r"\bbacklayer\w*\b", r"\bupstream\s+smoke\b"),
    "sensor": (r"\bsensor\w*\b", r"\bmeasurement\s+(?:fault|conflict)\b"),
    "forecast": (r"\bforecast\w*\b", r"\bpredict\w*\b", r"\bextrapolat\w*\b"),
    "threshold": (r"\bthreshold\b", r"\bcross(?:es|ed|ing)?\b"),
    "risk": (r"\brisk\b", r"\bhazard\w*\b"),
    "growth": (r"\bgrowth\b", r"\bdeveloped\b", r"\bexpan\w*\b", r"\bincreas\w*\b", r"\brise\b"),
    "decay": (r"\bdecay\b", r"\bdecreas\w*\b", r"\bdeclin\w*\b", r"\bdissipat\w*\b"),
    "continuity": (r"\bcontinuit\w*\b", r"\bconsistent\b", r"\bcoupl\w*\b"),
    "anomaly": (r"\banomal\w*\b", r"\binconsistent\b", r"\bviolation\b", r"\bconflict\b"),
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def balanced_object(text: str) -> str | None:
    start = text.find("{")
    while start >= 0:
        depth = 0
        quoted = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        start = text.find("{", start + 1)
    return None


def response_value(row: dict) -> Any:
    for key in ("response", "parsed", "output", "answer", "content", "text", "raw_response"):
        if key in row:
            return row[key]
    return row


def parse_object(value: Any) -> tuple[dict | None, bool]:
    if isinstance(value, dict):
        return value, True
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        return value[0], False
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    direct = text
    if text.startswith("```"):
        direct = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
    try:
        parsed = json.loads(direct)
        return (parsed, text == direct) if isinstance(parsed, dict) else (None, False)
    except json.JSONDecodeError:
        candidate = balanced_object(text)
        if candidate:
            try:
                parsed = json.loads(candidate)
                return (parsed, False) if isinstance(parsed, dict) else (None, False)
            except json.JSONDecodeError:
                pass
    return None, False


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if not isinstance(value, dict):
        return {prefix: value}
    output = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        output.update(flatten(item, path))
    return output


def normalize_choices(value: Any) -> list[str] | None:
    if isinstance(value, dict):
        if "choices" in value:
            return normalize_choices(value["choices"])
        for key in ("choice", "selected_option", "answer", "label", "prediction"):
            if key in value:
                result = normalize_choices(value[key])
                if result is not None:
                    return result
        return None
    if isinstance(value, list):
        labels = []
        for item in value:
            if not isinstance(item, str) or not re.fullmatch(r"[A-Z]", item.strip(), re.IGNORECASE):
                return None
            labels.append(item.strip().upper())
        return sorted(labels) if len(labels) == len(set(labels)) else None
    if not isinstance(value, str):
        return None
    match = re.fullmatch(r"\s*([A-Z])\s*", value, re.IGNORECASE)
    return [match.group(1).upper()] if match else None


def normalize_choice(value: Any) -> str | None:
    """Legacy single-choice adapter retained for older analysis scripts."""
    choices = normalize_choices(value)
    return choices[0] if choices and len(choices) == 1 else None


def normalize_open(value: Any) -> tuple[dict | None, dict | None, bool]:
    parsed, strict_json = parse_object(value)
    if parsed is None:
        return None, None, False
    report = parsed
    if "report" in parsed and isinstance(parsed["report"], dict):
        report = parsed["report"]
        strict_json = False
    prediction = report.get("prediction")
    if not isinstance(prediction, dict):
        for key in ("answer", "result", "fields"):
            if isinstance(report.get(key), dict):
                prediction = report[key]
                break
    format_valid = (
        strict_json and REQUIRED_OPEN_KEYS.issubset(report) and isinstance(report.get("prediction"), dict)
        and isinstance(report.get("evidence"), list) and isinstance(report.get("mechanism"), (str, list, dict))
        and isinstance(report.get("confidence"), (int, float)) and 0 <= float(report["confidence"]) <= 1
    )
    return prediction if isinstance(prediction, dict) else None, report, format_valid


def tokens(value: Any) -> Counter[str]:
    if isinstance(value, dict):
        value = " ".join(str(item) for item in value.values())
    elif isinstance(value, list):
        value = " ".join(str(item) for item in value)
    text = str(value).lower().replace("kg/m3", "kg m3").replace("m/s", "mps")
    text = re.sub(r"_(?:p10|p50|p90|p95|mean)\b", " ", text)
    text = text.replace("_", " ").replace("-", " ")
    text = NUMBER_PATTERN.sub(lambda match: f" num{float(match.group()):.2g} ", text)
    normalized = []
    for token in TOKEN_PATTERN.findall(text):
        if token in STOP_WORDS:
            continue
        normalized.append(ALIASES.get(token, token))
    return Counter(normalized)


def token_f1(a: Any, b: Any) -> float:
    left, right = tokens(a), tokens(b)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    overlap = sum((left & right).values())
    precision = overlap / sum(left.values())
    recall = overlap / sum(right.values())
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def claim_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        if "observation" in value:
            prefix = str(value.get("region", ""))
            return [f"{prefix} {value['observation']}".strip()]
        for key in ("statement", "observation", "claim", "text"):
            if key in value:
                return [str(value[key])]
        return [json.dumps(value, sort_keys=True)]
    if isinstance(value, list):
        output = []
        for item in value:
            output.extend(claim_list(item))
        return output
    return [str(value)]


def claim_metrics(predicted: Any, gold: Any, threshold: float = 0.30) -> tuple[float, float, float, float]:
    pred_claims, gold_claims = claim_list(predicted), claim_list(gold)
    if not pred_claims and not gold_claims:
        return 1.0, 1.0, 1.0, 0.0
    if not pred_claims:
        return 0.0, 0.0, 0.0, 0.0
    if not gold_claims:
        return 0.0, 0.0, 0.0, 1.0
    scores = [[token_f1(pred, target) for target in gold_claims] for pred in pred_claims]
    matched_pred = sum(max(row) >= threshold for row in scores)
    matched_gold = sum(max(scores[p][g] for p in range(len(pred_claims))) >= threshold
                       for g in range(len(gold_claims)))
    precision = matched_pred / len(pred_claims)
    recall = matched_gold / len(gold_claims)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    hallucination = 1.0 - precision
    return precision, recall, f1, hallucination


def physical_concepts(value: Any) -> set[str]:
    text = " ".join(claim_list(value)).lower().replace("_", " ")
    return {
        concept
        for concept, patterns in CONCEPT_PATTERNS.items()
        if any(re.search(pattern, text) for pattern in patterns)
    }


def concept_metrics(predicted: Any, gold: Any) -> tuple[float, float, float, float]:
    predicted_concepts = physical_concepts(predicted)
    gold_concepts = physical_concepts(gold)
    if not predicted_concepts and not gold_concepts:
        return 1.0, 1.0, 1.0, 0.0
    if not predicted_concepts:
        return 0.0, 0.0, 0.0, 0.0
    if not gold_concepts:
        return 0.0, 0.0, 0.0, 1.0
    overlap = predicted_concepts & gold_concepts
    precision = len(overlap) / len(predicted_concepts)
    recall = len(overlap) / len(gold_concepts)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1, 1.0 - precision


def numeric_values(value: Any) -> list[float]:
    return [float(match.group()) for claim in claim_list(value) for match in NUMBER_PATTERN.finditer(claim)]


def numeric_grounding_metrics(predicted: Any, gold: Any) -> tuple[float, float, float, float]:
    pred_values, gold_values = numeric_values(predicted), numeric_values(gold)
    if not pred_values and not gold_values:
        return 1.0, 1.0, 1.0, 0.0
    if not pred_values:
        return 0.0, 0.0, 0.0, 0.0
    if not gold_values:
        return 0.0, 0.0, 0.0, 0.0
    available = set(range(len(gold_values)))
    matched = 0
    for predicted_value in pred_values:
        candidates = []
        for index in available:
            expected = gold_values[index]
            tolerance = (max(0.02, 0.05 * abs(expected)) if abs(expected) >= 0.1 else
                         max(1e-7, 0.10 * abs(expected)))
            error = abs(predicted_value - expected)
            if error <= tolerance:
                candidates.append((error, index))
        if candidates:
            _, index = min(candidates)
            available.remove(index)
            matched += 1
    precision = matched / len(pred_values)
    recall = matched / len(gold_values)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Coverage measures whether required numeric slots were attempted, not whether
    # extra numbers were emitted. Correctness remains isolated in P/R/F1.
    coverage = min(len(pred_values), len(gold_values)) / len(gold_values)
    return precision, recall, f1, coverage


def macro_f1(pairs: list[tuple[Any, Any]]) -> float:
    labels = sorted({str(gold) for gold, _ in pairs} | {str(pred) for _, pred in pairs})
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(str(gold) == label and str(pred) == label for gold, pred in pairs)
        fp = sum(str(gold) != label and str(pred) == label for gold, pred in pairs)
        fn = sum(str(gold) == label and str(pred) != label for gold, pred in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return sum(scores) / len(scores)


# --- Slot-MacroF1 primitives (per-slot exact-match F1, macro-averaged over slots) ---
def dice(pred: Any, gold: Any) -> float:
    """Sørensen–Dice on option sets (choice component, unchanged from prior definition)."""
    ps, gs = set(pred or []), set(gold)
    if not ps and not gs:
        return 1.0
    if not ps or not gs:
        return 0.0
    return 2 * len(ps & gs) / (len(ps) + len(gs))


def slot_norm(v: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(v).lower()).strip()


def slot_f1(pairs: list[tuple[Any, Any]]) -> float:
    """Per-slot exact-match F1 = 2TP/(2TP+FP+FN). pairs = list[(gold, pred_or_None)]."""
    tp = fp = fn = 0
    for e, p in pairs:
        if p is None:
            fn += 1
        elif slot_norm(e) == slot_norm(p):
            tp += 1
        else:
            fp += 1
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom else 0.0


def slot_macro_f1(field_pairs: dict[str, list]) -> float:
    """Macro-average per-slot exact-match F1 over fields. field_pairs = {field: [(gold,pred)]}."""
    if not field_pairs:
        return 0.0
    vals = [slot_f1(ps) for ps in field_pairs.values()]
    return sum(vals) / len(vals) if vals else 0.0


def combine(choice_macro: float, open_slot_macro: float, has_c: bool, has_o: bool) -> float:
    if has_c and has_o:
        return (choice_macro + open_slot_macro) / 2
    if has_c:
        return choice_macro
    if has_o:
        return open_slot_macro
    return 0.0


def ece(rows: list[tuple[float, float]], bins: int = 10) -> float | None:
    if not rows:
        return None
    total = len(rows)
    value = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [(confidence, correct) for confidence, correct in rows
                  if low <= confidence < high or index == bins - 1 and confidence == 1.0]
        if bucket:
            value += len(bucket) / total * abs(
                sum(confidence for confidence, _ in bucket) / len(bucket) -
                sum(correct for _, correct in bucket) / len(bucket)
            )
    return value


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the two-sided 95% Wilson interval for a binomial proportion."""
    if total <= 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    centre = (proportion + z * z / (2.0 * total)) / denominator
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def aggregate(scored: list[dict], key_fields: tuple[str, ...]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in scored:
        groups[tuple(row.get(key) for key in key_fields)].append(row)
    reports = []
    for key, rows in sorted(groups.items(), key=lambda item: tuple(str(value) for value in item[0])):
        field_pairs: dict[str, list[tuple[Any, Any]]] = defaultdict(list)
        for row in rows:
            for field, pair in row["field_pairs"].items():
                field_pairs[field].append(tuple(pair))
        component = {
            field: {
                "accuracy": sum(gold == pred for gold, pred in pairs) / len(pairs),
                "macro_f1": macro_f1(pairs),
                "gold_label_counts": dict(sorted(Counter(str(gold) for gold, _ in pairs).items())),
                "predicted_label_counts": dict(sorted(Counter(
                    "<missing>" if pred is None else str(pred) for _, pred in pairs
                ).items())),
                "confusion_counts": dict(sorted(Counter(
                    f"{gold} -> {'<missing>' if pred is None else pred}" for gold, pred in pairs
                ).items())),
            }
            for field, pairs in sorted(field_pairs.items())
        }
        joint_successes = sum(row["joint_correct"] for row in rows)
        joint_ci_low, joint_ci_high = wilson_interval(int(joint_successes), len(rows))
        field_correct = sum(
            expected == actual
            for row in rows for expected, actual in row["field_pairs"].values()
        )
        field_total = sum(len(row["field_pairs"]) for row in rows)
        field_micro_accuracy = field_correct / field_total if field_total else 0.0
        nonlinear_partial_credit = sum(row["completion_ratio"] ** 3.5 for row in rows) / len(rows)
        completion_micro_accuracy = sum(row["completion_ratio"] for row in rows) / len(rows)
        partial_credit_by_exponent = {
            str(exponent): sum(row["completion_ratio"] ** exponent for row in rows) / len(rows)
            for exponent in (1.5, 2.0, 2.5, 3.0, 3.25, 3.5)
        }
        component_macro_f1 = (
            sum(metrics["macro_f1"] for metrics in component.values()) / len(component)
            if component else 0.0
        )
        calibration = [(row["confidence"], row["joint_correct"]) for row in rows if row["confidence"] is not None]
        reports.append({
            **dict(zip(key_fields, key)),
            "count": len(rows),
            "response_rate": sum(row["responded"] for row in rows) / len(rows),
            "format_valid_rate": sum(row["format_valid"] for row in rows) / len(rows),
            "joint_accuracy": joint_successes / len(rows),
            "joint_accuracy_wilson_95": [joint_ci_low, joint_ci_high],
            "field_correct": field_correct,
            "field_total": field_total,
            "field_micro_accuracy": field_micro_accuracy,
            "completion_micro_accuracy": completion_micro_accuracy,
            "nonlinear_partial_credit": nonlinear_partial_credit,
            "nonlinear_partial_credit_exponent": 3.5,
            "quadratic_partial_credit": partial_credit_by_exponent["2.0"],
            "partial_credit_by_exponent": partial_credit_by_exponent,
            "component_macro_f1": component_macro_f1,
            "component_metrics": component,
            "evidence_precision": sum(row["evidence_precision"] for row in rows) / len(rows),
            "evidence_recall": sum(row["evidence_recall"] for row in rows) / len(rows),
            "evidence_f1": sum(row["evidence_f1"] for row in rows) / len(rows),
            "mechanism_f1": sum(row["mechanism_f1"] for row in rows) / len(rows),
            "evidence_lexical_f1": sum(row["evidence_lexical_f1"] for row in rows) / len(rows),
            "mechanism_lexical_f1": sum(row["mechanism_lexical_f1"] for row in rows) / len(rows),
            "hallucination_rate": sum(row["hallucination_rate"] for row in rows) / len(rows),
            "numeric_grounding_precision": sum(row["numeric_grounding_precision"] for row in rows) / len(rows),
            "numeric_grounding_recall": sum(row["numeric_grounding_recall"] for row in rows) / len(rows),
            "numeric_grounding_f1": sum(row["numeric_grounding_f1"] for row in rows) / len(rows),
            "numeric_evidence_coverage": sum(row["numeric_evidence_coverage"] for row in rows) / len(rows),
            "confidence_coverage": len(calibration) / len(rows),
            "brier": (sum((confidence - correct) ** 2 for confidence, correct in calibration) / len(calibration)
                      if calibration else None),
            "ece_10": ece(calibration),
        })
    return reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    gold_rows = load_jsonl(args.gold)
    prediction_rows = [row for path in args.predictions for row in load_jsonl(path)]
    predictions = {}
    for row in prediction_rows:
        qa_id = row.get("qa_id")
        if not qa_id:
            continue
        previous = predictions.get(qa_id)
        if previous is None or (previous.get("status") != "ok" and row.get("status") == "ok"):
            predictions[qa_id] = row
    scored = []
    for gold in gold_rows:
        record = predictions.get(gold["qa_id"])
        record_ok = record is not None and record.get("status") == "ok"
        value = response_value(record) if record_ok else None
        confidence = None
        format_valid = False
        if gold["question_type"] == "choice":
            predicted_choices = normalize_choices(value)
            parsed_choice, _ = parse_object(value)
            if parsed_choice and isinstance(parsed_choice.get("confidence"), (int, float)):
                raw_confidence = float(parsed_choice["confidence"])
                if 0 <= raw_confidence <= 1:
                    confidence = raw_confidence
            expected = gold.get("correct_options")
            if expected is None:
                expected = [gold["correct_option"]]
            expected_choices = sorted(expected)
            expected_set = set(expected_choices)
            predicted_set = set(predicted_choices or [])
            choice_union = expected_set | predicted_set
            completion_ratio = (len(expected_set & predicted_set) / len(choice_union)
                                if choice_union else 0.0)
            field_pairs = {"choice_exact_set": (
                ",".join(expected_choices),
                ",".join(predicted_choices) if predicted_choices is not None else None,
            )}
            joint_correct = float(predicted_choices == expected_choices)
            format_valid = predicted_choices is not None
            evidence_p = evidence_r = evidence_score = mechanism_score = hallucination = 0.0
            evidence_lexical = mechanism_lexical = 0.0
            numeric_p = numeric_r = numeric_score = numeric_coverage = 0.0
        else:
            predicted, report, format_valid = normalize_open(value)
            gold_flat = flatten(gold["prediction"])
            predicted_flat = flatten(predicted or {})
            field_pairs = {field: (expected, predicted_flat.get(field)) for field, expected in gold_flat.items()}
            joint_correct = float(bool(field_pairs) and all(expected == actual for expected, actual in field_pairs.values()))
            completion_ratio = (sum(expected == actual for expected, actual in field_pairs.values()) /
                                len(field_pairs) if field_pairs else 0.0)
            if report:
                raw_confidence = report.get("confidence")
                if isinstance(raw_confidence, (int, float)) and 0 <= float(raw_confidence) <= 1:
                    confidence = float(raw_confidence)
                _, _, evidence_lexical, _ = claim_metrics(
                    report.get("evidence"), gold.get("evidence_claims", []))
                _, _, mechanism_lexical, _ = claim_metrics(
                    report.get("mechanism"), gold.get("mechanism_claims", []))
                evidence_p, evidence_r, evidence_score, evidence_h = concept_metrics(
                    report.get("evidence"), gold.get("evidence_claims", []))
                _, _, mechanism_score, mechanism_h = concept_metrics(
                    report.get("mechanism"), gold.get("mechanism_claims", []))
                hallucination = (evidence_h + mechanism_h) / 2
                numeric_p, numeric_r, numeric_score, numeric_coverage = numeric_grounding_metrics(
                    report.get("evidence"), gold.get("evidence_claims", []))
            else:
                evidence_p = evidence_r = evidence_score = mechanism_score = 0.0
                evidence_lexical = mechanism_lexical = 0.0
                hallucination = 0.0
                numeric_p = numeric_r = numeric_score = numeric_coverage = 0.0
        provenance = gold.get("provenance", {})
        scored.append({
            "qa_id": gold["qa_id"], "task": gold["task"], "track": gold["track"],
            "subtrack": gold["subtrack"], "question_type": gold["question_type"],
            "state_family": provenance.get("state_family"), "archetype": provenance.get("archetype"),
            "responded": record_ok, "format_valid": format_valid,
            "joint_correct": joint_correct, "field_pairs": field_pairs,
            "completion_ratio": completion_ratio,
            "evidence_precision": evidence_p, "evidence_recall": evidence_r, "evidence_f1": evidence_score,
            "mechanism_f1": mechanism_score, "hallucination_rate": hallucination,
            "evidence_lexical_f1": evidence_lexical, "mechanism_lexical_f1": mechanism_lexical,
            "numeric_grounding_precision": numeric_p, "numeric_grounding_recall": numeric_r,
            "numeric_grounding_f1": numeric_score, "numeric_evidence_coverage": numeric_coverage,
            "confidence": confidence,
        })
    report = {
        "schema_version": "FWB-FG9-SCORE-v2-exact-set-full-open",
        "gold_count": len(gold_rows), "prediction_record_count": len(prediction_rows),
        "prediction_file_count": len(args.predictions),
        "prediction_status_counts": dict(sorted(Counter(row.get("status", "missing") for row in prediction_rows).items())),
        "matched_prediction_count": sum(row["responded"] for row in scored),
        "overall": aggregate(scored, ("question_type",)),
        "by_track_type": aggregate(scored, ("track", "question_type")),
        "by_task_track_type": aggregate(scored, ("task", "track", "question_type")),
        "by_subtrack_type": aggregate(scored, ("subtrack", "question_type")),
        "by_state_type": aggregate(scored, ("state_family", "question_type")),
        "by_archetype_type": aggregate(scored, ("archetype", "question_type")),
        "notes": {
            "core": "Joint and component label metrics include missing/malformed outputs as incorrect.",
            "primary_metrics": "All question types use nonlinear_partial_credit = mean(completion_ratio^3.5). Open completion_ratio is correct_required_fields/requested_fields. Choice completion_ratio is Jaccard similarity between predicted and gold selected-option sets. Exact-set/JEM, linear completion, and component_macro_f1 remain mandatory audit metrics.",
            "partial_credit": "The global exponent 3.5 is fixed for every item, dataset, cell, question type, and model. Choice Jaccard penalizes both omissions and extra selections without rewarding true-negative unselected options. The score is deterministic and contains no per-cell or model-specific tuning.",
            "interval": "joint_accuracy_wilson_95 is the two-sided 95% Wilson interval for choice Accuracy or open JEM.",
            "format": "format_valid_rate is reported separately and never replaces ability metrics.",
            "evidence": "Evidence/mechanism F1 use a frozen physical-concept ontology; lexical F1 is also retained as a surface-form diagnostic.",
            "numeric": "Numeric grounding separately checks cited values against gold physical observations.",
            "judge": "LLM-judge scoring is optional and excluded from the primary leaderboard.",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gold_count": report["gold_count"], "matched": report["matched_prediction_count"],
                      "overall": report["overall"]}, indent=2))


if __name__ == "__main__":
    main()
