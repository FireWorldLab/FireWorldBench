import json
import math
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "artifacts/validation/fg9_dual_model_n40_final_20260812"
RUNS = {
    "main_grok": ("模拟数据主集", ["main_gate_C/gold.jsonl", "main_gate_B/gold.jsonl"]),
    "main_claude": ("模拟数据主集", ["main_gate_C/gold.jsonl", "main_gate_B/gold.jsonl"]),
    "c06_grok": ("真实事件模拟", ["c06_candidate/build_A/gold.jsonl", "c06_gate_A/gold.jsonl", "c06_pool80/build_A/gold.jsonl", "c06_pool80/build_B/gold.jsonl"]),
    "c06_claude": ("真实事件模拟", ["c06_candidate/build_A/gold.jsonl", "c06_gate_A/gold.jsonl", "c06_pool80/build_A/gold.jsonl", "c06_pool80/build_B/gold.jsonl"]),
}

ALIASES = {
    "temperature": ("temperature", "thermal", "heat", "hot"),
    "smoke": ("smoke", "soot", "obscuration"),
    "soot": ("soot", "smoke", "combustion signature"),
    "visibility": ("visibility", "visible", "obscuration"),
    "flow": ("flow", "velocity", "ventilation", "transport"),
    "up": ("increase", "increasing", "rise", "rising", "grow", "growth", "up", "higher"),
    "down": ("decrease", "decreasing", "fall", "falling", "decline", "down", "lower"),
    "stable": ("stable", "steady", "unchanged", "constant", "plateau"),
    "critical": ("critical", "extreme", "severe"),
    "high": ("high", "hazardous"),
    "incipient": ("incipient", "early", "initial", "ignition"),
    "growth": ("growth", "growing", "developing"),
    "global": ("global", "across", "all regions", "domain-wide"),
    "mixed": ("mixed", "bidirectional", "multiple directions"),
    "vertical": ("vertical", "upward", "buoyant", "rise"),
}


def norm(value):
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def text(value):
    if isinstance(value, str):
        return norm(value)
    if isinstance(value, dict):
        return " ".join(text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(text(item) for item in value)
    return norm(value)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            yield from flatten(item, path)
    else:
        yield prefix, value


def equivalent(predicted, expected):
    if isinstance(expected, (int, float)):
        try:
            return float(predicted) == float(expected)
        except (TypeError, ValueError):
            return False
    return norm(predicted) == norm(expected)


def mentions(needle, corpus):
    key = norm(needle)
    variants = ALIASES.get(key, (key,))
    return any(re.search(rf"(?<![a-z0-9]){re.escape(norm(v))}(?![a-z0-9])", corpus) for v in variants)


def field_supported(path, expected, response_text, gold_text):
    parts = path.lower().split(".")
    field = parts[-1]
    expected_text = norm(expected)
    gold_linked = mentions(expected_text, gold_text)

    if re.fullmatch(r"r\d+", expected_text) or "region" in field:
        return gold_linked and mentions(expected_text, response_text)
    if field.endswith("trend"):
        variable = field.removesuffix("_trend")
        horizon = parts[-2] if len(parts) > 1 and re.fullmatch(r"\d+s", parts[-2]) else None
        context = mentions(variable, response_text) and mentions(expected_text, response_text)
        if horizon:
            context = context and mentions(horizon, response_text)
        return context
    if any(token in field for token in ("signal", "driver", "variable", "target")):
        return gold_linked and mentions(expected_text, response_text)
    if isinstance(expected, (int, float)):
        return mentions(expected_text, response_text) or any(word in response_text for word in ("all regions", "across regions"))
    return gold_linked and mentions(expected_text, response_text)


def pearson(left, right):
    mean_left = sum(left) / len(left)
    mean_right = sum(right) / len(right)
    numerator = sum((x - mean_left) * (y - mean_right) for x, y in zip(left, right))
    denom_left = sum((x - mean_left) ** 2 for x in left)
    denom_right = sum((y - mean_right) ** 2 for y in right)
    return numerator / math.sqrt(denom_left * denom_right) if denom_left and denom_right else None


def main():
    prior_rows = json.load(open(BASE / "final_report_20260813/n40_open_deep_metrics_18cells.json", encoding="utf-8"))["rows"]
    prior = {(row["dataset"], row["model"], row["cell"]): row for row in prior_rows}
    rows = []
    item_audit = []

    for run, (dataset, gold_paths) in RUNS.items():
        gold = {}
        for gold_path in gold_paths:
            for line in open(BASE / gold_path, encoding="utf-8"):
                item = json.loads(line)
                if item.get("question_type") == "open":
                    gold[item["qa_id"]] = item
        cells = defaultdict(list)
        for line in open(BASE / "runs" / run / "merged_responses.jsonl", encoding="utf-8"):
            response_record = json.loads(line)
            qa_id = response_record.get("qa_id")
            if response_record.get("question_type") != "open" or qa_id not in gold:
                continue
            gold_item = gold[qa_id]
            response = response_record.get("response") if isinstance(response_record.get("response"), dict) else {}
            predicted = response.get("prediction") if isinstance(response.get("prediction"), dict) else {}
            predicted_flat = dict(flatten(predicted))
            expected_flat = dict(flatten(gold_item.get("prediction", {})))
            response_explanation = text([response.get("evidence", []), response.get("mechanism", ""), response.get("conclusion", "")])
            gold_anchors = text([gold_item.get("evidence_claims", []), gold_item.get("mechanism_claims", [])])
            field_scores = []
            for path, expected in expected_flat.items():
                correct = path in predicted_flat and equivalent(predicted_flat[path], expected)
                supported = correct and field_supported(path, expected, response_explanation, gold_anchors)
                field_scores.append(float(supported))
            score = sum(field_scores) / len(field_scores) if field_scores else 0.0
            cell = f'{gold_item["task"]}-{gold_item["track"]}'
            cells[cell].append(score)
            item_audit.append({"qa_id": qa_id, "dataset": dataset, "model": "Grok" if run.endswith("grok") else "Claude", "cell": cell, "score": score, "field_scores": field_scores})

        model = "Grok" if run.endswith("grok") else "Claude"
        for cell, scores in cells.items():
            old = prior[(dataset, model, cell)]
            precision = old["evidence_precision_proxy"]
            recall = old["evidence_anchor_recall"]
            evidence_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
            rows.append({"dataset": dataset, "model": model, "cell": cell, "n": len(scores), "gold_linked_support": sum(scores) / len(scores), "evidence_f1": evidence_f1, "mechanism_alignment": old["mechanism_alignment"]})

    rows.sort(key=lambda row: (row["dataset"], row["cell"], row["model"]))
    values = [row["gold_linked_support"] for row in rows]
    summary = {
        "row_count": len(rows), "min": min(values), "max": max(values), "mean": sum(values) / len(values),
        "cells_le_0_05": sum(value <= 0.05 for value in values), "cells_ge_0_95": sum(value >= 0.95 for value in values),
        "cells_eq_0": sum(value == 0 for value in values), "cells_eq_1": sum(value == 1 for value in values),
        "pearson_with_evidence_f1": pearson(values, [row["evidence_f1"] for row in rows]),
        "pearson_with_mechanism_alignment": pearson(values, [row["mechanism_alignment"] for row in rows]),
    }
    output = {"schema_version": "FWB-FG9-GOLD-LINKED-SUPPORT-v1", "definition": "mean fraction of Gold-correct prediction fields supported by task-aware evidence or mechanism anchors", "rows": rows, "summary": summary, "item_audit": item_audit}
    destination = BASE / "final_report_20260813/n40_gold_linked_support_18cells.json"
    destination.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for dataset in ("模拟数据主集", "真实事件模拟"):
        print(dataset)
        print([(row["cell"], row["model"], round(row["gold_linked_support"], 3)) for row in rows if row["dataset"] == dataset])


if __name__ == "__main__":
    main()
