#!/usr/bin/env python3
"""Compute the six FireWorldBench leaderboard metrics without an LLM judge."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def gold_linked_items(gold_rows, prediction_rows, support_scorer):
    predictions = {}
    for row in prediction_rows:
        if row.get("qa_id") and (row["qa_id"] not in predictions or row.get("status") == "ok"):
            predictions[row["qa_id"]] = row
    output = []
    for gold in gold_rows:
        record = predictions.get(gold["qa_id"], {})
        response = record.get("response") if record.get("status") == "ok" and isinstance(record.get("response"), dict) else {}
        predicted = response.get("prediction") if isinstance(response.get("prediction"), dict) else {}
        predicted_flat = dict(support_scorer.flatten(predicted))
        expected_flat = dict(support_scorer.flatten(gold.get("prediction", {})))
        explanation = support_scorer.text([response.get("evidence", []), response.get("mechanism", ""), response.get("conclusion", "")])
        anchors = support_scorer.text([gold.get("evidence_claims", []), gold.get("mechanism_claims", [])])
        scores = []
        for path, expected in expected_flat.items():
            correct = path in predicted_flat and support_scorer.equivalent(predicted_flat[path], expected)
            scores.append(float(correct and support_scorer.field_supported(path, expected, explanation, anchors)))
        output.append({"qa_id": gold["qa_id"], "task": gold["task"], "track": gold["track"],
                       "question_type": gold["question_type"],
                       "gold_linked_support": sum(scores) / len(scores) if scores else 0.0})
    return output


def key(row): return (row.get("task"), row.get("track"), row.get("question_type"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("fwb_base_scorer", package / "score_fg9_predictions.py")
    scorer = importlib.util.module_from_spec(spec); spec.loader.exec_module(scorer)
    support_spec = importlib.util.spec_from_file_location("fwb_support_scorer", package / "score_fg9_gold_linked_support.py")
    support_scorer = importlib.util.module_from_spec(support_spec); support_spec.loader.exec_module(support_scorer)
    deep_spec = importlib.util.spec_from_file_location("fwb_deep_scorer", package / "score_fg9_deep_metrics_by_cell.py")
    deep_scorer = importlib.util.module_from_spec(deep_spec); deep_spec.loader.exec_module(deep_scorer)
    gold_rows = load_jsonl(args.gold)
    prediction_rows = [row for path in args.predictions for row in load_jsonl(path)]
    prediction_map = {}
    for row in prediction_rows:
        if row.get("qa_id") and (row["qa_id"] not in prediction_map or row.get("status") == "ok"):
            prediction_map[row["qa_id"]] = row
    # Reuse the frozen public scorer by invoking its normalization/metric primitives.
    scored = []
    for gold in gold_rows:
        record = prediction_map.get(gold["qa_id"]); ok = record is not None and record.get("status") == "ok"
        value = scorer.response_value(record) if ok else None
        confidence = None
        if gold["question_type"] == "choice":
            pred = scorer.normalize_choices(value); expected = sorted(gold.get("correct_options", [gold.get("correct_option")]))
            union = set(pred or []) | set(expected); ratio = len(set(pred or []) & set(expected)) / len(union) if union else 0
            pairs = {"choice_exact_set": (",".join(expected), ",".join(pred) if pred is not None else None)}
            parsed, _ = scorer.parse_object(value)
            if parsed and isinstance(parsed.get("confidence"), (int, float)): confidence = float(parsed["confidence"])
            ef1 = mf1 = 0.0
        else:
            pred, report, _ = scorer.normalize_open(value); gf, pf = scorer.flatten(gold["prediction"]), scorer.flatten(pred or {})
            pairs = {field: (expected, pf.get(field)) for field, expected in gf.items()}; ratio = sum(a == b for a,b in pairs.values()) / len(pairs)
            if report:
                confidence = float(report["confidence"]) if isinstance(report.get("confidence"), (int,float)) else None
                free=deep_scorer.text([report.get("conclusion",""),report.get("evidence",[]),report.get("mechanism","")]); ft=deep_scorer.toks(free)
                anchors=[]
                for claim in gold.get("evidence_claims") or []:
                    observation=(claim.get("observation") or "").lower()
                    variable=next((v for v in deep_scorer.VARS if v in observation),None)
                    anchors.append((str(claim.get("region","")).lower(),variable))
                recall=sum(((region in free)+(variable in free if variable else 0))/(2 if variable else 1) for region,variable in anchors)/max(1,len(anchors))
                claims=set(value.lower() for value in re.findall(r"\bR\d+\b",free,re.I))|(ft&deep_scorer.VARS)
                anchored={value for anchor in anchors for value in anchor if value}
                precision=len(claims&anchored)/len(claims) if claims else 0
                ef1=2*precision*recall/(precision+recall) if precision+recall else 0
                gold_tokens=deep_scorer.toks(" ".join(c.get("statement","") for c in gold.get("mechanism_claims") or []))-{"the","and","with","that","this","from","into","first"}
                mechanism_tokens=deep_scorer.toks(deep_scorer.text(report.get("mechanism","")))
                mf1=len(gold_tokens&mechanism_tokens)/len(gold_tokens) if gold_tokens else 0
            else: ef1 = mf1 = 0.0
        scored.append({"task":gold["task"],"track":gold["track"],"question_type":gold["question_type"],"pairs":pairs,"ratio":ratio,"confidence":confidence,"correct":float(ratio==1),"evidence_f1":ef1,"mechanism_alignment":mf1})
    gls = {row["qa_id"]:row["gold_linked_support"] for row in gold_linked_items(gold_rows,prediction_rows,support_scorer)}
    for row,gold in zip(scored,gold_rows): row["gold_linked_support"] = gls[gold["qa_id"]]
    groups=defaultdict(list)
    for row in scored: groups[key(row)].append(row)
    def summarize(rows):
        fp=defaultdict(list)
        for row in rows:
            for field,pair in row["pairs"].items(): fp[field].append(pair)
        calibration=[(r["confidence"],r["correct"]) for r in rows if r["confidence"] is not None and 0<=r["confidence"]<=1]
        open_rows=[r for r in rows if r["question_type"]=="open"]
        return {"n":len(rows),"completion_accuracy":sum(r["ratio"] for r in rows)/len(rows),
                "macro_f1":sum(scorer.macro_f1(pairs) for pairs in fp.values())/len(fp) if fp else 0,
                "evidence_f1":sum(r["evidence_f1"] for r in open_rows)/len(open_rows) if open_rows else None,
                "mechanism_alignment":sum(r["mechanism_alignment"] for r in open_rows)/len(open_rows) if open_rows else None,
                "brier_score":sum((c-y)**2 for c,y in calibration)/len(calibration) if calibration else None,
                "gold_linked_support":sum(r["gold_linked_support"] for r in open_rows)/len(open_rows) if open_rows else None}
    result={"schema_version":"FWB-FG9-SIX-METRICS-v2-ACC","metrics":["completion_accuracy","macro_f1","evidence_f1","mechanism_alignment","brier_score","gold_linked_support"],
            "overall":summarize(scored),"by_task_track_type":[{"task":k[0],"track":k[1],"question_type":k[2],**summarize(v)} for k,v in sorted(groups.items())],
            "notes":{"direction":"Higher is better except Brier Score, where lower is better.","scope":"Completion Accuracy, Macro-F1 and Brier apply to choice and open. Evidence-F1, Mechanism Alignment and Gold-linked Support apply to open responses only.","completion_accuracy":"For choice, mean Jaccard similarity between predicted and Gold option sets. For open, mean fraction of required fields predicted correctly.","judge":"All six metrics are deterministic; no model judge is used."}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result["overall"],ensure_ascii=False,indent=2))

if __name__ == "__main__": main()
