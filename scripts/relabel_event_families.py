#!/usr/bin/env python3
"""Build the three-stage fire axis, event-family manifest, and grouped audits.

The input files are expected to be exported from the official HF repositories.
No source data is committed by this script.  Every output row is copied with
only ``fire_axis`` changed; physical_axis and all answer fields are preserved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FIRE_AXIS = {
    "T1 Localized Onset": "localized source, anomaly, violation, or fire-stage assessment",
    "T2 Coupled Propagation": "cross-region state, transport, coupling, or risk propagation",
    "T3 Critical Transition": "future outcome, threshold crossing, intervention, or counterfactual transition",
}
TASK_FIRE = {
    "L1-1": "T1 Localized Onset", "L1-3": "T1 Localized Onset", "L2-1": "T1 Localized Onset",
    "L1-2": "T2 Coupled Propagation", "L2-2": "T2 Coupled Propagation", "L2-3": "T2 Coupled Propagation",
    "L3-1": "T3 Critical Transition", "L3-2": "T3 Critical Transition", "L3-3": "T3 Critical Transition",
}

FAMILIES = [
    "Transportation and Infrastructure",
    "Office and Digital Facilities",
    "Healthcare and Education",
    "Commercial and Public Spaces",
    "Residential and Care Settings",
    "Industry, Energy, and Logistics",
    "Wildland and Wildland--Urban Interface (WUI)",
]
FAMILY_RULES = [
    (FAMILIES[6], ["wildland", "wildfire", "forest", "grassland", "wui", "wild_urban"]),
    (FAMILIES[0], ["rail", "metro", "transit", "transport", "vehicle", "road", "bridge", "tunnel", "airport", "port", "bus_depot"]),
    (FAMILIES[1], ["data_center", "datacenter", "server", "office", "digital", "telecom", "computer"]),
    (FAMILIES[2], ["hospital", "health", "clinic", "school", "university", "college", "education", "laboratory", "laboratory_room"]),
    (FAMILIES[3], ["mall", "market", "retail", "shop", "restaurant", "hotel", "stadium", "theater", "museum", "commercial", "public"]),
    (FAMILIES[4], ["house", "home", "residential", "apartment", "dwelling", "nursing", "care", "dormitory"]),
    (FAMILIES[5], ["industrial", "factory", "plant", "tank", "process", "pipe", "warehouse", "logistics", "loading", "energy", "battery", "charging", "refinery", "workshop"]),
]
EXPLICIT_FAMILY = {
    "aircraft_hangar": FAMILIES[0],
    "canadian_townhouse_row": FAMILIES[4],
    "chemical_packaging_warehouse": FAMILIES[5],
    "crosswind_canyon": FAMILIES[6],
    "exhibition_hall_booths": FAMILIES[3],
    "fireworks_mixing_workshop": FAMILIES[5],
    "fireworks_storage_magazines": FAMILIES[5],
    "high_bay_rack_warehouse": FAMILIES[5],
    "industrial_service_tunnel": FAMILIES[5],
    "mixed_conifer_valley": FAMILIES[6],
    "school_vocational_workshop": FAMILIES[2],
    "semiconductor_cleanroom": FAMILIES[1],
    "stepped_pine_slope": FAMILIES[6],
    "switchgear_transformer_room": FAMILIES[5],
    "theatre_backstage": FAMILIES[3],
    "timber_heritage_courtyard": FAMILIES[3],
    "two_level_underground_garage": FAMILIES[0],
    "waste_sorting_recycling_hall": FAMILIES[5],
    "windward_ridge_crest": FAMILIES[6],
    "wui_road_edge": FAMILIES[6],
}


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                yield number, json.loads(line)


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def norm(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")


def classify_fire(row):
    task = row.get("task")
    label = TASK_FIRE.get(task)
    text = " ".join(str(row.get(key, "")) for key in ("question", "task"))
    boundary = bool(re.search(r"\b(future|forecast|predict|threshold|intervention|counterfactual)\b", text, re.I))
    return label, {"rule": f"task:{task}", "boundary_review": boundary}


def classify_family(archetype):
    token = norm(archetype)
    if token in EXPLICIT_FAMILY:
        return EXPLICIT_FAMILY[token], [token], "matched"
    hits = [(family, [word for word in words if re.search(rf"(?:^|_){re.escape(norm(word))}(?:_|$)", token)]) for family, words in FAMILY_RULES]
    hits = [(family, words) for family, words in hits if words]
    if not hits:
        return None, [], "unmatched"
    max_hits = max(len(words) for _, words in hits)
    winners = [(family, words) for family, words in hits if len(words) == max_hits]
    if len(winners) > 1:
        return winners[0][0], [x[0] for x in winners], "ambiguous"
    return winners[0][0], winners[0][1], "matched"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", nargs="+", type=Path, required=True)
    ap.add_argument("--questions", nargs="+", type=Path, required=True)
    ap.add_argument("--item-scores", nargs="*", type=Path, default=[])
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    gold_by_qa = {}
    event_rows = defaultdict(lambda: {"event_id": None, "archetypes": set(), "sources": set(), "qa_count": 0})
    audit = []
    for path in args.gold:
        source = path.as_posix()
        for line, row in read_jsonl(path):
            qa_id = row.get("qa_id")
            if not qa_id:
                audit.append({"kind": "gold_missing_qa_id", "source": source, "line": line})
                continue
            fire, fire_meta = classify_fire(row)
            provenance = row.get("provenance") or {}
            event_id = provenance.get("event_id") or provenance.get("world_id")
            archetype = provenance.get("archetype") or (provenance.get("metadata") or {}).get("archetype")
            family, matched, family_status = classify_family(archetype)
            if qa_id in gold_by_qa and gold_by_qa[qa_id]["event_id"] != event_id:
                audit.append({"kind": "qa_id_conflict", "qa_id": qa_id, "source": source})
            gold_by_qa[qa_id] = {"fire_axis": fire, "fire_meta": fire_meta, "event_id": event_id,
                                 "family": family, "family_status": family_status, "archetype": archetype,
                                 "physical_axis": row.get("physical_axis"), "task": row.get("task")}
            if event_id:
                item = event_rows[event_id]
                item["event_id"] = event_id; item["qa_count"] += 1; item["sources"].add(source)
                if archetype: item["archetypes"].add(archetype)
            else:
                audit.append({"kind": "missing_event_id", "qa_id": qa_id, "source": source})
            if family_status != "matched":
                audit.append({"kind": f"family_{family_status}", "qa_id": qa_id, "event_id": event_id,
                              "archetype": archetype, "matched": matched})

    # Enrich and rewrite question files while retaining every original field.
    for path in args.questions:
        out = args.output / "relabelled" / path.name
        rows = []
        for line, row in read_jsonl(path):
            qa_id = row.get("qa_id")
            info = gold_by_qa.get(qa_id)
            if info is None:
                audit.append({"kind": "question_unmatched_gold", "qa_id": qa_id, "source": path.as_posix(), "line": line,
                              "excluded_from_formal_output": True})
                continue
            else:
                row["fire_axis"] = info["fire_axis"]
            rows.append(row)
        write_jsonl(out, rows)

    # Also provide relabelled gold files, useful for testResult consumers.
    for path in args.gold:
        out = args.output / "relabelled_gold" / path.name
        rows = []
        for _, row in read_jsonl(path):
            info = gold_by_qa.get(row.get("qa_id"))
            if info: row["fire_axis"] = info["fire_axis"]
            rows.append(row)
        write_jsonl(out, rows)

    manifest = []
    for event_id, item in sorted(event_rows.items()):
        archetypes = sorted(item["archetypes"])
        family_results = [classify_family(x) for x in archetypes]
        families = sorted({x[0] for x in family_results if x[0]})
        status = "matched" if len(families) == 1 and all(x[2] == "matched" for x in family_results) else ("ambiguous" if families else "unmatched")
        manifest.append({"event_id": event_id, "family": families[0] if len(families) == 1 else None,
                         "family_status": status, "archetypes": archetypes, "question_count": item["qa_count"],
                         "source_files": sorted(item["sources"])})
    (args.output / "event_family_manifest.json").write_text(json.dumps({"families": FAMILIES, "events": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Item scores are the official deterministic scorer's per-item values.
    metric_rows = []
    for path in args.item_scores:
        model = path.stem.removesuffix(".six")
        for _, score in read_jsonl(path):
            info = gold_by_qa.get(score.get("qa_id"), {})
            metric_rows.append({"model": model, "track": model.rsplit("_", 1)[-1], "interface": model.rsplit("_", 1)[-1],
                                "split": path.parts[-3] if len(path.parts) >= 3 else "unknown",
                                "event_id": info.get("event_id"), "event_family": next((m["family"] for m in manifest if m["event_id"] == info.get("event_id")), None),
                                "fire_axis": info.get("fire_axis"), "physical_axis": info.get("physical_axis"),
                                "question_type": score.get("question_type"), "task": score.get("task"),
                                "qa_id": score.get("qa_id"), "ratio": score.get("ratio"), "item_f1": score.get("item_f1"),
                                "evidence_f1": score.get("evidence_f1"), "mechanism_alignment": score.get("mechanism_alignment"),
                                "gold_linked_support": score.get("gold_linked_support"), "confidence": score.get("confidence")})
    if metric_rows:
        grouped = defaultdict(list)
        for row in metric_rows:
            grouped[(row["model"], row["track"], row["interface"], row["split"], row["event_family"], row["question_type"])].append(row)
        summaries = []
        for key, rows in sorted(grouped.items(), key=lambda x: str(x[0])):
            def avg(field):
                vals = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
                return sum(vals) / len(vals) if vals else None
            summaries.append(dict(zip(("model", "track", "interface", "split", "event_family", "question_type"), key),
                                    n=len(rows), completion_accuracy=avg("ratio"), f1=avg("item_f1"), brier_score=avg("brier_component"),
                                    evidence_f1=avg("evidence_f1"), mechanism_alignment=avg("mechanism_alignment"),
                                    gold_linked_support=avg("gold_linked_support")))
        # Brier is (confidence - completion ratio)^2 in the frozen scorer.
        for row in metric_rows:
            if isinstance(row.get("confidence"), (int, float)) and isinstance(row.get("ratio"), (int, float)):
                row["brier_component"] = (row["confidence"] - row["ratio"]) ** 2
        for summary in summaries:
            rows = grouped[tuple(summary[k] for k in ("model", "track", "interface", "split", "event_family", "question_type"))]
            vals = [r["brier_component"] for r in rows if "brier_component" in r]
            summary["brier_score"] = sum(vals) / len(vals) if vals else None
        with (args.output / "metrics_by_event_family.csv").open("w", encoding="utf-8", newline="") as handle:
            fields = list(summaries[0]) if summaries else []
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summaries)
        write_jsonl(args.output / "metric_items.jsonl", metric_rows)
    report = {"schema_version": "FWB-FG9-THREE-STAGE-EVENT-FAMILY-v1", "fire_axis": FIRE_AXIS,
              "task_to_fire_axis": TASK_FIRE, "family_names": FAMILIES,
              "counts": {"gold_rows": len(gold_by_qa), "events": len(manifest), "audit_rows": len(audit),
                          "fire_axis": dict(Counter(x["fire_axis"] for x in gold_by_qa.values())),
                          "physical_axis": dict(Counter(x.get("physical_axis") for x in gold_by_qa.values()))},
              "audit": audit}
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
