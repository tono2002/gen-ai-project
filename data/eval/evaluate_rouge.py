"""Evaluate SummarAI against AMI's human reference summaries (ROUGE).

For each of our 30 transcripts (mapped to its AMI ES meeting), we run the real
summarizer and compare its output to the human-written AMI abstractive summary.

Two automatic metrics, no manual labeling:
  1. ROUGE-1/2/L , our full output (summary + takeaways + actions) vs the AMI
     human reference (abstract + decisions + actions). Standard summarization metric.
  2. Action-item coverage (recall), fraction of the AMI human ACTION sentences
     that a predicted action item covers (ROUGE-L F >= ACTION_THRESHOLD).

Run:  .venv/bin/python data/eval/evaluate_rouge.py
"""

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from rouge_score import rouge_scorer

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
from src.app import analyze  # noqa: E402  (reuse the real pipeline)

ABS_DIR = HERE / "ami_raw" / "abstractive"
TRANSCRIPTS = HERE / "transcripts"
MAPPING = json.loads((HERE / "id_mapping.json").read_text())
ACTION_THRESHOLD = 0.30

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def parse_abssumm(es_code: str) -> dict:
    """Return {'abstract': [...], 'actions': [...], 'decisions': [...]} sentence lists."""
    f = ABS_DIR / f"{es_code}.abssumm.xml"
    tree = ET.parse(f)
    root = tree.getroot()
    out = {"abstract": [], "actions": [], "decisions": [], "problems": []}
    for section in out:
        node = root.find(section)
        if node is not None:
            for s in node.findall("sentence"):
                if s.text and s.text.strip():
                    out[section].append(s.text.strip())
    return out


def render_pred(result: dict) -> str:
    parts = [result.get("summary", "")]
    parts += [t.get("text", "") for t in result.get("key_takeaways", [])]
    for a in result.get("action_items", []):
        owner = a.get("owner") or ""
        parts.append(f"{owner} {a.get('task', '')}".strip())
    return " ".join(p for p in parts if p)


def action_coverage(pred_actions: list[str], gold_actions: list[str]) -> tuple[int, int]:
    if not gold_actions:
        return 0, 0
    covered = 0
    for g in gold_actions:
        best = max((scorer.score(g, p)["rougeL"].fmeasure for p in pred_actions), default=0.0)
        if best >= ACTION_THRESHOLD:
            covered += 1
    return covered, len(gold_actions)


def main() -> None:
    rows = []
    agg = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    cov_hit, cov_tot = 0, 0

    for tid in sorted(MAPPING, key=lambda x: int(x.split("_")[1])):
        es = MAPPING[tid]
        transcript = (TRANSCRIPTS / f"{tid}.txt").read_text(encoding="utf-8")
        gold = parse_abssumm(es)
        reference = " ".join(gold["abstract"] + gold["decisions"] + gold["actions"])

        result = analyze(transcript)
        hypothesis = render_pred(result)
        pred_actions = [f"{(a.get('owner') or '')} {a.get('task', '')}".strip()
                        for a in result.get("action_items", [])]

        sc = scorer.score(reference, hypothesis)
        for k in agg:
            agg[k] += sc[k].fmeasure
        hit, tot = action_coverage(pred_actions, gold["actions"])
        cov_hit += hit
        cov_tot += tot

        rows.append({"id": tid, "es": es,
                     "rouge1": round(sc["rouge1"].fmeasure, 3),
                     "rouge2": round(sc["rouge2"].fmeasure, 3),
                     "rougeL": round(sc["rougeL"].fmeasure, 3),
                     "action_cov": f"{hit}/{tot}"})
        print(f"  {tid:<9} {es:<8} R1={sc['rouge1'].fmeasure:.2f} "
              f"R2={sc['rouge2'].fmeasure:.2f} RL={sc['rougeL'].fmeasure:.2f} "
              f"actions={hit}/{tot}", flush=True)

    n = len(rows)
    summary = {"meetings": n,
               "rouge1": round(agg["rouge1"] / n, 3),
               "rouge2": round(agg["rouge2"] / n, 3),
               "rougeL": round(agg["rougeL"] / n, 3),
               "action_item_recall": round(cov_hit / cov_tot, 3) if cov_tot else None,
               "action_items_covered": f"{cov_hit}/{cov_tot}"}
    (HERE / "results.json").write_text(json.dumps({"summary": summary, "per_meeting": rows}, indent=2))

    print("\n" + "=" * 50)
    print(f"  Meetings           : {n}")
    print(f"  ROUGE-1 (avg)      : {summary['rouge1']}")
    print(f"  ROUGE-2 (avg)      : {summary['rouge2']}")
    print(f"  ROUGE-L (avg)      : {summary['rougeL']}")
    print(f"  Action-item recall : {summary['action_item_recall']} "
          f"({summary['action_items_covered']})")
    print("=" * 50)
    print("Saved results.json")


if __name__ == "__main__":
    main()
