"""Evaluate the agentic /ask endpoint: faithfulness, relevancy, latency, guardrails.

This scores the agent layer — the bounded tool-use loop behind
POST /api/projects/{id}/ask (see run_agent() in src/app.py). There is no
human reference for free-form Q&A, so evaluation here uses an LLM-as-judge
in place of reference-based RAGAS scores:

  1. Faithfulness   — is every claim in the answer supported by the tool-call
                       trace returned alongside it? (judged by Claude, given
                       the trace as ground truth)
  2. Answer relevancy — does the answer actually address the question asked?
  3. Latency         — wall-clock seconds per question.
  4. Guardrail tests — (a) a question with no grounding in the project must
                       be declined, not hallucinated; (b) tool-call count
                       must stay within MAX_AGENT_STEPS (anti agent-loop).

Run (server must NOT be running — this calls run_agent() in-process):
  .venv/bin/python data/eval/evaluate_agent.py <project_id>
"""

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))

import anthropic  # noqa: E402
from src.app import MAX_AGENT_STEPS, run_agent  # noqa: E402

JUDGE_SYSTEM_PROMPT = """\
You are an evaluation judge for a meeting-assistant agent. You will see a
question, the agent's tool-call trace (the ground truth it had access to),
and its final answer. Score two things from 0.0 to 1.0:

faithfulness  — 1.0 if every factual claim in the answer is supported by the
                trace; 0.0 if it invents anything not in the trace.
relevancy     — 1.0 if the answer directly addresses the question; 0.0 if it
                ignores or evades it."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "faithfulness": {"type": "number"},
        "relevancy": {"type": "number"},
        "reason": {"type": "string"},
    },
    "required": ["faithfulness", "relevancy", "reason"],
    "additionalProperties": False,
}


DECLINE_SCHEMA = {
    "type": "object",
    "properties": {
        "declined_without_hallucinating": {"type": "boolean"},
        "reason": {"type": "string"},
    },
    "required": ["declined_without_hallucinating", "reason"],
    "additionalProperties": False,
}


def judge_decline(question: str, answer: str) -> dict:
    """LLM judge: did the agent correctly decline an ungrounded question instead of inventing an answer?"""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=(
            "Judge whether the agent's answer below correctly declines to answer "
            "(states the information isn't in its meeting records) rather than "
            "inventing/hallucinating a specific answer to the question."
        ),
        messages=[{
            "role": "user",
            "content": f"Question: {question}\n\nAgent's answer: {answer}",
        }],
        output_config={"format": {"type": "json_schema", "schema": DECLINE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


def judge(question: str, trace: list, answer: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=300,
        system=JUDGE_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Tool trace: {json.dumps(trace, ensure_ascii=False, default=str)}\n\n"
                f"Answer: {answer}"
            ),
        }],
        output_config={"format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
    )
    text = next(b.text for b in response.content if b.type == "text")
    return json.loads(text)


# Grounded questions — should be answerable from a project's saved meetings.
GROUNDED_QUESTIONS = [
    "What action items are still open and who owns them?",
    "What were the key decisions made?",
    "Is anything overdue?",
]

# Adversarial / ungrounded question — must be declined, not hallucinated.
GUARDRAIL_QUESTION = "What did we decide about the Q4 marketing budget for Brazil?"


def main():
    if len(sys.argv) < 2:
        print("Usage: evaluate_agent.py <project_id>")
        sys.exit(1)
    project_id = sys.argv[1]

    results = []
    print(f"Running {len(GROUNDED_QUESTIONS)} grounded questions against project {project_id}...\n")
    for q in GROUNDED_QUESTIONS:
        start = time.time()
        result = run_agent(q, project_id)
        latency = time.time() - start
        scores = judge(q, result["trace"], result["answer"])
        n_tool_calls = len(result["trace"])
        results.append({
            "question": q,
            "latency_s": round(latency, 2),
            "tool_calls": n_tool_calls,
            "within_step_cap": n_tool_calls <= MAX_AGENT_STEPS,
            **scores,
        })
        print(f"  Q: {q}")
        print(f"     latency={latency:.2f}s  tool_calls={n_tool_calls}  "
              f"faithfulness={scores['faithfulness']}  relevancy={scores['relevancy']}")

    print(f"\nGuardrail test (ungrounded question): {GUARDRAIL_QUESTION}")
    start = time.time()
    guard_result = run_agent(GUARDRAIL_QUESTION, project_id)
    guard_latency = time.time() - start
    decline_verdict = judge_decline(GUARDRAIL_QUESTION, guard_result["answer"])
    declined = decline_verdict["declined_without_hallucinating"]
    print(f"  latency={guard_latency:.2f}s  declined_without_hallucinating={declined} ({decline_verdict['reason']})")
    print(f"  answer: {guard_result['answer'][:200]}")

    avg_faith = sum(r["faithfulness"] for r in results) / len(results)
    avg_rel = sum(r["relevancy"] for r in results) / len(results)
    avg_latency = sum(r["latency_s"] for r in results) / len(results)
    all_within_cap = all(r["within_step_cap"] for r in results)

    print("\n── Summary ──")
    print(f"avg faithfulness: {avg_faith:.2f}")
    print(f"avg relevancy:    {avg_rel:.2f}")
    print(f"avg latency:      {avg_latency:.2f}s")
    print(f"all within step cap ({MAX_AGENT_STEPS}): {all_within_cap}")
    print(f"guardrail (declined w/o hallucinating): {declined}")

    out_path = HERE / "agent_eval_results.json"
    out_path.write_text(json.dumps({
        "per_question": results,
        "guardrail_test": {
            "question": GUARDRAIL_QUESTION,
            "latency_s": round(guard_latency, 2),
            "declined_without_hallucinating": declined,
            "answer": guard_result["answer"],
        },
        "summary": {
            "avg_faithfulness": round(avg_faith, 3),
            "avg_relevancy": round(avg_rel, 3),
            "avg_latency_s": round(avg_latency, 2),
            "all_within_step_cap": all_within_cap,
        },
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
