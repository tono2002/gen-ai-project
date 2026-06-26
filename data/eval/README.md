# Agent evaluation

We evaluate the **agent layer** — the tool-using Claude loop behind
`POST /api/projects/{project_id}/ask` (`run_agent()` in `src/app.py`) — rather
than the core summarisation step, since there's no fixed reference answer for
free-form Q&A over a project's meeting history.

## Why this design

There's no labeled dataset for "what's the right answer to an open-ended
question about these meetings." Instead, `evaluate_agent.py` uses an
**LLM-as-judge** approach (the style RAGAS popularized for RAG/agent systems):
Claude itself scores each answer against the agent's own tool-call trace,
which is the ground truth the agent actually had access to when it answered.

## Metrics

| Metric | What it measures |
|---|---|
| Faithfulness | Is every claim in the answer backed by something in the tool trace? |
| Answer relevancy | Does the answer actually address the question asked? |
| Latency | Wall-clock seconds per question |
| Step-cap compliance | Did the agent stay within `MAX_AGENT_STEPS` (anti agent-loop)? |
| Guardrail test | Does the agent decline an ungrounded question instead of hallucinating? |

## Files

| File | What |
|---|---|
| `evaluate_agent.py` | Runs a set of grounded questions + one adversarial question against a real project, scores faithfulness/relevancy via LLM-judge, measures latency and step count |
| `agent_eval_results.json` | The numbers from the last run |

## Reproduce

```bash
# Run against a project that already has a few saved meetings
.venv/bin/python data/eval/evaluate_agent.py <project_id>
```
