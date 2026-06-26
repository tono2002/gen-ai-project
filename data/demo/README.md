# Demo dataset — Project Atlas

A small, hand-written set of meetings used to **demo the agent live**. It is
deliberately separate from the AMI evaluation set in [`../eval/`](../eval/),
which stays the rigorous, corpus-based evaluation. These demo meetings exist
because AMI's action items have vague deadlines ("30 minutes"), so they can't
exercise the agent's accountability features.

## The scenario

A product team ("Project Atlas") building a customer analytics dashboard, across
four meetings in June 2026. The data is designed so every agent capability has
something real to show **as of 26 June 2026**:

| Capability | What to ask | Why it works |
|---|---|---|
| Semantic search | "What did we decide about dark mode?" | Decisions spread across meetings |
| Action items by owner | "What does Priya own?" | Explicit owners throughout |
| **Overdue (accountability)** | "What's overdue?" | ~7 action items have deadlines before 26 June |
| Cross-meeting memory | "What's the launch date?" | It changes: **July 15 → July 22** across two meetings |
| Draft recap | "Draft a recap of the kickoff." | Each meeting has clean decisions + actions |

The launch-date change (kickoff sets July 15; the mid-sprint check-in pushes it
to July 22) is the clearest demonstration of the **project-memory** positioning:
a single meeting can't answer "what's the launch date *now*?" — the agent has to
reason across the history.

## How to load it

After filling in `.env` (Anthropic + Gemini keys, and your Supabase):

```bash
.venv/bin/python data/demo/seed_demo.py
```

It creates a project called **"Project Atlas (demo)"**, runs each transcript
through the real summarisation pipeline, embeds the results, and saves them.
Re-running is safe — it skips meetings already saved.

## Note on overdue-but-done items

Some "overdue" items (e.g. staging, the layout revision) are commitments whose
stated deadline has passed. The system surfaces every past-due *commitment* — it
does not track real-world completion, because the meetings never state it.
Closing that loop (marking items done) is noted as future work, not a bug.
