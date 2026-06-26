# Prompts

SummarAI uses two prompts. The first is the **summarisation prompt**, sent on every file upload. The second is the **RAG prompt**, sent when a user asks a question in the Ask panel. Both are defined in [`src/app.py`](../src/app.py).

---

## 1. Summarisation prompt

### System prompt (verbatim)

```
You are SummarAI, an expert meeting analyst. From a raw meeting transcript
(possibly noisy or unpunctuated) produce three things:

1. summary - AT MOST 2 short sentences. State only the meeting's purpose and
its overall outcome. Nothing else goes here.

2. key_takeaways - the real substance of the meeting as short, scannable
bullets. Be COMPREHENSIVE: every important fact, figure, result, risk, or
point raised must appear here. Lose no information - the 2-sentence summary
deliberately omits detail, so the takeaways must carry all of it. Keep each
bullet brief and telegraphic (ideally under 14 words), no filler words. Tag
each bullet:
   - "decision" - something agreed, decided, or set as a target or policy.
   - "note" - any other important point, fact, number, result, or concern.

3. action_items - concrete tasks someone committed to. Give owner and
deadline ONLY when actually stated; never invent them. Skip vague intentions.

Write everything in English. Keep proper names, product names, and figures
exact.
```

### Why the prompt is written this way

- **"AT MOST 2 short sentences"** keeps the summary scannable. The detail is pushed into the takeaways instead, so the summary stays an at-a-glance orientation.
- **"Lose no information"** in the takeaways is the safety net. Because the summary is so short, the takeaways have to carry everything important.
- **"never invent them"** for owner and deadline stops the model from guessing who is responsible for a task. We would rather leave the owner blank than attribute a task to the wrong person.
- **Tagging each bullet decision or note** forces a distinction that ordinary summaries blur, which is what makes the output useful for follow-up.

### Enforced output structure (JSON schema)

The reply is constrained at the API level to this shape, so it always parses:

```json
{
  "summary": "string (at most 2 sentences)",
  "key_takeaways": [
    { "text": "short bullet", "type": "decision | note" }
  ],
  "action_items": [
    { "task": "string", "owner": "string or null", "deadline": "string or null" }
  ]
}
```

`owner` and `deadline` are `null` whenever the meeting did not state them. The model never fills them in by guessing.

### Model

Claude Haiku 4.5 (`claude-haiku-4-5`), the fastest and cheapest model in the Claude 4 family. It was chosen because the summarisation step is not the slow part of the pipeline (transcription is), so paying for a larger model would add cost without a meaningful speed or quality gain for this task.

---

## 2. RAG prompt

Used by `POST /api/projects/{project_id}/ask`. The retrieved meeting summaries are injected into the user message as a `<context>` block; Claude's role here is synthesis and formatting, not knowledge retrieval.

### System prompt (verbatim)

```
You are SummarAI's meeting assistant. You are given structured summaries from
past meetings in a project. Use ONLY the information in the provided context
to answer the user's question.

Rules:
- If the answer is present in the context, answer clearly and cite the meeting
  title(s) you drew from.
- If the answer is not in the context, say so explicitly — do not guess or
  hallucinate facts.
- Keep your answer concise and factual.
- Reference meetings by their title and date.
```

### User message structure

```
<context>
[Meeting 1: "Title" — YYYY-MM-DD]
Summary: ...
Key takeaways: [{"text": "...", "type": "decision|note"}, ...]
Action items: [{"task": "...", "owner": "...", "deadline": "..."}, ...]

[Meeting 2: "Title" — YYYY-MM-DD]
...
</context>

Question: <user question>
```

### Why the RAG prompt is written this way

- **"Use ONLY the information in the provided context"** is the core grounding constraint. The model's parametric knowledge is not authoritative about this team's meetings; the retrieved records are.
- **"cite the meeting title(s)"** ensures every claim is traceable. The frontend renders these citations as source chips with meeting title, date, and cosine similarity score.
- **"say so explicitly"** when the answer is not present prevents the model from drifting into general knowledge. A clean "I couldn't find that in the saved meetings" is more useful than a plausible-sounding hallucination.

### Model

Claude Haiku 4.5, same as the summarisation step. The RAG generation call is short (context + question → answer) and does not require a larger model.

### Embedding model

Gemini `embedding-001` via the `google-genai` SDK. Two task types are used:

| Call site | Task type | Rationale |
|---|---|---|
| Indexing (on save) | `retrieval_document` | Optimises the vector for being retrieved |
| Querying (on Ask) | `retrieval_query` | Optimises the vector for finding documents |

Using asymmetric task types improves cosine similarity scores between a user question and its relevant meeting summaries because Gemini `embedding-001` was trained to optimise these two roles separately.
