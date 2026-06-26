"""SummarAI — English meeting summarizer with project management + RAG Q&A.

Pipeline: audio/video or transcript → Whisper (if audio) → Claude → a 2-sentence
summary, typed key-takeaway bullets (decision / note), and action items.
Summaries can be saved to projects stored in Supabase.

RAG feature: saved summaries are embedded with Gemini text-embedding-001 (1536 dims).
POST /api/projects/{project_id}/ask accepts a natural-language question and
returns a grounded answer with source citations drawn from past meetings.
"""

import json
import os
import re
import tempfile
from datetime import date, datetime
from pathlib import Path

import anthropic
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="SummarAI")

STATIC_DIR = Path(__file__).resolve().parent / "static"

AUDIO_EXTENSIONS = {".mp4", ".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac"}
TEXT_EXTENSIONS = {".txt", ".md", ".vtt", ".srt"}
MAX_TRANSCRIPT_CHARS = 300_000

WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "base.en")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ssooqczamcqxpvcpeebv.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNzb29xY3phbWNxeHB2Y3BlZWJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODEyNzU0MTEsImV4cCI6MjA5Njg1MTQxMX0.4KbYBc9qY-6Gq_jsdiTWZTdis14xEpPzW0G66qAuq4E",
)
SUPABASE_HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "return=representation",
}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {
            "type": "string",
            "description": "At most 2 short sentences: the meeting's purpose and overall outcome.",
        },
        "key_takeaways": {
            "type": "array",
            "description": "Every important point from the transcript as short, scannable bullets. Lose no information.",
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Brief, punchy bullet (ideally under 14 words). No filler.",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["decision", "note"],
                        "description": "'decision' if agreed/decided/set as a target or policy; otherwise 'note'.",
                    },
                },
                "required": ["text", "type"],
                "additionalProperties": False,
            },
        },
        "action_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "task": {"type": "string"},
                    "owner": {"type": ["string", "null"]},
                    "deadline": {"type": ["string", "null"]},
                },
                "required": ["task", "owner", "deadline"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "key_takeaways", "action_items"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are SummarAI, an expert meeting analyst. From a raw meeting transcript \
(possibly noisy or unpunctuated) produce three things:

1. summary — AT MOST 2 short sentences. State only the meeting's purpose and \
its overall outcome. Nothing else goes here.

2. key_takeaways — the real substance of the meeting as short, scannable \
bullets. Be COMPREHENSIVE: every important fact, figure, result, risk, or \
point raised must appear here. Lose no information — the 2-sentence summary \
deliberately omits detail, so the takeaways must carry all of it. Keep each \
bullet brief and telegraphic (ideally under 14 words), no filler words. Tag \
each bullet:
   - "decision" — something agreed, decided, or set as a target or policy.
   - "note" — any other important point, fact, number, result, or concern.

3. action_items — concrete tasks someone committed to. Give owner and \
deadline ONLY when actually stated; never invent them. Skip vague intentions.

Write everything in English. Keep proper names, product names, and figures \
exact."""

# ── RAG system prompt ─────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """\
You are SummarAI's meeting assistant. You are given structured summaries from \
past meetings in a project. Use ONLY the information in the provided context \
to answer the user's question.

Rules:
- If the answer is present in the context, answer clearly and cite the meeting \
  title(s) you drew from.
- If the answer is not in the context, say so explicitly — do not guess or \
  hallucinate facts.
- Keep your answer concise and factual.
- Reference meetings by their title and date."""

_whisper_model = None


def transcribe(path: str) -> str:
    global _whisper_model
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="Audio transcription requires faster-whisper. "
            "Install it with: pip install faster-whisper",
        )
    if _whisper_model is None:
        _whisper_model = WhisperModel(
            WHISPER_MODEL,
            compute_type="int8",
            cpu_threads=os.cpu_count() or 4,
        )
    segments, _info = _whisper_model.transcribe(
        path,
        beam_size=1,
        vad_filter=True,
        condition_on_previous_text=False,
        language="en",
    )
    return " ".join(segment.text.strip() for segment in segments)


def analyze(transcript: str) -> dict:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"<transcript>\n{transcript}\n</transcript>"}],
        output_config={"format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
    )
    text = next(block.text for block in response.content if block.type == "text")
    return json.loads(text)


def sb(method: str, path: str, **kwargs):
    """Thin wrapper for Supabase REST calls."""
    url = f"{SUPABASE_URL}/rest/v1/{path}"
    r = httpx.request(method, url, headers=SUPABASE_HEADERS, **kwargs)
    if r.status_code >= 400:
        print(f"SUPABASE ERROR: {r.status_code} — {r.text}")
        raise HTTPException(status_code=502, detail=f"Supabase error: {r.text}")
    return r.json()


# ── RAG helpers ───────────────────────────────────────────────────────────────

def embed(text: str, task_type: str = "RETRIEVAL_DOCUMENT") -> list[float]:
    """Embed text with Gemini embedding-001 (1536 dims for pgvector).

    task_type makes the embeddings *asymmetric*: documents are embedded with
    RETRIEVAL_DOCUMENT (at save time) and questions with RETRIEVAL_QUERY (at
    search time). Gemini optimises the two roles differently, so using the
    correct task type for each measurably improves retrieval ranking over
    embedding both with a single task type.
    """
    from google import genai
    from google.genai import types
    gemini_key = os.environ.get("GEMINI_API_KEY")
    if not gemini_key:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY is not set. Add it to your .env file to enable RAG.",
        )
    client = genai.Client(api_key=gemini_key)
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(
            output_dimensionality=1536,
            task_type=task_type,
        ),
    )
    return result.embeddings[0].values


def build_summary_text(s: dict) -> str:
    """Flatten a summarization record into a single string for embedding."""
    takeaways = s.get("key_takeaways", [])
    if isinstance(takeaways, str):
        takeaways = json.loads(takeaways)
    actions = s.get("action_items", [])
    if isinstance(actions, str):
        actions = json.loads(actions)

    parts = [
        f"Meeting: {s.get('meeting_title', 'Untitled')}",
        f"Summary: {s.get('summary', '')}",
    ]
    if takeaways:
        parts.append("Key takeaways: " + " | ".join(t["text"] for t in takeaways))
    if actions:
        parts.append("Action items: " + " | ".join(
            f"{a['task']} (owner: {a['owner'] or 'unassigned'}, deadline: {a['deadline'] or 'none'})"
            for a in actions
        ))
    return "\n".join(parts)


def retrieve(project_id: str, query_embedding: list[float], top_k: int = 3) -> list[dict]:
    """Call the match_summarizations Postgres function via Supabase RPC."""
    url = f"{SUPABASE_URL}/rest/v1/rpc/match_summarizations"
    payload = {
        "query_embedding": query_embedding,
        "match_project_id": project_id,
        "match_count": top_k,
    }
    r = httpx.post(url, headers=SUPABASE_HEADERS, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Supabase RPC error: {r.text}")
    return r.json()


def generate_answer(question: str, context_meetings: list[dict]) -> dict:
    """Build the augmented prompt and call Claude to produce a grounded answer."""
    if not context_meetings:
        return {
            "answer": "No relevant meetings were found in this project to answer your question.",
            "sources": [],
        }

    # Build context block
    context_parts = []
    for i, m in enumerate(context_meetings, 1):
        date_str = m.get("created_at", "")[:10]
        context_parts.append(
            f"[Meeting {i}: \"{m['meeting_title']}\" — {date_str}]\n"
            f"Summary: {m['summary']}\n"
            f"Key takeaways: {json.dumps(m.get('key_takeaways', []), ensure_ascii=False)}\n"
            f"Action items: {json.dumps(m.get('action_items', []), ensure_ascii=False)}"
        )
    context_block = "\n\n".join(context_parts)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1024,
        system=RAG_SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": (
                f"<context>\n{context_block}\n</context>\n\n"
                f"Question: {question}"
            ),
        }],
    )
    answer_text = next(block.text for block in response.content if block.type == "text")

    sources = [
        {
            "id": m["id"],
            "meeting_title": m["meeting_title"],
            "created_at": m.get("created_at", ""),
            "similarity": round(m.get("similarity", 0), 3),
        }
        for m in context_meetings
    ]
    return {"answer": answer_text, "sources": sources}


# ── Agentic meeting assistant ─────────────────────────────────────────────────
# A SINGLE tool-using agent. Given a question about a project, it decides which
# tools to call (semantic search, list meetings, list/overdue action items, draft
# a recap), reads the results, and loops — up to MAX_AGENT_STEPS — before
# answering. project_id is bound server-side into every tool, so the model can
# never reach another project's data (a built-in access guardrail). "Specialist"
# behaviour (drafting a recap) is exposed as a TOOL, not a second agent, to keep
# the mental model simple: one agent, one loop, a toolbox, a step cap.

MAX_AGENT_STEPS = 5        # hard cap on tool-use rounds → prevents infinite agent loops
MAX_QUESTION_CHARS = 2_000  # input guardrail on the user's question

AGENT_SYSTEM_PROMPT = """\
You are SummarAI's meeting assistant for ONE project. Today's date is {today}.

You help the user reason over the meetings saved in this project — their
summaries, decisions, notes, and action items. You have tools to: list every
meeting, search meetings semantically, list action items, find overdue action
items, and draft a follow-up recap.

How to work:
- Decide which tool(s) answer the question, call them, then read the results.
- Base every factual claim ONLY on what the tools return. Never invent
  meetings, owners, deadlines, decisions, or numbers.
- Cite the meeting title(s) you used. If the tools return nothing relevant,
  say so plainly — do not guess or fall back on outside knowledge.
- Be efficient: don't call more tools than you need. As soon as you can answer,
  answer."""


AGENT_TOOLS = [
    {
        "name": "list_meetings",
        "description": (
            "List every meeting saved in this project (title, date, one-line "
            "summary). Use for an overview, for counting meetings, or when the "
            "user names no specific topic."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "search_meetings",
        "description": (
            "Semantic search over this project's meetings. Returns the most "
            "relevant meetings with full summary, key takeaways, and action "
            "items. Use when the user asks about a topic, decision, or detail."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look for, in natural language."},
                "top_k": {"type": "integer", "description": "How many meetings to retrieve (1-5). Default 3."},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "list_action_items",
        "description": (
            "List action items across all meetings in this project, optionally "
            "filtered by owner. Each item has task, owner, deadline, and the "
            "meeting it came from."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {
                    "type": "string",
                    "description": "Filter to a person's name (case-insensitive). Use 'unassigned' for items with no owner. Omit for all.",
                },
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "find_overdue_action_items",
        "description": (
            "Return action items whose stated deadline is before a date "
            "(default: today). Items with no deadline, or a deadline that can't "
            "be parsed, are reported separately — never counted as overdue."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string", "description": "ISO date YYYY-MM-DD to compare against. Defaults to today."},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "draft_followup",
        "description": (
            "Draft a short, professional follow-up recap email for a specific "
            "meeting (one-line summary, decisions, action items with owners and "
            "deadlines). Provide the meeting title; best match is used."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "meeting_title": {"type": "string", "description": "The meeting to recap (matched by title)."},
            },
            "required": ["meeting_title"],
            "additionalProperties": False,
        },
    },
]


# ── Tool helpers ──────────────────────────────────────────────────────────────

def _as_list(value):
    """key_takeaways / action_items come back as a JSON list (jsonb) or a JSON
    string depending on the path — normalise both to a Python list."""
    if not value:
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return value


def _project_summarizations(project_id: str) -> list[dict]:
    rows = sb(
        "GET",
        f"summarizations?project_id=eq.{project_id}"
        "&select=id,meeting_title,summary,key_takeaways,action_items,created_at"
        "&order=created_at.desc",
    )
    for r in rows:
        r["key_takeaways"] = _as_list(r.get("key_takeaways"))
        r["action_items"] = _as_list(r.get("action_items"))
    return rows


def _source(m: dict) -> dict:
    src = {
        "id": m["id"],
        "meeting_title": m.get("meeting_title", ""),
        "created_at": m.get("created_at", ""),
    }
    if "similarity" in m and m["similarity"] is not None:
        src["similarity"] = round(m["similarity"], 3)
    return src


def _flatten_actions(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        for a in r.get("action_items", []):
            out.append({
                "task": a.get("task", ""),
                "owner": a.get("owner"),
                "deadline": a.get("deadline"),
                "meeting_title": r.get("meeting_title", ""),
                "meeting_date": (r.get("created_at") or "")[:10],
            })
    return out


# Formats tried after stripping ordinal suffixes and commas (e.g. "June 18th,
# 2026" → "June 18 2026"). A deadline with no year defaults to the current year.
_DATE_FORMATS = (
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%Y/%m/%d",
    "%B %d %Y", "%b %d %Y", "%d %B %Y", "%d %b %Y",
    "%B %d", "%b %d", "%d %B", "%d %b",
)
_ORDINAL_RE = re.compile(r"(\d{1,2})(st|nd|rd|th)\b", re.IGNORECASE)


def _parse_date(s):
    """Best-effort parse of a free-text deadline → a date, or None.

    Handles ISO dates and natural phrasings the summariser emits ("June 18",
    "June 18th, 2026", "18 June", "06/18/2026"). Anything genuinely relative
    ("next Friday", "after lunch", "30 minutes") returns None and is reported
    as *unparseable* rather than silently treated as overdue.
    """
    if not s or not isinstance(s, str):
        return None
    cleaned = _ORDINAL_RE.sub(r"\1", s.strip()).replace(",", " ")
    cleaned = " ".join(cleaned.split())  # collapse whitespace
    try:
        return date.fromisoformat(cleaned[:10])
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            d = datetime.strptime(cleaned, fmt)
            if d.year == 1900:  # formats without a year default to 1900
                d = d.replace(year=date.today().year)
            return d.date()
        except ValueError:
            continue
    return None


# ── Tool implementations (each returns (text_for_model, sources)) ─────────────

def tool_list_meetings(project_id: str):
    rows = _project_summarizations(project_id)
    if not rows:
        return ("No meetings have been saved to this project yet.", [])
    lines = [
        f'- "{r["meeting_title"]}" ({(r.get("created_at") or "")[:10]}): {r.get("summary", "")}'
        for r in rows
    ]
    return (f"{len(rows)} meeting(s) in this project:\n" + "\n".join(lines),
            [_source(r) for r in rows])


def tool_search_meetings(project_id: str, query: str, top_k: int = 3):
    top_k = max(1, min(int(top_k or 3), 5))
    try:
        qvec = embed(query, task_type="RETRIEVAL_QUERY")
    except HTTPException as exc:
        return (f"Semantic search is unavailable ({exc.detail}). Try list_meetings instead.", [])
    matches = retrieve(project_id, qvec, top_k=top_k)
    if not matches:
        return ("No meetings matched that query.", [])
    parts = []
    for i, m in enumerate(matches, 1):
        parts.append(
            f'[{i}] "{m["meeting_title"]}" — {(m.get("created_at") or "")[:10]} '
            f'(similarity {round(m.get("similarity", 0), 3)})\n'
            f'Summary: {m.get("summary", "")}\n'
            f'Key takeaways: {json.dumps(_as_list(m.get("key_takeaways")), ensure_ascii=False)}\n'
            f'Action items: {json.dumps(_as_list(m.get("action_items")), ensure_ascii=False)}'
        )
    return ("\n\n".join(parts), [_source(m) for m in matches])


def tool_list_action_items(project_id: str, owner: str | None = None):
    rows = _project_summarizations(project_id)
    actions = _flatten_actions(rows)
    if owner:
        o = owner.strip().lower()
        if o in ("unassigned", "none", "nobody", "null", "no owner"):
            actions = [a for a in actions if not a["owner"]]
        else:
            actions = [a for a in actions if a["owner"] and o in a["owner"].lower()]
    if not actions:
        return ("No matching action items found.", [])
    lines = [
        f'- {a["task"]} | owner: {a["owner"] or "unassigned"} | '
        f'deadline: {a["deadline"] or "none"} | from "{a["meeting_title"]}" ({a["meeting_date"]})'
        for a in actions
    ]
    srcs = {r["id"]: _source(r) for r in rows if r.get("action_items")}
    return (f"{len(actions)} action item(s):\n" + "\n".join(lines), list(srcs.values()))


def tool_find_overdue_action_items(project_id: str, as_of_date: str | None = None):
    rows = _project_summarizations(project_id)
    actions = _flatten_actions(rows)
    as_of = _parse_date(as_of_date) or date.today()
    overdue, unparseable = [], []
    for a in actions:
        if not a["deadline"]:
            continue
        d = _parse_date(a["deadline"])
        if d is None:
            unparseable.append(a)
        elif d < as_of:
            overdue.append((d, a))
    overdue.sort(key=lambda x: x[0])
    parts = [f"As of {as_of.isoformat()}:"]
    if overdue:
        parts.append(f"OVERDUE ({len(overdue)}):")
        parts += [
            f'- {a["task"]} | owner: {a["owner"] or "unassigned"} | '
            f'due {d.isoformat()} | from "{a["meeting_title"]}"'
            for d, a in overdue
        ]
    else:
        parts.append("No overdue action items with parseable deadlines.")
    if unparseable:
        parts.append(
            f"Deadline stated but not parseable ({len(unparseable)}): "
            + "; ".join(f'{a["task"]} ("{a["deadline"]}")' for a in unparseable)
        )
    return ("\n".join(parts), [_source(r) for r in rows if r.get("action_items")])


def tool_draft_followup(project_id: str, meeting_title: str):
    rows = _project_summarizations(project_id)
    if not rows:
        return ("No meetings are saved in this project to draft from.", [])
    needle = (meeting_title or "").strip().lower()
    match = next((r for r in rows if needle and needle in r.get("meeting_title", "").lower()), None)
    if match is None:  # fall back to semantic best-match
        try:
            qvec = embed(meeting_title, task_type="RETRIEVAL_QUERY")
            ms = retrieve(project_id, qvec, top_k=1)
            if ms:
                match = next((r for r in rows if r["id"] == ms[0]["id"]), None)
        except HTTPException:
            pass
    if match is None:
        match = rows[0]
    context = (
        f'Meeting: {match["meeting_title"]} ({(match.get("created_at") or "")[:10]})\n'
        f'Summary: {match.get("summary", "")}\n'
        f'Key takeaways: {json.dumps(match.get("key_takeaways", []), ensure_ascii=False)}\n'
        f'Action items: {json.dumps(match.get("action_items", []), ensure_ascii=False)}'
    )
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=600,
        system=(
            "You draft a concise, professional follow-up recap email from a "
            "meeting record. Use ONLY the given facts — never invent owners or "
            "deadlines. Include a one-line summary, the key decisions, and a "
            "clear action-item list with owners and deadlines. Under 200 words."
        ),
        messages=[{"role": "user", "content": context}],
    )
    draft = "".join(b.text for b in resp.content if b.type == "text")
    return (f'Draft follow-up for "{match["meeting_title"]}":\n\n{draft}', [_source(match)])


_TOOL_IMPLS = {
    "list_meetings": tool_list_meetings,
    "search_meetings": tool_search_meetings,
    "list_action_items": tool_list_action_items,
    "find_overdue_action_items": tool_find_overdue_action_items,
    "draft_followup": tool_draft_followup,
}


def run_agent(question: str, project_id: str) -> dict:
    """Run the single tool-using agent loop, bounded by MAX_AGENT_STEPS.

    project_id is bound into every tool call here, so the model supplies only
    semantic arguments and can never query another project.
    """
    client = anthropic.Anthropic()
    system = AGENT_SYSTEM_PROMPT.format(today=date.today().isoformat())
    messages = [{"role": "user", "content": question}]
    trace: list[dict] = []
    sources: dict[str, dict] = {}

    for step in range(1, MAX_AGENT_STEPS + 1):
        resp = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1500,
            system=system,
            tools=AGENT_TOOLS,
            messages=messages,
        )
        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            return {
                "answer": answer,
                "sources": list(sources.values()),
                "trace": trace,
                "steps": step,
                "hit_cap": False,
            }

        # Execute every tool the model requested this round.
        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            impl = _TOOL_IMPLS.get(block.name)
            if impl is None:
                text, srcs = (f"Unknown tool '{block.name}'.", [])
            else:
                try:
                    text, srcs = impl(project_id, **block.input)
                except Exception as exc:  # noqa: BLE001 — surface tool errors to the model
                    text, srcs = (f"Tool error: {exc}", [])
            trace.append({"tool": block.name, "input": block.input, "result": text[:1500]})
            for s in srcs:
                sources[s["id"]] = s
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": text,
            })
        messages.append({"role": "user", "content": tool_results})

    # Step cap reached → force a final answer with NO further tools.
    final = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=1500,
        system=system + (
            "\n\nYou have reached the tool-call limit. Give your best answer now "
            "using only what the tools already returned. Do not request more tools."
        ),
        messages=messages,
    )
    answer = "".join(b.text for b in final.content if b.type == "text")
    return {
        "answer": answer,
        "sources": list(sources.values()),
        "trace": trace,
        "steps": MAX_AGENT_STEPS,
        "hit_cap": True,
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/process")
async def process(file: UploadFile = File(...)):
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key.",
        )

    extension = Path(file.filename or "").suffix.lower()
    if extension in TEXT_EXTENSIONS:
        raw = await file.read()
        transcript = raw.decode("utf-8", errors="replace")
    elif extension in AUDIO_EXTENSIONS:
        with tempfile.NamedTemporaryFile(suffix=extension, delete=False) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            transcript = transcribe(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{extension}'. "
            f"Use audio/video ({', '.join(sorted(AUDIO_EXTENSIONS))}) "
            f"or text ({', '.join(sorted(TEXT_EXTENSIONS))}).",
        )

    transcript = transcript.strip()
    if not transcript:
        raise HTTPException(status_code=400, detail="The file contains no usable text.")
    if len(transcript) > MAX_TRANSCRIPT_CHARS:
        raise HTTPException(
            status_code=400,
            detail="Transcript is too long for the POC (max ~300K characters). "
            "Split the meeting into parts.",
        )

    try:
        result = analyze(transcript)
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="Invalid Anthropic API key.")
    except anthropic.APIError as exc:
        raise HTTPException(status_code=502, detail=f"Claude API error: {exc.message}")
    except (json.JSONDecodeError, StopIteration):
        raise HTTPException(status_code=502, detail="Model returned a malformed response. Try again.")

    result["transcript_chars"] = len(transcript)
    return result


# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: str = ""


@app.get("/api/projects")
def list_projects():
    return sb("GET", "projects?select=*&order=created_at.desc")


@app.post("/api/projects")
def create_project(body: ProjectCreate):
    rows = sb("POST", "projects", json={"name": body.name, "description": body.description})
    return rows[0] if isinstance(rows, list) else rows


@app.delete("/api/projects/{project_id}")
def delete_project(project_id: str):
    sb("DELETE", f"projects?id=eq.{project_id}")
    return {"ok": True}


# ── Summarizations ────────────────────────────────────────────────────────────

class SummarizationSave(BaseModel):
    project_id: str
    meeting_title: str
    language: str = "en"
    detected_language: str = "en"
    transcript_chars: int
    summary: str
    key_takeaways: list
    action_items: list


@app.get("/api/projects/{project_id}/summarizations")
def list_summarizations(project_id: str):
    return sb("GET", f"summarizations?project_id=eq.{project_id}&order=created_at.desc")


@app.post("/api/summarizations")
def save_summarization(body: SummarizationSave):
    payload = body.model_dump()
    payload["key_takeaways"] = json.dumps(payload["key_takeaways"])
    payload["action_items"] = json.dumps(payload["action_items"])

    # ── RAG: embed the summary for later retrieval ────────────────────────────
    # We embed a flattened text representation of the full summarization.
    # If GEMINI_API_KEY is missing we skip silently — the save still succeeds,
    # the record just won't be retrievable via /ask until re-embedded.
    try:
        text_to_embed = build_summary_text({
            "meeting_title": body.meeting_title,
            "summary": body.summary,
            "key_takeaways": body.key_takeaways,
            "action_items": body.action_items,
        })
        payload["embedding"] = embed(text_to_embed)
    except HTTPException:
        # No GEMINI_API_KEY — save without embedding, warn in logs
        print("Warning: GEMINI_API_KEY not set — summarization saved without embedding.")
    except Exception as exc:
        print(f"Warning: embedding failed ({exc}) — summarization saved without embedding.")
    # ─────────────────────────────────────────────────────────────────────────

    rows = sb("POST", "summarizations", json=payload)
    return rows[0] if isinstance(rows, list) else rows


@app.delete("/api/summarizations/{summ_id}")
def delete_summarization(summ_id: str):
    sb("DELETE", f"summarizations?id=eq.{summ_id}")
    return {"ok": True}


# ── RAG: Ask a question about a project's meeting history ─────────────────────

class AskBody(BaseModel):
    question: str
    top_k: int = 3


@app.post("/api/projects/{project_id}/ask")
def ask_project(project_id: str, body: AskBody):
    """Agentic endpoint. A single tool-using agent answers questions about this
    project's meetings: it chooses among semantic search, meeting listing,
    action-item, overdue, and draft-recap tools, observes the results, and loops
    (capped at MAX_AGENT_STEPS) before producing a grounded, cited answer.

    Returns:
        answer   — the agent's grounded response
        sources  — meetings cited (id, title, date, similarity when available)
        trace    — the tool calls the agent made (for transparency / the demo)
        steps    — how many reasoning rounds it used
    """
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(q) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Question is too long (max {MAX_QUESTION_CHARS} characters).",
        )
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is not set.")
    return run_agent(q, project_id)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
