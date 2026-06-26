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
import tempfile
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

def embed(text: str) -> list[float]:
    """Embed text using Gemini embedding-001 (truncated to 1536 dims for pgvector compatibility)."""
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
        config=types.EmbedContentConfig(output_dimensionality=1536),
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


# ── Agent: bounded tool-use loop over the meeting history ─────────────────────
#
# A single tool-using agent (not multi-agent) answers project questions by
# calling read-only tools in a loop. Guardrails: a hard step cap (anti-loop),
# a grounded system prompt (anti-hallucination), and a full tool-call trace
# returned to the frontend for transparency / live-demo inspection.

MAX_AGENT_STEPS = 5

AGENT_SYSTEM_PROMPT = """\
You are SummarAI's meeting assistant, an agent with tools to inspect a \
project's meeting history. Answer the user's question by calling tools to \
gather evidence, then give a grounded, concise answer.

Rules:
- Ground every claim in tool results. Never invent meetings, owners, dates, \
  or facts that were not returned by a tool.
- If after using the available tools the answer isn't supported by any \
  meeting, say so explicitly instead of guessing.
- Cite meeting titles when you reference their content.
- Use the fewest tool calls needed — don't call a tool you've already \
  called with the same arguments.
- When asked to draft something (a recap, a follow-up email), use the \
  draft_followup tool rather than writing it yourself from memory."""

AGENT_TOOLS = [
    {
        "name": "search_meetings",
        "description": "Semantic search over this project's past meeting summaries. Use this first to find meetings relevant to the question.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural-language search query."},
                "top_k": {"type": "integer", "description": "Max meetings to return.", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_meeting",
        "description": "Fetch the full record (summary, all key takeaways, all action items) for one meeting by id.",
        "input_schema": {
            "type": "object",
            "properties": {"meeting_id": {"type": "string"}},
            "required": ["meeting_id"],
        },
    },
    {
        "name": "list_action_items",
        "description": "List action items across every meeting in the project, optionally filtered by owner.",
        "input_schema": {
            "type": "object",
            "properties": {
                "owner": {"type": "string", "description": "Filter to items owned by this person (case-insensitive substring match). Omit to list all."},
            },
        },
    },
    {
        "name": "find_overdue_action_items",
        "description": "List action items whose deadline is a parseable date before as_of_date. Items with vague deadlines (e.g. 'next meeting') are skipped, not assumed overdue.",
        "input_schema": {
            "type": "object",
            "properties": {
                "as_of_date": {"type": "string", "description": "ISO date (YYYY-MM-DD) to compare deadlines against. Defaults to today."},
            },
        },
    },
    {
        "name": "draft_followup",
        "description": "Draft a short follow-up recap/email for one meeting, grounded in its stored summary and action items.",
        "input_schema": {
            "type": "object",
            "properties": {"meeting_id": {"type": "string"}},
            "required": ["meeting_id"],
        },
    },
]


def _tool_search_meetings(project_id: str, args: dict) -> dict:
    query_vector = embed(args["query"])
    matches = retrieve(project_id, query_vector, top_k=args.get("top_k", 3))
    return {
        "meetings": [
            {
                "id": m["id"],
                "meeting_title": m["meeting_title"],
                "created_at": m.get("created_at", ""),
                "similarity": round(m.get("similarity", 0), 3),
                "summary": m["summary"],
            }
            for m in matches
        ]
    }


def _tool_get_meeting(project_id: str, args: dict) -> dict:
    rows = sb(
        "GET",
        f"summarizations?id=eq.{args['meeting_id']}&project_id=eq.{project_id}"
        "&select=id,project_id,meeting_title,language,detected_language,"
        "transcript_chars,summary,key_takeaways,action_items,created_at",
    )
    if not rows:
        return {"error": "Meeting not found in this project."}
    return rows[0]


def _all_summarizations(project_id: str) -> list[dict]:
    return sb("GET", f"summarizations?project_id=eq.{project_id}&order=created_at.desc&select=*")


def _tool_list_action_items(project_id: str, args: dict) -> dict:
    owner_filter = (args.get("owner") or "").strip().lower()
    items = []
    for s in _all_summarizations(project_id):
        actions = s.get("action_items", [])
        if isinstance(actions, str):
            actions = json.loads(actions)
        for a in actions:
            if owner_filter and owner_filter not in (a.get("owner") or "").lower():
                continue
            items.append({**a, "meeting_title": s["meeting_title"], "meeting_id": s["id"]})
    return {"action_items": items}


def _tool_find_overdue_action_items(project_id: str, args: dict) -> dict:
    import datetime as _dt
    as_of = args.get("as_of_date")
    as_of_date = _dt.date.fromisoformat(as_of) if as_of else _dt.date.today()

    overdue = []
    for s in _all_summarizations(project_id):
        actions = s.get("action_items", [])
        if isinstance(actions, str):
            actions = json.loads(actions)
        for a in actions:
            deadline = a.get("deadline")
            if not deadline:
                continue
            try:
                deadline_date = _dt.date.fromisoformat(deadline[:10])
            except ValueError:
                continue  # vague deadline like "next meeting" — skip, don't guess
            if deadline_date < as_of_date:
                overdue.append({**a, "meeting_title": s["meeting_title"], "meeting_id": s["id"]})
    return {"overdue_action_items": overdue, "as_of_date": as_of_date.isoformat()}


def _tool_draft_followup(project_id: str, args: dict) -> dict:
    """Bounded, grounded generation — a tool the agent calls, not a second agent."""
    record = _tool_get_meeting(project_id, args)
    if "error" in record:
        return record

    actions = record.get("action_items", [])
    if isinstance(actions, str):
        actions = json.loads(actions)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=512,
        system=(
            "Draft a short, professional follow-up recap email for one meeting. "
            "Use ONLY the summary and action items given below — invent nothing. "
            "Keep it under 150 words."
        ),
        messages=[{
            "role": "user",
            "content": (
                f"Meeting: {record['meeting_title']}\n"
                f"Summary: {record['summary']}\n"
                f"Action items: {json.dumps(actions, ensure_ascii=False)}"
            ),
        }],
    )
    draft = next(block.text for block in response.content if block.type == "text")
    return {"draft": draft}


AGENT_TOOL_IMPLS = {
    "search_meetings": _tool_search_meetings,
    "get_meeting": _tool_get_meeting,
    "list_action_items": _tool_list_action_items,
    "find_overdue_action_items": _tool_find_overdue_action_items,
    "draft_followup": _tool_draft_followup,
}


def run_agent(question: str, project_id: str) -> dict:
    """Bounded tool-use loop: Claude requests a tool, we execute it, repeat.

    Guardrails: MAX_AGENT_STEPS caps the loop (anti agent-loop / runaway
    reasoning); every tool is read-only or a single bounded generation call,
    so there is no way for the agent to take an irreversible action.
    """
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": question}]
    trace = []

    for step in range(MAX_AGENT_STEPS):
        response = client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=1024,
            system=AGENT_SYSTEM_PROMPT,
            tools=AGENT_TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            answer_text = next(
                (b.text for b in response.content if b.type == "text"), ""
            )
            return {"answer": answer_text, "sources": _trace_to_sources(trace), "trace": trace}

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            impl = AGENT_TOOL_IMPLS.get(block.name)
            try:
                result = impl(project_id, block.input) if impl else {"error": f"Unknown tool {block.name}"}
            except Exception as exc:
                result = {"error": str(exc)}
            trace.append({"tool": block.name, "input": block.input, "output": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "I wasn't able to reach a grounded answer within the step limit. Try rephrasing or narrowing the question.",
        "sources": _trace_to_sources(trace),
        "trace": trace,
    }


def _trace_to_sources(trace: list[dict]) -> list[dict]:
    """Collect cited meetings out of search_meetings / get_meeting tool calls for the UI."""
    seen = {}
    for entry in trace:
        output = entry["output"]
        candidates = output.get("meetings") if entry["tool"] == "search_meetings" else (
            [output] if entry["tool"] == "get_meeting" and "id" in output else []
        )
        for m in candidates:
            seen[m["id"]] = {
                "id": m["id"],
                "meeting_title": m["meeting_title"],
                "created_at": m.get("created_at", ""),
                "similarity": m.get("similarity"),
            }
    return list(seen.values())


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
    """
    Agentic Q&A endpoint: a single tool-using Claude agent answers questions
    about this project's meeting history. The agent loops over read-only
    tools (semantic search, full-record lookup, action-item listing,
    overdue detection, follow-up drafting) up to MAX_AGENT_STEPS times, then
    returns a grounded answer.

    Returns:
        answer  — Claude's response, grounded in tool results
        sources — meetings the agent looked at while answering
        trace   — full tool-call trace, for transparency / live-demo inspection
    """
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if not os.environ.get("GEMINI_API_KEY"):
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not set.")

    return run_agent(body.question, project_id)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
