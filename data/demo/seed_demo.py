"""Seed a demo project with realistic, dated meetings.

The AMI corpus meetings have vague deadlines ("30 minutes"), so the agent's
headline accountability feature — "what's overdue?" — has nothing real to show.
This script loads the hand-written Project Atlas transcripts (data/demo/
transcripts/), runs each through the REAL summarisation pipeline (Claude), and
saves the results — with Gemini embeddings — into a dedicated demo project so
search / list-action-items / overdue / draft-recap all have meaningful data.

Run it AFTER filling in .env (ANTHROPIC_API_KEY, GEMINI_API_KEY, and your own
SUPABASE_URL / SUPABASE_ANON_KEY):

    .venv/bin/python data/demo/seed_demo.py

Re-running reuses the existing demo project and skips meetings already saved.
"""
import json
import sys
from pathlib import Path

# Make `import app` work from src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import app  # noqa: E402

PROJECT_NAME = "Project Atlas (demo)"
TRANSCRIPTS = Path(__file__).resolve().parent / "transcripts"

# (meeting title, backdated created_at, transcript filename)
MEETINGS = [
    ("Project Atlas — Kickoff",             "2026-06-02T12:00:00Z", "01_kickoff.txt"),
    ("Project Atlas — Design Review",       "2026-06-09T12:00:00Z", "02_design_review.txt"),
    ("Project Atlas — Sprint Planning",     "2026-06-16T12:00:00Z", "03_sprint_planning.txt"),
    ("Project Atlas — Mid-Sprint Check-in", "2026-06-23T12:00:00Z", "04_mid_sprint_checkin.txt"),
]


def get_or_create_project() -> str:
    projects = app.sb("GET", "projects?select=id,name")
    for p in projects:
        if p["name"] == PROJECT_NAME:
            print(f"Reusing existing project {p['id']}")
            return p["id"]
    row = app.sb("POST", "projects", json={
        "name": PROJECT_NAME,
        "description": "Synthetic demo data for the SummarAI agent (dated action items).",
    })
    pid = (row[0] if isinstance(row, list) else row)["id"]
    print(f"Created project {pid}")
    return pid


def already_saved(project_id: str) -> set[str]:
    rows = app.sb("GET", f"summarizations?project_id=eq.{project_id}&select=meeting_title")
    return {r["meeting_title"] for r in rows}


def main() -> None:
    pid = get_or_create_project()
    existing = already_saved(pid)

    for title, created_at, fname in MEETINGS:
        if title in existing:
            print(f"  • skip (already saved): {title}")
            continue
        text = (TRANSCRIPTS / fname).read_text(encoding="utf-8")
        print(f"  • analysing: {title} …")
        result = app.analyze(text)  # {summary, key_takeaways, action_items}

        payload = {
            "project_id": pid,
            "meeting_title": title,
            "language": "en",
            "detected_language": "en",
            "transcript_chars": len(text),
            "summary": result["summary"],
            "key_takeaways": json.dumps(result["key_takeaways"]),
            "action_items": json.dumps(result["action_items"]),
            "created_at": created_at,
        }
        try:
            payload["embedding"] = app.embed(app.build_summary_text({
                "meeting_title": title,
                "summary": result["summary"],
                "key_takeaways": result["key_takeaways"],
                "action_items": result["action_items"],
            }))
        except Exception as exc:  # noqa: BLE001
            print(f"    (embedding skipped — search will be limited: {exc})")

        app.sb("POST", "summarizations", json=payload)
        n_take = len(result["key_takeaways"])
        n_act = len(result["action_items"])
        print(f"    saved — {n_take} takeaways, {n_act} action items")

    print(f"\nDone. Open the app, select {PROJECT_NAME!r}, and try:")
    print("  • What action items are overdue?")
    print("  • Who owns the API?")
    print("  • What did we decide about the launch date?")
    print("  • Draft a recap of the kickoff.")


if __name__ == "__main__":
    main()
