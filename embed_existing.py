"""One-time script to embed all existing summarizations that have no embedding.

Run this if you have meetings saved before the RAG feature was added.
It fetches all summarizations without an embedding, embeds them with Gemini,
and writes the vectors back to Supabase. Safe to run multiple times.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

import httpx
from google import genai
from google.genai import types

SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ssooqczamcqxpvcpeebv.supabase.co")
SUPABASE_KEY = os.environ.get(
    "SUPABASE_ANON_KEY",
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNzb29xY3phbWNxeHB2Y3BlZWJ2Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3ODEyNzU0MTEsImV4cCI6MjA5Njg1MTQxMX0.4KbYBc9qY-6Gq_jsdiTWZTdis14xEpPzW0G66qAuq4E",
)
HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
}

gemini_key = os.environ.get("GEMINI_API_KEY")
if not gemini_key:
    raise SystemExit("GEMINI_API_KEY is not set. Add it to your .env file.")

client = genai.Client(api_key=gemini_key)


def embed(text: str) -> list:
    """Embed text using Gemini embedding-001, truncated to 1536 dims."""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=1536),
    )
    return result.embeddings[0].values


# Fetch all summarizations without an embedding
r = httpx.get(
    f"{SUPABASE_URL}/rest/v1/summarizations?embedding=is.null&select=*",
    headers=HEADERS,
)
rows = r.json()
print(f"Found {len(rows)} summarizations without embeddings")

if not rows:
    print("Nothing to do.")
else:
    for row in rows:
        takeaways = row.get("key_takeaways", [])
        if isinstance(takeaways, str):
            takeaways = json.loads(takeaways)
        actions = row.get("action_items", [])
        if isinstance(actions, str):
            actions = json.loads(actions)

        text = "\n".join([
            f"Meeting: {row.get('meeting_title', 'Untitled')}",
            f"Summary: {row.get('summary', '')}",
            "Key takeaways: " + " | ".join(t["text"] for t in takeaways),
            "Action items: " + " | ".join(a["task"] for a in actions),
        ])

        vector = embed(text)

        patch = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/summarizations?id=eq.{row['id']}",
            headers=HEADERS,
            json={"embedding": vector},
        )
        status = "✓" if patch.status_code < 300 else "✗"
        print(f"  {status} {row.get('meeting_title', row['id'])} — status {patch.status_code}")

print("Done.")
