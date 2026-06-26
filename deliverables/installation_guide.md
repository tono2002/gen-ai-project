# SummarAI: Installation & Execution Guide

This guide takes you from a fresh clone to a running app. It assumes you are comfortable in a terminal.

## Requirements

| Requirement | Notes |
|---|---|
| Python 3.9+ | Developed and tested on 3.9 |
| An Anthropic API key | Used for the summarisation step ([console.anthropic.com](https://console.anthropic.com)) |
| A Gemini API key | Used for the RAG embedding step — free at [aistudio.google.com](https://aistudio.google.com) |
| ~1 GB free disk | For the Whisper transcription model, downloaded on first use |

You do **not** need to install ffmpeg separately — audio decoding is handled by a library bundled with the transcription engine.

## Setup

```bash
# 1. Clone and enter the project
git clone https://github.com/tono2002/nlp-project.git
cd nlp-project

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1        # Windows (PowerShell)

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

Copy the example environment file and add your keys:

```bash
cp .env.example .env
```

Then open `.env` and set:

```
ANTHROPIC_API_KEY=sk-ant-...      # required — summarisation
GEMINI_API_KEY=AIza...            # required — RAG embeddings
```

The Supabase variables (`SUPABASE_URL`, `SUPABASE_ANON_KEY`) are optional — working defaults are shipped in `app.py`. Your `.env` is git-ignored so no keys will be committed.

**Getting a Gemini API key:** go to [aistudio.google.com](https://aistudio.google.com), sign in with a Google account, and click **Get API key**. The free tier allows 1,500 embedding requests per day — more than enough for normal use.

## Database setup (required for projects and RAG)

The app uses Supabase for persisting meeting summaries and the RAG vector index. Two SQL migrations must be run in order.

1. Open the [Supabase SQL Editor](https://supabase.com/dashboard) for your project.
2. Run **`supabase_schema.sql`** — creates the `projects` and `summarizations` tables.
3. Run **`supabase_schema_rag.sql`** — enables `pgvector`, adds the `embedding` column to `summarizations`, creates the cosine-distance index, and defines the `match_summarizations` search function.

If you only run the first migration, the app works for summarisation but the Ask panel will return no results.

## Running the app

```bash
uvicorn src.app:app --reload
```

Uvicorn prints a local URL — open it in a browser:

```
Uvicorn running on http://127.0.0.1:8000
```

You'll see the upload area. Drop in an audio file (`.mp4`, `.mp3`, `.wav`, `.m4a`, …) or a text transcript (`.txt`, `.vtt`, …) and click **Summarise**.

A few notes on first use:

- The first time you process **audio**, the app downloads the Whisper `base.en` model (~150 MB). This happens once; later runs reuse it.
- Transcription runs on the CPU and takes roughly a tenth of the recording's length (a 20-minute meeting transcribes in about two minutes). Text transcripts skip this step entirely.
- To trade accuracy for speed, set `WHISPER_MODEL=tiny.en` in `.env`.

## Back-filling existing meetings for RAG

If you have meetings saved before the RAG migration was applied, run the back-fill script once to embed them:

```bash
python embed_existing.py
```

This fetches all summarisations without embeddings, embeds them with Gemini, and writes the vectors back to Supabase. It is safe to run multiple times.

## Project structure

```
nlp-project/
├── src/
│   ├── app.py              # FastAPI app: transcription, summarisation, RAG endpoints
│   └── static/
│       └── index.html      # Single-page front end (includes Ask panel)
├── data/eval/
│   ├── transcribe_all.py   # Batch-transcribes the evaluation recordings
│   ├── evaluate_rouge.py   # Scores summaries against AMI human references
│   ├── _match.py           # Maps recordings to AMI meeting IDs
│   ├── transcripts/        # Whisper transcripts of the 30 eval meetings
│   └── results.json        # Evaluation results
├── docs/                   # User manual, this guide
├── deliverables/           # Report, executive summary, slides, reflections
├── prompts/                # System prompt documentation (summarisation + RAG)
├── requirements.txt
├── supabase_schema.sql     # Core tables (run first)
├── supabase_schema_rag.sql # pgvector + embedding column + search function (run second)
├── embed_existing.py       # Back-fill script for meetings saved before RAG
├── .env.example
└── README.md
```

## Reproducing the evaluation

The numbers in the technical report come from `data/eval/`. To regenerate them:

```bash
# 1. Download the AMI human annotations (~23 MB, not committed)
mkdir -p data/eval/ami_raw && cd data/eval/ami_raw
curl -sO https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip -q ami_public_manual_1.6.2.zip && cd -

# 2. Run the evaluation (uses the committed transcripts in data/eval/transcripts/)
python data/eval/evaluate_rouge.py
```

This runs SummarAI on the 30 evaluation transcripts and prints ROUGE scores plus action-item coverage, writing the full breakdown to `data/eval/results.json`.

## Troubleshooting

**`anthropic.AuthenticationError`**: the key in `.env` is missing or wrong. Check it at console.anthropic.com.

**`ANTHROPIC_API_KEY is not set`**: you started the server before creating `.env`, or in a different shell. Confirm `.env` exists in the project root and restart.

**`GEMINI_API_KEY is not set`**: add `GEMINI_API_KEY=AIza...` to `.env`. The app starts without it but the Ask panel will return an error.

**Ask panel returns "No relevant meetings found"**: the meeting was saved before the RAG migration. Run `python embed_existing.py` to back-fill embeddings.

**First audio upload hangs for a minute**: it's downloading the Whisper model. Watch the terminal; it only happens once.

**Transcription feels slow**: set `WHISPER_MODEL=tiny.en` in `.env` for a faster (slightly less accurate) model.
