# Individual Reflection: Bojana

## My contributions

### Backend and NLP pipeline
I built the entire server-side application (`src/app.py`). This includes the FastAPI server, the file upload endpoint that accepts audio, video, and text formats, and the full two-stage NLP pipeline: routing audio files through `faster-whisper` for transcription, then passing the resulting transcript to Claude Haiku 4.5 to extract the summary, key takeaways, and action items. I also built all the project management endpoints — creating, listing, and deleting projects and saved summarizations — which talk to Supabase via a thin HTTP wrapper I wrote to avoid adding a full Supabase client library as a dependency.

One non-trivial part of the backend was handling the different input types correctly. Audio and video files need to be written to a temporary file on disk before Whisper can process them, because the model requires a file path rather than a byte stream. Text and transcript files can be read directly into memory. Getting this routing right and cleaning up temporary files reliably (even when transcription fails) required careful use of try/finally.

### System prompt and output schema
I designed the system prompt and the JSON output schema that enforces the structure Claude must return. The schema defines three required fields — `summary`, `key_takeaways`, and `action_items` — with strict types and no additional properties allowed. Early versions of the prompt produced inconsistent output: action items appearing inside the summary, takeaways missing their type tag, or the model inventing owner names that were not in the transcript. I iterated on the prompt until the model reliably separated the three output types and only included owners and deadlines when they were explicitly stated in the meeting. The final system prompt and schema are saved in `prompts/` so the team can reference and defend them.

### Labeled transcripts
I created approximately 8 transcripts for `data/eval/eval_set.jsonl`, covering standup meetings, sprint planning sessions, and one edge case where a speaker discusses a future action hypothetically without committing to it — a scenario designed to test whether the model invents phantom action items.

## What I learned

This project gave me a real understanding of prompt engineering as an NLP technique rather than just a trick. When we started, the model would sometimes invent owners that were not mentioned in the transcript, or merge two separate action items into one. Fixing this required understanding why the model was doing it — it was following conversational implicature, inferring responsibility from context the way a human would — and then writing a prompt constraint specific enough to override that default behavior. That felt much more like NLP work than I expected. I also came to appreciate what makes LLM-based information extraction different from traditional approaches: the model handles paraphrase, messy grammar, and implicit references naturally, which would each require explicit rules in a classical pipeline. The cost is that you cannot fully predict or control the output, which is exactly why structured output schemas and evaluation against gold labels are necessary.
