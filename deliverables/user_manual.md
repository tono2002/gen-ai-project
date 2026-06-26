# SummarAI: User Manual

## What it does

SummarAI takes a meeting recording and gives you back a written record you can actually use. Hand it an audio file or a text transcript, and it returns:

- a **summary**: two sentences on the meeting's purpose and outcome;
- **key takeaways**: the main points, each tagged as a *decision* or a *note*;
- **action items**: concrete tasks, with the owner and deadline whenever they were stated.

You can save any result into a **project**, so a team builds up a history of summarised meetings in one place.

The app works in English.

## Getting started

Open the app in a browser. If you are running it locally, follow the [installation guide](installation_guide.md) and then go to `http://localhost:8000`. No account or sign-in is needed.

The screen has a projects panel on the left (which you can hide with the ☰ button) and the summariser in the middle.

## Summarising a meeting

**1. Add the recording.** Drag a file onto the upload area, or click it to browse. SummarAI accepts:

| Type | Formats |
|---|---|
| Audio / video | `.mp4`, `.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`, `.flac` |
| Text transcript | `.txt`, `.md`, `.vtt`, `.srt` |

If you upload audio, the app transcribes it automatically before summarising. Transcription runs locally, the audio never leaves your machine. If you already have the transcript as text, upload that instead and the app skips straight to summarising (faster, and no transcription errors).

**2. (Optional) Save it to a project.** Switch on *"Save this meeting to a project?"*, give the meeting a title, and pick a project. The result will be filed there once it finishes. You can create a new project from the panel on the left at any time.

**3. Click Summarise.** A progress bar shows the current step while it works. Audio takes longer than text, because it has to be transcribed first:

| Meeting length | Rough wait |
|---|---|
| Text transcript | a few seconds |
| ~10 minutes of audio | under a minute |
| ~30 minutes of audio | two to three minutes |

**4. Read the results.** Three cards appear:

- **Summary**: a two-sentence overview.
- **Key takeaways**: short bullets. Green ones are decisions; the others are notes.
- **Action items**: each task with its owner and deadline shown as small tags. Tick a task off to mark it done.

If the meeting had no real action items (for example, an informational briefing), the list will say so rather than inventing tasks.

## Working with projects

Open a project from the left-hand panel to see every meeting saved under it, laid out as a timeline. Click any entry to reopen the full summary, takeaways, and action items. You can delete individual meetings or whole projects from the same place.

On a phone the panel is hidden by default; tap the ☰ button to open it, and it tucks away again once you pick a project.

## Asking questions about a project

Below the meeting timeline inside any project, you will find the **Ask about this project** panel. Type a question in natural language and press **Ask** (or hit Enter).

Examples of questions that work well:

- *"What did we decide about the supplier contract?"*
- *"Who owns the onboarding task?"*
- *"What were the main risks raised in June?"*
- *"Has the budget been approved?"*

The system searches across all saved meetings in the project, finds the most relevant ones, and gives you a grounded answer. Each answer includes **source chips** showing which meetings it drew from, along with the date. You can use these to go back and read the full meeting record if you want more detail.

A few things to know:

- **It only knows what is in the saved meetings.** If a decision was made in a meeting that was not saved to the project, the system will say so rather than guess.
- **Meetings need an embedding to be searchable.** Any meeting saved while the Gemini API key was configured is searchable immediately. If a meeting shows in the timeline but is not being found by the Ask panel, ask a team member to run the back-fill script (`embed_existing.py`).
- **The answer is a synthesis, not a quote.** The system reads the relevant meetings and writes a concise answer. Check the source chips if you need to verify the exact wording from the original record.

## Things to keep in mind

This is a proof of concept, and it has limits worth knowing before you rely on it:

- **It doesn't know who is speaking.** The transcription turns speech into text but doesn't label speakers. Action-item owners are taken from the words themselves, so if nobody is named ("can you handle that?"), the owner is left blank.
- **It assumes English audio.** The transcription model is tuned for English; other languages will transcribe poorly.
- **Audio quality matters.** Heavy background noise, strong accents, or people talking over each other will lower transcription accuracy, and that carries through to the summary.
- **It only hears the conversation.** Anything shown on a screen or whiteboard isn't captured.
- **Check before you share.** On occasion the model will surface an action item that was only implied, or miss one that was. Treat the output as a strong first draft, not a finished record.
