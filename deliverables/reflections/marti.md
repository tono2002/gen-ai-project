# Individual Reflection: Martí

## My contributions

### Frontend
I built the entire frontend as a single HTML file (`src/static/index.html`) with no framework dependencies — plain JavaScript and CSS only. The interface includes a drag-and-drop upload area that accepts audio, video, and text files, a results panel with the summary, color-coded takeaway bubbles (purple for decisions, grey for notes), an action item checklist, and a collapsible sidebar showing the timeline of summarizations saved to each project. The layout is fully responsive and works on iPhone screens, which matters because the intended user is someone in a meeting who might pull up past results on their phone.

The most technically involved part was keeping the UI in sync with the backend without a framework. Transcription can take several minutes for long recordings, so the interface needed to show a loading state that felt live — I used a spinner with a running elapsed timer so the user knows the app is still working rather than frozen. The results then render progressively as the JSON comes back from the API. I also handled all error states explicitly (unsupported file type, API key missing, model error) so the user gets a clear message rather than a silent failure.

### Failure mode analysis
After the evaluation was run, I went through all the cases where the model missed an action item or produced a wrong one and documented the most informative ones in `data/eval/failure_analysis.md`. Each case includes the relevant transcript excerpt, what the model returned, what the gold label was, and a hypothesis for why it failed. The main patterns I identified were: Whisper mishearing proper names (which causes the owner field to be wrong even when the task itself is correct), the model generating phantom action items when a speaker discusses something hypothetically without committing to it, correct zero-output on non-meeting input like a podcast episode (which is actually good behavior and worth documenting), and attribution failures when responsibility is distributed across multiple people or implied by context rather than stated directly.

### Labeled transcripts
I contributed approximately 8 labeled transcripts to `eval_set.jsonl`, deliberately designing them to be difficult: meetings where no one is explicitly assigned a task, meetings where two people share responsibility for one deliverable, and transcripts with deadlines stated informally ("by end of week", "before the holiday") to test whether the model extracts them correctly.

## What I learned

The most important thing I took away from this project is how NLP errors propagate through a pipeline. Whisper's transcription errors don't stay in the transcription layer — they flow directly into the LLM output. If Whisper mishears a person's name, the action item extracted by Claude will have the wrong owner, even if the model does everything else correctly. This kind of cascading failure is hard to see from an aggregate F1 score alone, and it made me understand why failure mode analysis is treated as a required component of the assignment rather than just a nice extra. I also learned what information extraction actually means as an NLP task — it sits between pure classification (assign a label to a whole input) and generation (produce free text), and the challenge is that the output structure is constrained but the surface form is open, which is exactly why a simple regex or rule-based approach would fail on real, messy meeting transcripts.
