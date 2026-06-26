# Individual Reflection: Jo

## My contributions

### Supabase database layer
I designed and set up the entire persistence layer for the app. I wrote `supabase_schema.sql`, which defines two tables: `projects` (for grouping meetings) and `summarizations` (for storing the full structured output of each meeting). A key design decision was storing `key_takeaways` and `action_items` as JSONB columns rather than as flat text. This means the structured data — the individual takeaway bullets, their type tags, the task/owner/deadline fields for each action item — is preserved as queryable structured data in the database rather than being serialized into a string that would have to be parsed back out every time. The schema also includes a foreign key from `summarizations` to `projects` with cascade delete, so deleting a project cleans up all its associated meetings automatically.

I also configured row-level security (RLS) policies so the app works without requiring users to create accounts. The POC uses Supabase's anonymous key with permissive read/write policies, which is appropriate for a proof of concept and documented as a known limitation for any future production deployment.

Beyond the schema, I verified that the backend endpoints Antonio built matched what Supabase expected — checking that the REST calls for creating, listing, and deleting projects and summarizations all worked correctly against the live database.

### Field review
I researched the meeting-AI and automatic transcription industry and wrote the field review section of the technical report. I looked at the major existing tools — Otter.ai, Fireflies.ai, Zoom AI Companion, Fathom, and tl;dv — and analysed what they offer and what they consistently lack. Most of them produce a single undifferentiated list of bullet points; none separate decisions from notes or distinguish between a commitment someone made and a fact someone mentioned. Existing tools also produce output in the same language as the meeting, which is a real barrier for international teams. I cited sources from ACL Anthology, arXiv, and industry reports to ground the review in published research rather than just product marketing.

### Labeled transcripts
I contributed approximately 8 labeled transcripts to `eval_set.jsonl`, including excerpts from public city council meeting minutes. These are useful because they are real, verbatim, and messy — speakers talk over each other, use incomplete sentences, and make references that require context to understand — which is exactly the kind of input that stresses the model.

## What I learned

Doing the field review made me understand the difference between what NLP research has achieved and what is actually deployed in real products. There is a large body of work on meeting summarization, action item extraction, and dialogue act classification going back years, yet the mainstream commercial tools still produce simple undifferentiated bullet lists. The research-to-product gap is real and much wider than I expected. I also developed a clearer understanding of the different NLP subtasks our system combines: automatic speech recognition (Whisper), abstractive summarization (the two-sentence summary), and named entity and relation extraction (pulling out the task, owner, and deadline as structured fields). Treating them as one task called "summarization" undersells the complexity — each one has its own failure modes, benchmarks, and literature, and the overall system quality is limited by whichever stage performs worst on a given input.
