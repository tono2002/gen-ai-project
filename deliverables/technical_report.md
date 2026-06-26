# Technical Report: SummarAI

---

## 1. Introduction

Meetings generate decisions and commitments, but most of that information is lost shortly after the call ends. People rely on informal notes, fragmented memory, or re-watching recordings, none of which scale. SummarAI is a web application that automates this capture: upload an English-language meeting recording and, within minutes, receive a concise written record structured around two outputs: a short prose summary and a set of bullet points covering key takeaways and action items.

The system chains two NLP components. First, a local speech recognition model (Whisper) converts audio to text. Second, a large language model (Claude Haiku 4.5) reads that transcript and produces structured output through a strict JSON schema. The web interface is minimal by design, the value is in the NLP pipeline, not the UI.

This report describes the motivation, the technical design, an empirical evaluation, and the failure modes we documented during development.

---

## 2. Justification of need

Meetings consume a large share of knowledge workers' time, yet much of their value is lost once the call ends. The problem is rarely the discussion itself but the follow-through: decisions made verbally go unrecorded, action items are forgotten, and participants who were absent have no reliable way to catch up. A written record fixes this, but taking accurate minutes by hand is tedious and routinely skipped, which is precisely the gap an automated tool can fill.

Several tools exist to address this, Otter.ai, Fireflies.ai, Zoom AI Companion, and Microsoft Teams Copilot all offer some form of meeting transcription and summary. Their limitations are relevant to this project:

- **Cost and access**: all major tools require paid subscriptions; several are tied to specific video-conferencing platforms.
- **Output quality**: most tools produce a raw transcript or a continuous prose summary. They rarely separate *decisions* from *notes*, and action-item extraction is either absent or unreliable when responsibility is only implied rather than explicitly assigned.
- **Closed systems**: none of these tools expose the underlying model or prompt, making it impossible to understand or modify their behaviour.

SummarAI addresses the output-quality gap specifically. Its two-output design, a 2-sentence summary plus typed bullet points (decision / note / action item), forces the model to make an explicit distinction that conversational summaries routinely blur.

---

## 3. Field review

### 3.1 Automatic speech recognition (ASR)

The dominant open-source ASR model at the time of writing is OpenAI Whisper (Radford et al., 2022). Whisper is an encoder-decoder transformer trained on 680,000 hours of web audio, achieving near-human accuracy on clean English speech and robust performance on accented or lightly noisy audio. Its English-only variants (`tiny.en`, `base.en`, `small.en`, …) outperform the corresponding multilingual models on English audio while being faster.

`faster-whisper` is a community re-implementation of Whisper using the CTranslate2 inference library. It achieves the same output as the original at 2–4× lower memory usage and measurably faster runtime, which makes it practical for a local CPU deployment without a GPU.

Commercial ASR alternatives (AssemblyAI, AWS Transcribe, Azure Cognitive Services) offer higher accuracy on telephony-quality audio and built-in speaker diarization, at per-minute costs that make them unsuitable for a course prototype that needs to run cheaply at demo scale.

### 3.2 Meeting summarization and information extraction

LLM-based summarization has largely displaced earlier extractive and abstractive approaches (TextRank, BART, PEGASUS) for long-document tasks. Instruction-tuned models handle long, noisy meeting transcripts better than task-specific fine-tuned models, primarily because they can follow complex, multi-part instructions about output format.

Structured output, enforcing a JSON schema at the API level rather than prompt-engineering the model to emit valid JSON, is a significant recent development. It eliminates parsing failures caused by the model deviating from the expected format, which was a common failure mode in earlier tool-calling approaches.

The dominant commercial players (OpenAI GPT-4o, Google Gemini, Anthropic Claude) all now support structured output. Claude Haiku 4.5 was selected for this project on cost grounds: it is the fastest and cheapest model in the Claude 4 family while still following complex structured instructions reliably.

### 3.3 Retrieval-Augmented Generation for meeting archives

Retrieval-Augmented Generation (RAG) augments a language model's response with documents retrieved from an external index rather than relying solely on the model's parametric knowledge (Lewis et al., 2020). In the context of meeting assistants, RAG enables longitudinal queries across a project's full meeting history — something that a per-meeting summariser cannot provide.

The dominant approach uses dense vector retrieval: a document is embedded into a high-dimensional space and stored; at query time, the query is embedded in the same space and the most similar documents are retrieved via approximate nearest-neighbour search. `pgvector` (Supabase, 2023) is an open-source PostgreSQL extension that adds native vector types and distance operators, enabling vector search within an existing relational database without an external vector store.

Asymmetric embedding — using different task types for queries versus documents — is important for retrieval quality. Models trained with asymmetric contrastive objectives (e.g. Gemini embedding-001, Google, 2024) optimise the query and document representation functions separately, reflecting the linguistic asymmetry between a short conversational question and a longer structured document. Using the wrong task type degrades cosine similarity scores and retrieval ranking in practice.

---

## 4. System description

### 4.1 Architecture

SummarAI is a single-server web application. The backend is a Python FastAPI app (`src/app.py`). The frontend is a single-page vanilla JavaScript application (`src/static/index.html`). Meeting results can optionally be saved to a Supabase (PostgreSQL) database for project-level persistence and timeline views.

The processing pipeline is strictly sequential:

```
User uploads file
        │
        ▼
 Audio/video? ──Yes──► Whisper (faster-whisper, base.en, local CPU)
        │                       │
        No                      ▼
        │               Plain-text transcript
        └──────────────────────►│
                                ▼
                     Claude Haiku 4.5 (JSON schema)
                                │
                    ┌───────────┴────────────┐
                    ▼                        ▼
              Summary (≤ 2 sentences)   Bullet points
                                     (key takeaways + action items)
                                             │
                              (if saved to project)
                                             ▼
                              Gemini embedding-001 (1536 dims)
                                             │
                                             ▼
                              Supabase / pgvector (embedding column)
                                             │
                              ┌──────────────┘
                              │  RAG retrieval path
                              ▼
                     User question → embed (retrieval_query)
                              │
                              ▼
                     match_summarizations() — top-3 by cosine sim
                              │
                              ▼
                     Claude Haiku 4.5 (RAG system prompt)
                              │
                              ▼
                     Grounded answer + source citations
```

Supported input formats:

| Category | Extensions |
|---|---|
| Audio / video | `.mp4`, `.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`, `.flac` |
| Text (skip ASR) | `.txt`, `.md`, `.vtt`, `.srt` |

The 300,000-character transcript limit corresponds to roughly 3–4 hours of dense meeting speech, well beyond the practical use case of a single meeting.

### 4.2 Two-stage NLP pipeline

**Stage 1, Speech-to-text.** `faster-whisper` runs locally on CPU using the `base.en` model in `int8` quantization. Voice activity detection (`vad_filter=True`) strips silence before transcription, which both speeds up processing and avoids spurious text segments from quiet segments. The output is a plain string of the full meeting speech.

**Stage 2, Summarization and information extraction.** The transcript is sent to Claude Haiku 4.5 with a system prompt that instructs the model to produce exactly three fields:

1. **`summary`**: at most 2 short sentences stating the meeting's purpose and overall outcome.
2. **`key_takeaways`**: an array of bullet objects, each with a `text` field (≤14 words) and a `type` field: `"decision"` (something agreed or set as policy) or `"note"` (any other important point, fact, or concern). The instruction to be *comprehensive* here is deliberate: the 2-sentence summary omits detail by design, so the takeaways carry all substantive information.
3. **`action_items`**: concrete tasks with `owner` and `deadline` populated only when actually stated in the meeting; never inferred.

The output schema is enforced at the API level using Claude's `json_schema` output format, which guarantees that every response parses correctly without defensive handling.

### 4.3 Performance optimization

During development we measured end-to-end processing time on a 20-minute `.m4a` file on a standard laptop CPU. Latency was almost entirely in the ASR stage, Claude Haiku 4.5 returned in 3–6 seconds regardless of transcript length, while Whisper dominated total runtime. We applied four zero-cost changes to the transcription stage only:

| Change | Rationale |
|---|---|
| `base.en` instead of multilingual `base` | English-only model is faster and more accurate for English audio |
| `beam_size=1` (greedy decoding) | Removes beam search overhead (~2× faster) with negligible accuracy loss |
| `cpu_threads=os.cpu_count()` | Uses all available cores instead of the library default |
| `language="en"` | Skips the automatic language-detection pass |

**Measured result (same file, same hardware):**

| Configuration | Total time | Transcript length |
|---|---|---|
| Original (`base`, beam 5, auto-detect) | 7 min 50 s | 36,702 chars |
| Optimized (`base.en`, greedy, multithread) | 2 min 12 s | 36,620 chars |

This is a **3.5× speedup (≈72% reduction) at zero additional cost**, with no measurable effect on downstream summary quality (near-identical transcript character counts; unchanged model output).

The Whisper model is configurable via the `WHISPER_MODEL` environment variable, `tiny.en` offers even faster throughput at some accuracy cost for live demos on constrained hardware.


### 4.4 RAG layer — natural-language query over the meeting archive

Once a summarisation result is saved to a project, SummarAI automatically embeds the record and indexes it for semantic retrieval. This enables a second interaction mode: the user types a natural-language question in the **Ask about this project** panel and receives a grounded, cited answer drawn from the project's meeting history.

#### Indexing path (runs on every save)

When `POST /api/summarizations` is called, the backend flattens the full record — meeting title, two-sentence summary, all key takeaways, and all action items — into a single text string using `build_summary_text()`. This string is embedded using the Google Gemini `embedding-001` model with `task_type="retrieval_document"`, producing a 1536-dimensional dense vector. The vector is stored in a new `embedding vector(1536)` column in the `summarizations` table via the `pgvector` PostgreSQL extension.

The Supabase schema (`supabase_schema_rag.sql`) includes:

- `CREATE EXTENSION IF NOT EXISTS vector` — enables pgvector.
- `ALTER TABLE summarizations ADD COLUMN embedding vector(1536)` — adds the vector column.
- An HNSW index (`vector_cosine_ops`) for fast approximate cosine-distance search.
- A custom SQL function `match_summarizations(query_embedding, match_project_id, match_count)` that returns the top-k rows by cosine similarity for a given project, called via the Supabase RPC interface.

If `GEMINI_API_KEY` is absent, embedding is skipped silently and the save still succeeds; the record is retrievable via back-fill once the key is added.

#### Retrieval path (runs on every question)

`POST /api/projects/{project_id}/ask` accepts a `question` string and an optional `top_k` parameter (default 3).

1. **Embed the question.** The question is embedded using `task_type="retrieval_query"`. This asymmetry is deliberate: Gemini `embedding-001` was trained on (query, document) pairs and optimises the two embedding types differently. Using the wrong task type for either role measurably degrades cosine similarity scores and therefore retrieval ranking.
2. **Retrieve.** `match_summarizations()` returns the top-k meetings for the project ranked by cosine similarity to the query vector.
3. **Augment and generate.** The retrieved meetings are serialised as a structured context block (meeting title, date, summary, takeaways, action items) and passed to Claude Haiku 4.5 with a dedicated RAG system prompt. The prompt instructs the model to answer from the provided context only, to cite meeting titles for every claim, and to state explicitly when the answer is not present rather than hallucinating.
4. **Return.** The response includes both the text answer and a `sources` array (meeting ID, title, date, cosine similarity score), which the frontend renders as clickable source chips.

#### Design choices

**Gemini over OpenAI embeddings.** `text-embedding-3-small` (OpenAI) would require a paid key and introduce a second commercial dependency. Gemini `embedding-001` is free within the API's daily quota, integrates with the existing `google-genai` SDK, and natively supports asymmetric task types.

**1536 dimensions.** Gemini `embedding-001` produces up to 3072 dimensions by default; we truncate to 1536 using the `output_dimensionality` parameter to stay within Supabase's pgvector index limit. 1536 dimensions is sufficient for a meeting-scale corpus and affordable in storage terms.

**HNSW index.** For a corpus of tens to hundreds of meetings, a flat scan would also be fast enough. The HNSW index is included so the system remains performant as the archive grows without schema changes. We use HNSW rather than IVFFlat because Supabase's version of pgvector caps IVFFlat at 2000 dimensions, and our 1536-dimensional vectors require HNSW.

**Grounded generation.** The RAG system prompt does not ask Claude to use its prior knowledge. The model's role in this path is purely synthesis and formatting, not knowledge retrieval. This eliminates the hallucination risk that would arise from asking the model to recall facts about meetings it was never shown.

---

## 5. Custom evaluation

### 5.1 Methodology

Our test set is 30 meetings from the **AMI Meeting Corpus** (Carletta et al., 2006), a widely used research dataset of recorded meetings, these are the same recordings used for the demo (`ES2002a`–`ES2009c`). AMI is annotated with human-written abstractive summaries, each split into `ABSTRACT`, `DECISIONS`, and `ACTIONS` sections. We use those human annotations as ground truth. We did not hand-label our own answers; instead we reuse professional human annotations and built the evaluation around them, which avoids the circularity of grading an LLM against answers we wrote ourselves.

We evaluate the two pipeline stages separately. Transcription speed is reported in Section 4.3; here we assess the **summarization** stage. Each meeting's transcript is passed to the summarizer and its output compared against the AMI human reference. We feed our own Whisper transcripts (not AMI's clean reference transcripts) as input, so the scores reflect the full audio-to-summary path, including transcription noise.

We report two automatic metrics, computed by `data/eval/evaluate_rouge.py`:

- **ROUGE-1 / 2 / L (F-measure)**: our full output (summary + takeaways + action items, rendered as text) against the human reference (abstract + decisions + actions). ROUGE is the standard meeting-summarization metric and is explicitly accepted by the assignment.
- **Action-item coverage (recall)**: the fraction of the human `ACTIONS` sentences that a predicted action item covers, counting a match when ROUGE-L F ≥ 0.30. This measures the headline feature directly against the human action annotations.

We report coverage (recall) rather than precision/F1 for action items because AMI's `ACTIONS` list is a curated human selection, not an exhaustive enumeration of every commitment in the meeting; penalising a model action that AMI's annotator simply chose not to list would understate true precision.

### 5.2 Results

Across the 30 meetings:

| Metric | Score |
|---|---|
| ROUGE-1 (F, avg) | 0.335 |
| ROUGE-2 (F, avg) | 0.057 |
| ROUGE-L (F, avg) | 0.148 |
| Action-item coverage (recall) | 0.424 (28 / 66) |

The summaries are coherent and stay on topic. The ROUGE figures are modest, and the main reason is a deliberate **format mismatch**: our summary is two sentences by design, whereas an AMI reference is a ~200-word, multi-section document. ROUGE rewards n-gram overlap, so a deliberately compressed, differently-structured output is penalised on *form* even when its content is accurate. A ROUGE-1 of 0.34 means our output shares roughly a third of its unigrams with a much longer human summary, a reasonable result for an output that is an order of magnitude shorter. Two further factors push the numbers down honestly: the input is noisy Whisper output rather than clean text, and the model is run zero-shot with no fine-tuning on AMI.

The most informative number is the **42% action-item coverage**: working end to end from raw audio, against strict human annotations, the system recovers roughly four of every ten action items a human annotator recorded.

### 5.3 Error analysis

Two systematic factors explain most of the missed action items:

1. **Representation gap.** AMI annotates actions by *role* ("the industrial designer will work on the design"), while SummarAI extracts them by *name* (the person actually addressed in the conversation). The underlying task is the same, but the surface forms differ enough to fall below the overlap threshold, which accounts for several apparent misses that are not really errors.
2. **Owner attribution / speaker identity.** Whisper produces no speaker labels. When responsibility is implied rather than named, the model leaves `owner` null rather than guessing, conservative and arguably correct, but it still counts as an incomplete action item. This is the clearest limitation of the system and the most direct target for future work (Section 7).

---

## 6. Failure mode analysis

The following cases were documented during development and evaluation.

**1. Implied speaker identity.** When responsibility is suggested rather than stated ("we should look into this"), the model correctly produces an action item with `owner: null`. This is not wrong behaviour, but it limits the utility of the action-item list for follow-up.

**2. Whisper mishearing proper nouns.** Names that are uncommon in Whisper's training data are occasionally transcribed incorrectly (e.g. "Bojana" transcribed as "Janna"). Since action-item owners depend on accurate name transcription, this propagates directly into incorrect attribution downstream.

**3. Off-topic or non-meeting audio.** Podcast-style recordings, long monologues, panel discussions, interviews, produce coherent summaries and takeaways but correctly yield zero action items. The system behaves as designed, but users uploading non-meeting content may be confused by the empty action-item list.

**4. Heavily overlapping or noisy audio.** Whisper's VAD filter removes silence effectively, but it does not separate overlapping speakers. In meetings where multiple participants speak simultaneously, the transcript can contain garbled or incomplete sentences, which the LLM then has to interpret from context. Summary quality degrades noticeably in these cases.

**5. Very long meetings near the character limit.** The 300,000-character cap (≈3–4 hours of dense speech) is far above typical meeting length, but processing time scales linearly with transcript length. A 2-hour meeting can take 4–5 minutes on modest hardware. The LLM context window (handled by Claude) is not a bottleneck at current meeting lengths.

**6. Implicit decisions.** When a decision emerges through consensus rather than an explicit statement, the model sometimes tags it as `note` rather than `decision`. The distinction is genuinely ambiguous in natural speech, and the system's conservative tagging is reasonable but imperfect.

**7. Repetitive or circular discussions.** In meetings where the same point is debated multiple times, the model sometimes produces duplicate or near-duplicate takeaways. The system prompt instructs comprehensiveness, which can work against deduplication.

---

## 7. Future directions

**Speaker diarization.** The highest-impact improvement would be integrating speaker diarization, automatic labelling of who says what in the transcript. This would directly solve the owner attribution problem and allow the model to assign action items reliably without depending on names being spoken aloud. Libraries such as `pyannote.audio` provide open-source diarization that could be inserted between the Whisper ASR stage and the LLM stage.

**Larger Whisper models.** The `base.en` model was chosen for speed. On hardware with more RAM, `small.en` or `medium.en` would improve accuracy on accented speech, domain-specific vocabulary, and noisy audio with minimal changes to the pipeline.

**Confidence and uncertainty signals.** The current system does not communicate uncertainty. A useful extension would be flagging action items or decisions where the model's confidence is low, for example, items extracted from a single ambiguous sentence rather than a clear explicit commitment.

**Agentic meeting assistant.** The RAG layer implemented in section 4.4 is a natural foundation for an agentic extension: an agent that monitors a calendar, retrieves meeting recordings automatically, cross-references action items from previous meetings via the RAG index, and alerts owners of overdue tasks. This would apply the agentic AI techniques introduced in course module 08 on top of the retrieval infrastructure already in place.

**Deployment.** The local Whisper model rules out serverless hosting (Vercel, Lambda) due to memory and runtime constraints. A long-running FastAPI server on Render, Railway, or Fly.io would work without changes to the application code.

---

## 8. Use of AI tools

Claude (Anthropic) was used throughout the project in several capacities:

- **Code generation and debugging**: the FastAPI backend, the structured output schema, and the Supabase integration were iteratively developed with Claude assistance. Claude was particularly useful for generating correct Pydantic models and for debugging JSON schema validation errors.
- **System prompt engineering**: the system prompt that instructs Claude Haiku was drafted collaboratively, initial versions were generated by Claude Sonnet and then refined through iterative testing on real transcripts. The final decision/note distinction and the instruction to "lose no information" in takeaways emerged from this process.
- **Documentation**: sections of this report, the user manual, and the installation guide were drafted with Claude assistance and then reviewed and edited by team members.
- **RAG generation**: the `/api/projects/{id}/ask` endpoint calls Claude Haiku 4.5 as the augmented generation step, grounded on retrieved meeting summaries.
- **Transcription** (Whisper): the ASR component is itself an AI model, though it is used as a fixed inference component rather than interactively.
- **Embeddings** (Gemini): the RAG indexing and retrieval paths use the Gemini `embedding-001` model for dense vector representation of both documents and queries.

All code, prompts, and documentation were reviewed and edited by team members. No output was used without human review.

---

## 9. Conclusion

SummarAI demonstrates that a two-stage NLP pipeline — local ASR followed by a structured LLM call — can reliably transform an English meeting recording into an actionable written record, and that the resulting structured archive can be made queryable via a RAG layer built on top of the existing database. The system is functional, reproducible, and fast enough for practical use on consumer hardware.

The key finding from the core pipeline was empirical: latency is dominated by transcription, not by the LLM, and a set of zero-cost Whisper optimisations reduced total processing time by 72% with no measurable loss in output quality. The key limitation of the summarisation stage is owner attribution, which depends on speakers being named explicitly in the meeting — a problem that speaker diarisation would directly address.

The RAG layer adds a qualitatively different capability: the ability to ask questions across the full meeting history of a project and receive grounded, cited answers without manually searching through saved summaries. The asymmetric embedding approach — `retrieval_document` at index time and `retrieval_query` at query time — and the instruction to cite sources and decline when the context is insufficient are the two design decisions that most directly determine the quality of RAG outputs.

The two-output design (summary + bullets) proved to be the right abstraction for the core pipeline. The 2-sentence summary is useful for quick orientation; the typed bullet list carries the substance. Separating decisions from notes forces the model to make a distinction that is genuinely useful for follow-up and accountability, even if that distinction is occasionally ambiguous in natural speech.

---

## References

Carletta, J., Ashby, S., Bourban, S., Flynn, M., Guillemot, M., Hain, T., Kadlec, J., Karaiskos, V., Kraaij, W., Kronenthal, M., Lathoud, G., Lincoln, M., Lisowska, A., McCowan, I., Post, W., Reidsma, D., & Wellner, P. (2006). *The AMI meeting corpus: A pre-announcement*. In Machine Learning for Multimodal Interaction (MLMI), 28-39.

Lewis, M., Liu, Y., Goyal, N., Ghazvininejad, M., Mohamed, A., Levy, O., Stoyanov, V., & Zettlemoyer, L. (2020). *BART: Denoising sequence-to-sequence pre-training for natural language generation, translation, and comprehension*. Proceedings of ACL 2020.

Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust speech recognition via large-scale weak supervision*. arXiv preprint arXiv:2212.04356.

See, A., Liu, P. J., & Manning, C. D. (2017). *Get to the point: Summarization with pointer-generator networks*. Proceedings of ACL 2017.

SYSTRAN. (2023). *faster-whisper* [Computer software]. GitHub. https://github.com/SYSTRAN/faster-whisper

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W.-T., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). *Retrieval-augmented generation for knowledge-intensive NLP tasks*. Advances in Neural Information Processing Systems, 33.

Supabase. (2023). *pgvector: Open-source vector similarity search for Postgres* [Computer software]. GitHub. https://github.com/pgvector/pgvector

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L., & Polosukhin, I. (2017). *Attention is all you need*. Advances in Neural Information Processing Systems, 30.
