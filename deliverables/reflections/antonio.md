# Individual Reflection: Antonio

## My contributions

### Whisper optimization
I profiled the full pipeline to find where time was being spent. The assumption going in was that the LLM call would be the slow part — in practice, Claude Haiku returns in a few seconds, while Whisper on a 20-minute audio file was taking close to 8 minutes with the default configuration. I implemented four optimizations, all of them zero-cost in the sense that they required no extra hardware or paid services:

1. Switched from the multilingual `base` model to `base.en` — the English-only variant is faster and more accurate for English audio because it does not reserve capacity for other languages.
2. Set `beam_size=1` (greedy decoding) instead of the library default of 5. Beam search improves accuracy by exploring multiple hypotheses, but for our task the downstream summary was indistinguishable between beam=1 and beam=5, so we were paying a ~2× latency cost for no benefit.
3. Set `cpu_threads` to use all available cores, so transcription runs fully parallel.
4. Passed `language="en"` explicitly to skip the automatic language detection pass, which runs a short audio segment through the model before transcription even starts.

I measured the result on the same 20-minute `.m4a` file on the same hardware before and after: transcription + analysis went from 7 minutes 50 seconds down to 2 minutes 12 seconds — a ~3.5× speedup with near-identical transcript quality (character counts differed by under 100, and the downstream summaries were unchanged). This is documented in the technical report with the actual numbers.

### Evaluation script
I built `data/eval/evaluate.py`. The script loads all transcripts from `eval_set.jsonl`, calls the same `analyze()` function used in production (not a mock), compares the predicted action items against the gold labels using the matching rule the team agreed on, and outputs per-meeting results plus micro-averaged precision, recall, and F1 to `data/eval/results.json`. The matching logic uses normalized text similarity with a configurable threshold (0.6 by default) and greedy one-to-one assignment between predicted and gold items, so each gold item can only be matched once.

### Installation guide
I wrote `docs/installation_guide.md`, covering Python version requirements, virtual environment setup, ffmpeg installation on macOS, Linux, and Windows, `.env` configuration, and the exact commands to run the server. The goal was that someone cloning the repo for the first time could follow it without asking anyone for help.

### Labeled transcripts
I contributed approximately 8 labeled transcripts to `eval_set.jsonl`, including sprint planning sessions and one deliberately ambiguous case where responsibility is implied but no specific person is named — a scenario that stresses the owner attribution part of the model.

## What I learned

Working on this project gave me a much more concrete understanding of how ASR models like Whisper actually work under the hood — specifically the trade-off between multilingual and language-specific models, and what beam search is doing in a seq2seq decoder. Before this I understood beam search as a concept from the course; seeing it actually cost us 3.5× in latency with no measurable quality benefit made the accuracy-efficiency trade-off feel real rather than theoretical. I also developed a much clearer sense of what precision and recall mean as evaluation metrics for an information extraction task. We could have just measured accuracy, but a single accuracy number would have hidden the difference between the model missing real action items (recall failures) and the model inventing ones that weren't there (precision failures) — which are two completely different failure modes with different implications for whether you should trust the system.
