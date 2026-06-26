# Custom Evaluation

We evaluate **SummarAI's summarization** against the **AMI Meeting Corpus** human reference summaries, no manual labeling, automatic scoring.

## Why this design
- Our 30 demo recordings are AMI scenario meetings (`ES2002a` … `ES2009c`).
- AMI ships **human-written abstractive summaries** with `ABSTRACT`, `DECISIONS`, and `ACTIONS` sections, our ground truth.
- We score the **summarization stage** (text → summary) with **ROUGE**. Transcription is evaluated separately (the Whisper speed finding); feeding clean reference summaries isolates summary quality from ASR noise.

## Metrics (see `results.json`)
1. **ROUGE-1 / 2 / L**: our full output (summary + takeaways + action items) vs the AMI human reference (abstract + decisions + actions).
2. **Action-item coverage (recall)**: fraction of AMI's human ACTION sentences a predicted action item covers (ROUGE-L F ≥ 0.30).

## Files
| File | What |
|---|---|
| `transcribe_all.py` | Transcribed the 30 recordings → `transcripts/` (Whisper) |
| `transcripts/` | Our Whisper transcripts (input to the summarizer) |
| `_match.py` / `id_mapping.json` | Maps `audio_N` → AMI `ES` code by audio duration |
| `evaluate_rouge.py` | Runs the real summarizer on each transcript, scores ROUGE + action coverage |
| `results.json` | The numbers (for the report) |

## Reproduce
```bash
# 1. Get AMI human annotations (not committed: 23 MB, gitignored)
mkdir -p data/eval/ami_raw && cd data/eval/ami_raw
curl -sO https://groups.inf.ed.ac.uk/ami/AMICorpusAnnotations/ami_public_manual_1.6.2.zip
unzip -q ami_public_manual_1.6.2.zip && cd -

# 2. Run the evaluation
.venv/bin/python data/eval/evaluate_rouge.py
```

## Source / honesty note
Recordings are from the **AMI Meeting Corpus** (Carletta et al.). Reference summaries are AMI's **human** annotations. We defined the evaluation design, the metric mapping (structured output → prose reference), and the analysis ourselves.
