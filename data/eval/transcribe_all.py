"""Batch-transcribe the meeting recordings into text files for the eval set.

Uses the same Whisper settings as the app (base.en, greedy, multithread).
Resumable: skips files already transcribed. Output → data/eval/transcripts/.

Run:  .venv/bin/python data/eval/transcribe_all.py
"""

import os
import re
import time
from pathlib import Path

from faster_whisper import WhisperModel

AUDIO_DIR = Path("/Users/tonoabad/Desktop/NLP Assignment/audios meetings ")  # note trailing space
OUT_DIR = Path(__file__).resolve().parent / "transcripts"
OUT_DIR.mkdir(exist_ok=True)


def natural_key(p: Path) -> int:
    nums = re.findall(r"\d+", p.name)
    return int(nums[0]) if nums else 0


def main() -> None:
    model = WhisperModel("base.en", compute_type="int8", cpu_threads=os.cpu_count() or 4)
    files = sorted(AUDIO_DIR.glob("*.wav"), key=natural_key)
    print(f"{len(files)} recordings to transcribe", flush=True)

    for i, f in enumerate(files, 1):
        out = OUT_DIR / (f.stem.replace(" ", "_") + ".txt")
        if out.exists() and out.stat().st_size > 0:
            print(f"[{i}/{len(files)}] skip {f.name} (already done)", flush=True)
            continue
        t0 = time.time()
        segments, _info = model.transcribe(
            str(f),
            beam_size=1,
            vad_filter=True,
            condition_on_previous_text=False,
            language="en",
        )
        text = " ".join(s.text.strip() for s in segments)
        out.write_text(text, encoding="utf-8")
        print(f"[{i}/{len(files)}] {f.name} -> {out.name} "
              f"({len(text)} chars, {time.time() - t0:.0f}s)", flush=True)

    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
