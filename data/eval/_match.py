"""Map our transcripts (audio_N) to AMI ES codes by exact audio duration.

The original recordings were renamed audio_1..30; the ES-named copies live in
../audios. Each meeting has a unique length, so duration is a reliable key.
audio_N durations are from the original ffprobe pass (recorded below); ES
durations are read live from the ES folder.
"""
import json
import os
import subprocess

ES_DIR = "/Users/tonoabad/Desktop/NLP Assignment/audios"
HERE = os.path.dirname(os.path.abspath(__file__))

# audio_N -> duration in whole seconds (from the original recordings)
AUDIO_DUR = {
    1: 1272, 2: 2279, 3: 2423, 4: 2624, 5: 1139, 6: 2109, 7: 2256, 8: 2351,
    9: 1049, 10: 2345, 11: 2334, 12: 2222, 13: 2313, 14: 2295, 15: 1731,
    16: 1284, 17: 2183, 18: 2181, 19: 1967, 20: 1206, 21: 1685, 22: 2377,
    23: 1043, 24: 1249, 25: 2231, 26: 2102, 27: 2625, 28: 1402, 29: 1435, 30: 1956,
}


def es_durations():
    out = {}
    for f in os.listdir(ES_DIR):
        if f.lower().endswith(".wav") and not f.startswith("."):
            d = subprocess.check_output(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "csv=p=0", os.path.join(ES_DIR, f)], text=True).strip()
            out[f.split(".")[0]] = int(float(d))
    return out


es = es_durations()
es_by_dur = {dur: code for code, dur in es.items()}
mapping, unmatched = {}, []
for n, dur in AUDIO_DUR.items():
    if dur in es_by_dur:
        mapping[f"audio_{n}"] = es_by_dur[dur]
    else:
        unmatched.append((n, dur))

print(f"matched {len(mapping)}/30")
if unmatched:
    print("UNMATCHED:", unmatched)
for k in sorted(mapping, key=lambda x: int(x.split("_")[1])):
    print(f"  {k} -> {mapping[k]}")
json.dump(mapping, open(os.path.join(HERE, "id_mapping.json"), "w"), indent=2)
print("saved id_mapping.json")
