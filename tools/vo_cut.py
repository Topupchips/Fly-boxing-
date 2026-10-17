"""
Cut the silences out of voiceover / face-cam takes and re-time the words.

    python tools/vo_cut.py plan.json transcripts.json out_dir

plan.json:   {"name": {"src": "clip.mp4", "wav": "clip16k.wav", "start": 0.0, "end": 3.9}, ...}
transcripts: output of the speech-to-text step (word start times per wav)

For each take it finds pauses with a speech-energy detector, keeps every
stretch of speech (plus a little padding so it never sounds clipped), and
writes:
  out_dir/<name>.json   keep intervals (source seconds) + words re-timed to the cut
"""
from __future__ import annotations

import json
import os
import sys
import wave

import numpy as np

MIN_GAP = 0.16       # pauses longer than this get cut
PAD = 0.07           # breathing room kept on each side of speech
HOP = 0.01


def load(p):
    with wave.open(p) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
        return x, w.getframerate()


def speech_mask(x, sr):
    n = int(HOP * sr)
    frames = len(x) // n
    rms = np.sqrt(np.mean(x[: frames * n].reshape(frames, n) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    lo, hi = np.percentile(db, 10), np.percentile(db, 95)
    thr = lo + 0.32 * (hi - lo)
    m = db > thr
    # close tiny holes inside words, drop tiny blips
    k = int(0.06 / HOP)
    m = np.convolve(m.astype(float), np.ones(k), "same") > 0
    return m


def keep_intervals(m, t0, t1):
    out, start = [], None
    for i, v in enumerate(m):
        t = i * HOP
        if v and start is None:
            start = t
        if (not v or i == len(m) - 1) and start is not None:
            out.append([start, t])
            start = None
    out = [[max(t0, a - PAD), min(t1, b + PAD)] for a, b in out if b > t0 and a < t1]
    merged = []
    for a, b in out:
        if merged and a - merged[-1][1] < MIN_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [[round(a, 3), round(b, 3)] for a, b in merged if b - a > 0.08]


def protect_words(keeps, starts, t0, t1, lead=0.06, tail=0.3):
    """Never clip a word: begin pieces just before their first word, give every word
    some room after it, and drop pieces (breaths, clicks) that contain no word."""
    keeps = [list(k) for k in keeps]
    for t in starts:
        inside = next((k for k in keeps if k[0] - 0.3 <= t <= k[1]), None)
        if inside is None:
            keeps.append([max(t0, t - lead), min(t1, t + tail)])
            continue
        inside[0] = min(inside[0], max(t0, t - lead))
        inside[1] = max(inside[1], min(t1, t + tail))
    keeps.sort()
    merged = []
    for a, b in keeps:
        if merged and a - merged[-1][1] < MIN_GAP:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    merged = [k for k in merged if any(k[0] - 0.01 <= t <= k[1] for t in starts)]
    if starts and merged:                       # let the final word ring out
        merged[-1][1] = max(merged[-1][1], min(t1, max(starts) + 0.45))
    return [[round(a, 3), round(b, 3)] for a, b in merged]


def remap(t, keeps):
    acc = 0.0
    for a, b in keeps:
        if t < a:
            return acc
        if t <= b:
            return acc + (t - a)
        acc += b - a
    return acc


def main(plan_path, tr_path, out_dir):
    plan = json.load(open(plan_path))
    tr = json.load(open(tr_path))
    os.makedirs(out_dir, exist_ok=True)
    for name, p in plan.items():
        x, sr = load(p["wav"])
        m = speech_mask(x, sr)
        t0, t1 = p.get("start", 0.0), p.get("end", len(x) / sr)
        keeps = keep_intervals(m, t0, t1)
        words = [w for w in tr[p["wav"]]["words"] if t0 - 0.05 <= w["t"] < t1]
        keeps = protect_words(keeps, [w["t"] for w in words], t0, t1)
        retimed = [{"w": w["w"], "t": round(remap(w["t"], keeps), 3)} for w in words]
        total = sum(b - a for a, b in keeps)
        for i, w in enumerate(retimed):
            w["e"] = retimed[i + 1]["t"] if i + 1 < len(retimed) else round(total, 3)
        json.dump({"src": p["src"], "keeps": keeps, "dur": round(total, 3), "words": retimed},
                  open(os.path.join(out_dir, f"{name}.json"), "w"), indent=1)
        raw = t1 - t0
        print(f"{name:8s} {raw:5.2f}s -> {total:5.2f}s  ({len(keeps)} pieces)  {' '.join(w['w'] for w in retimed)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
