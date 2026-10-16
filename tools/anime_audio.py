"""
Original soundtrack and sound effects for the Fly Boxing reel, synthesized
from scratch with numpy/scipy (no samples, no existing music).

build_soundtrack(total, events, marks, ducks, rng) returns a mono float array:
  * a 140 bpm fight track in E minor (drums, distorted saw bass riff,
    power-chord stabs, square-wave lead) with sections switched by `marks`:
    "full", "muffled" (comedy lowpass), "off" (silence for a riser or the KO)
    and "outro"
  * `ducks`: (start, end) spans where the music dips under a hit-stop
  * `events`: (time, kind[, duration]) sound effects placed on the frame
"""
from __future__ import annotations

import numpy as np
from scipy import signal

SR = 44100
BPM = 140
STEP = 60 / BPM / 4                       # one 16th note


def _t(dur):
    return np.arange(int(dur * SR)) / SR


def env(n, attack=0.002, decay=0.15):
    t = np.arange(n) / SR
    return np.minimum(1, t / max(attack, 1e-6)) * np.exp(-t / decay)


def filt(x, kind, f, order=2):
    sos = signal.butter(order, f, btype=kind, fs=SR, output="sos")
    return signal.sosfilt(sos, x)


def reverb(x, rng, dur=1.4, mix=0.35, tone=5000):
    n = int(dur * SR)
    ir = rng.normal(0, 1, n) * np.exp(-np.arange(n) / SR / (dur / 5))
    ir = filt(ir, "low", tone)
    ir /= np.sqrt((ir ** 2).sum()) + 1e-9
    wet = signal.fftconvolve(np.concatenate([x, np.zeros(n)]), ir)[: len(x) + n]
    dry = np.concatenate([x, np.zeros(n)])
    return dry + wet * mix


def saw(f, t):
    return 2 * ((f * t) % 1.0) - 1


def square(f, t):
    return np.sign(np.sin(2 * np.pi * f * t))


def mix_in(buf, clip, start):
    start = int(start)
    if start >= len(buf) or start + len(clip) <= 0:
        return
    if start < 0:
        clip, start = clip[-start:], 0
    end = min(len(buf), start + len(clip))
    buf[start:end] += clip[: end - start]


def norm(x, peak=1.0):
    return x / (np.abs(x).max() + 1e-9) * peak


# ---------------------------------------------------------------- effects
def sfx_punch(rng):
    t = _t(0.3)
    thump = np.sin(2 * np.pi * (110 * t - 120 * t * t)) * env(len(t), 0.001, 0.07)
    crack = filt(rng.normal(0, 1, len(t)), "high", 1500) * env(len(t), 0.0005, 0.015)
    return reverb(0.9 * thump + 0.6 * crack, rng, 0.5, 0.15)


def sfx_impact(rng, big=False):
    """Hit-stop boom: sub drop + crack + noise burst, with a long tail."""
    dur = 2.2 if big else 1.2
    t = _t(dur)
    f0, f1 = (160, 28) if big else (140, 38)
    phase = 2 * np.pi * np.cumsum(f1 + (f0 - f1) * np.exp(-t / 0.12)) / SR
    sub = np.sin(phase) * env(len(t), 0.001, 0.45 if big else 0.28)
    crack = filt(rng.normal(0, 1, len(t)), "high", 1200) * env(len(t), 0.0005, 0.03)
    body = filt(rng.normal(0, 1, len(t)), "band", [200, 2500]) * env(len(t), 0.001, 0.12)
    x = 1.1 * sub + 0.7 * crack + 0.5 * body
    x = np.tanh(x * 1.6)
    return reverb(x, rng, 2.5 if big else 1.4, 0.45 if big else 0.3, 4000)


def sfx_block(rng):
    t = _t(0.35)
    ring = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((520, 1), (1310, 0.6), (2270, 0.35)))
    x = 0.5 * ring * env(len(t), 0.001, 0.06) + 0.35 * filt(rng.normal(0, 1, len(t)), "high", 2000) * env(len(t), 0.0005, 0.01)
    return reverb(x, rng, 0.6, 0.2)


def sfx_whoosh(rng, dur=0.28):
    n = int(dur * SR)
    noise = rng.normal(0, 1, n)
    out = np.zeros(n)
    chunks = 12
    for i in range(chunks):          # sweep a band-pass upward through the whoosh
        a, b = i * n // chunks, (i + 1) * n // chunks
        fc = 400 + 3200 * (i / chunks) ** 1.5
        out[a:b] = filt(noise, "band", [fc * 0.6, fc * 1.6])[a:b]
    return 0.5 * norm(out) * np.hanning(n)


def sfx_zip(rng):
    """Teleport-style dodge: fast upward chirp + airy burst."""
    t = _t(0.16)
    f = 700 + 5200 * (t / t[-1]) ** 2
    chirp = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.hanning(len(t))
    air = filt(rng.normal(0, 1, len(t)), "high", 3500) * env(len(t), 0.001, 0.04)
    return 0.4 * chirp + 0.3 * air


def sfx_riser(rng, dur):
    t = _t(dur)
    n = len(t)
    p = t / dur
    noise = rng.normal(0, 1, n)
    out = np.zeros(n)
    chunks = 24
    for i in range(chunks):
        a, b = i * n // chunks, (i + 1) * n // chunks
        fc = 300 + 7000 * (i / chunks) ** 2
        out[a:b] = filt(noise, "band", [fc * 0.5, min(fc * 1.8, 18000)])[a:b]
    tone_f = 110 * 2 ** (p * 3)                     # three octaves up
    phase = 2 * np.pi * np.cumsum(tone_f) / SR
    tone = np.sin(phase) + 0.5 * np.sin(phase * 1.5) + 0.3 * np.sin(phase * 2)
    tone *= 1 + 0.15 * np.sin(2 * np.pi * (4 + 14 * p) * t)    # quickening wobble
    rumble = filt(rng.normal(0, 1, n), "low", 90) * 3 + np.sin(2 * np.pi * 42 * t)
    x = 0.35 * norm(out) + 0.25 * norm(tone) + 0.4 * norm(rumble)
    return x * (p ** 1.6)


def sfx_reverse_cymbal(rng, dur=0.7):
    n = int(dur * SR)
    x = filt(rng.normal(0, 1, n), "high", 4000)
    return 0.35 * norm(x) * (np.arange(n) / n) ** 3


def sfx_shing(rng):
    t = _t(1.2)
    parts = ((1870, 1), (2953, 0.7), (4110, 0.5), (5640, 0.35), (7330, 0.2))
    x = sum(a * np.sin(2 * np.pi * f * t) * np.exp(-t / (0.25 + i * 0.05)) for i, (f, a) in enumerate(parts))
    x += filt(rng.normal(0, 1, len(t)), "high", 6000) * env(len(t), 0.0005, 0.03)
    return 0.3 * reverb(x, rng, 1.0, 0.3)


def sfx_bell():
    t = _t(1.6)
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((880, 1), (1320, 0.5), (2210, 0.25), (3150, 0.12)))
    return 0.32 * tone * env(len(t), 0.001, 0.5)


def sfx_wah(rng):
    """Comedy 'wah-wah': two sagging slide-whistle notes."""
    out = []
    for f0, f1, dur in ((700, 560, 0.32), (540, 300, 0.75)):
        t = _t(dur)
        f = f0 + (f1 - f0) * (t / dur) ** 0.7
        f = f * (1 + 0.025 * np.sin(2 * np.pi * 6 * t))
        x = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.25 * np.sin(4 * np.pi * np.cumsum(f) / SR)
        out.append(x * np.minimum(1, t / 0.03) * np.minimum(1, (dur - t) / 0.08))
    return 0.35 * np.concatenate(out)


# ---------------------------------------------------------------- music
BASS = [0, 0, 12, 0, 0, 0, 10, 0, 0, 0, 7, 0, 3, 0, 5, 7,
        0, 0, 12, 0, 0, 0, 10, 0, 0, 0, 7, 0, 12, 10, 7, 5]
LEAD = [7, None, 10, None, 12, None, 10, 7, 5, None, 7, None, 3, None, 0, None]   # per 8th, 2 bars


def fight_music(total, rng):
    n = int(total * SR) + SR
    drums, bass, stabs, lead = (np.zeros(n) for _ in range(4))
    kt = _t(0.35)
    kick = np.sin(2 * np.pi * np.cumsum(45 + 120 * np.exp(-kt / 0.03)) / SR) * env(len(kt), 0.001, 0.13)
    kick = np.tanh(kick * 2.2)
    snare_n = int(0.25 * SR)
    snare = filt(rng.normal(0, 1, snare_n), "band", [900, 7000]) * env(snare_n, 0.001, 0.07) * 0.8
    snare += np.sin(2 * np.pi * 190 * np.arange(snare_n) / SR) * env(snare_n, 0.001, 0.04) * 0.6
    snare = reverb(snare, rng, 0.8, 0.35)
    hat_n = int(0.05 * SR)
    hat = filt(rng.normal(0, 1, hat_n), "high", 7000) * env(hat_n, 0.0005, 0.012) * 0.35
    ohat_n = int(0.22 * SR)
    ohat = filt(rng.normal(0, 1, ohat_n), "high", 6000) * env(ohat_n, 0.001, 0.08) * 0.3

    def bass_note(semi):
        t = _t(STEP * 1.05)
        f = 82.41 * 2 ** (semi / 12)
        x = saw(f, t) + 0.5 * saw(f * 1.005, t) + 0.6 * np.sin(2 * np.pi * f / 2 * t)
        x = np.tanh(3 * filt(x, "low", 900)) * env(len(t), 0.003, 0.12)
        return x

    def stab():
        t = _t(0.3)
        x = sum(saw(f, t) for f in (164.81, 246.94, 329.63, 329.63 * 1.004))
        x = np.tanh(2.5 * filt(x, "low", 2600)) * env(len(t), 0.004, 0.11)
        return x

    def lead_note(semi):
        t = _t(STEP * 2)
        f = 329.63 * 2 ** (semi / 12) * (1 + 0.006 * np.sin(2 * np.pi * 6 * t))
        x = square(f, t) * 0.6 + saw(f * 2, t) * 0.2
        return filt(x, "low", 3500) * env(len(t), 0.005, 0.16)

    bass_cache = {s: bass_note(s) for s in set(BASS)}
    lead_cache = {s: lead_note(s) for s in set(v for v in LEAD if v is not None)}
    stab_clip = stab()
    steps = int(total / STEP) + 2
    for i in range(steps):
        pos = int(i * STEP * SR)
        b16, bar = i % 16, (i // 16)
        fill = bar % 4 == 3 and b16 >= 12
        if b16 in (0, 7, 8, 10) and not fill:
            mix_in(drums, kick, pos)
        if b16 in (4, 12) or (fill and b16 in (12, 13, 14, 15)):
            mix_in(drums, snare * (0.7 if fill and b16 != 12 else 1), pos)
        if b16 % 2 == 0:
            mix_in(drums, hat, pos)
        if b16 == 14 and not fill:
            mix_in(drums, ohat, pos)
        mix_in(bass, bass_cache[BASS[i % 32]], pos)
        if b16 in (0, 10):
            mix_in(stabs, stab_clip, pos)
        if i % 2 == 0:
            v = LEAD[(i // 2) % 16]
            if v is not None:
                mix_in(lead, lead_cache[v], pos)
    return 0.9 * norm(drums) + 0.55 * norm(bass) + 0.3 * norm(stabs) + 0.16 * norm(lead)


def _gain_curve(n, spans, ramp=0.02):
    """spans: list of (t0, t1, gain) -> per-sample gain with short ramps."""
    g = np.zeros(n)
    r = int(ramp * SR)
    for t0, t1, v in spans:
        a, b = int(t0 * SR), min(n, int(t1 * SR))
        if b <= a:
            continue
        seg = np.full(b - a, float(v))
        k = min(r, (b - a) // 2)
        if k:
            seg[:k] *= np.linspace(0, 1, k)
            seg[-k:] *= np.linspace(1, 0, k)
        g[a:b] = np.maximum(g[a:b], seg)
    return g


# ---------------------------------------------------------------- mixdown
def build_soundtrack(total, events, marks, ducks, rng):
    n = int(total * SR)
    full = fight_music(total, rng)[:n]
    muffled = filt(full, "low", 650, order=4)
    modes = {"full": (1.0, 0.0), "muffled": (0.0, 0.7), "off": (0.0, 0.0), "outro": (0.6, 0.0)}
    marks = sorted(marks)
    spans_full, spans_muff = [], []
    for k, (t0, mode) in enumerate(marks):
        t1 = marks[k + 1][0] if k + 1 < len(marks) else total
        gf, gm = modes[mode]
        spans_full.append((t0, t1 + 0.02, gf))
        spans_muff.append((t0, t1 + 0.02, gm))
    duck = np.ones(n)
    for t0, t1 in ducks:
        a, b = int(t0 * SR), min(n, int(t1 * SR))
        duck[a:b] = 0.22
    duck = np.convolve(duck, np.ones(1200) / 1200, mode="same")     # smooth the dips
    music = (full * _gain_curve(n, spans_full) + muffled * _gain_curve(n, spans_muff)) * duck * 0.42

    fx = np.zeros(n)
    last = {}
    for ev in sorted(events, key=lambda e: e[0]):
        t, kind = ev[0], ev[1]
        if t - last.get(kind, -9) < {"punch": 0.08, "whoosh": 0.12, "zip": 0.1, "block": 0.08}.get(kind, 0):
            continue
        last[kind] = t
        i = int(t * SR)
        if kind == "punch":
            mix_in(fx, 0.8 * sfx_punch(rng), i)
        elif kind == "block":
            mix_in(fx, 0.7 * sfx_block(rng), i)
        elif kind == "whoosh":
            mix_in(fx, 0.45 * sfx_whoosh(rng), i)
        elif kind == "zip":
            mix_in(fx, 0.6 * sfx_zip(rng), i)
        elif kind == "impact":
            mix_in(fx, 1.0 * sfx_impact(rng), i)
        elif kind == "ko":
            rc = sfx_reverse_cymbal(rng)
            mix_in(fx, rc, i - len(rc))
            mix_in(fx, 1.25 * sfx_impact(rng, big=True), i)
        elif kind == "riser":
            mix_in(fx, 0.9 * sfx_riser(rng, ev[2]), i)
        elif kind == "burst":
            mix_in(fx, 1.0 * sfx_impact(rng), i)
            mix_in(fx, 0.8 * sfx_shing(rng), i)
        elif kind == "shing":
            mix_in(fx, 0.9 * sfx_shing(rng), i)
        elif kind == "slam":
            mix_in(fx, 0.85 * sfx_impact(rng), i)
        elif kind == "bell":
            mix_in(fx, sfx_bell(), i)
        elif kind == "bell3":
            for k in range(3):
                mix_in(fx, sfx_bell(), i + int(k * 0.22 * SR))
        elif kind == "wah":
            mix_in(fx, sfx_wah(rng), i)
    out = music + fx * 0.75
    out = np.tanh(out * 1.15)
    return norm(out, 0.89)
