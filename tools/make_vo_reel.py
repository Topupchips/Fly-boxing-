"""
Voiceover cut of the Fly Boxing reel: your face-cam takes drive the timing,
the simulation provides the B-roll, and captions follow every word.

    # 1. cut silences + re-time words (see tools/vo_cut.py) -> cut/*.json
    # 2. plan the B-roll lengths from the takes
    python tools/make_vo_reel.py plan cut/ broll_plan.json
    # 3. render the B-roll
    python tools/make_reel.py --font Archivo-VF.ttf --plan broll_plan.json --out broll/x.mp4
    # 4. assemble
    python tools/make_vo_reel.py assemble cut/ broll/ out.mp4 --font Archivo-VF.ttf [--code CODE]

Takes (by name in cut/): l38 intro, l42 training, l44 neurons, l45 title fight,
l46 call to action, l51 Base44 code.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import wave

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anime_audio as A  # noqa: E402
import anime_fx as fx  # noqa: E402
from make_reel import Type  # noqa: E402

W, H, FPS = 1080, 1920, 30
ORANGE, BLUE, WHITE, GOLD = (255, 122, 26), (46, 155, 214), (242, 241, 238), (255, 206, 64)
KO_FRAMES = 31          # impact frames + flash + hold of the K.O. hit-stop (see make_reel.hitstop)
TAGLINE = "Google just open-sourced a fly brain."

def load_cut(cut_dir):
    return {n[:-5]: json.load(open(os.path.join(cut_dir, n))) for n in os.listdir(cut_dir) if n.endswith(".json")}


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def wt(take, word, nth=1):
    hits = [w["t"] for w in take["words"] if norm(w["w"]) == word]
    return hits[nth - 1]


# ---------------------------------------------------------------- plan
def sections(cut, broll=None):
    """Final timeline: list of dicts (kind, dur, ...) and VO placements."""
    l38, l42, l44, l45 = cut["l38"], cut["l42"], cut["l44"], cut["l45"]
    t_google, t_fruit, t_so, t_and2 = (wt(l38, "google") - 0.05, wt(l38, "fruit", 2) - 0.04,
                                         wt(l38, "so") - 0.04, wt(l38, "and", 3) - 0.04)
    lead_in = 0.3
    secs = [
        {"kind": "broll", "shot": "cold", "dur": None},
        {"kind": "broll", "shot": "freeze", "dur": lead_in + t_google},
        {"kind": "face", "take": "l38", "a": t_google, "b": t_fruit},
        {"kind": "broll", "shot": "brainspin", "dur": t_so - t_fruit},
        {"kind": "face", "take": "l38", "a": t_so, "b": t_and2},
        {"kind": "broll", "shot": "fightcut", "dur": l38["dur"] - t_and2 + 0.12},
    ]
    gap = 0.08
    t42 = wt(l42, "so") - 0.04
    secs += [
        {"kind": "broll", "shot": "gen0", "dur": gap + t42},
        {"kind": "broll", "shot": "powerup", "dur": l42["dur"] - t42 + 0.2},
        {"kind": "broll", "shot": "brains", "dur": l44["dur"] + 0.22},
    ]
    a45 = wt(l45, "and", 2) - 0.02
    lead = max(1.0, 0.15 + a45 + 0.9 - 1.8)            # fight time between the VS screen and the knockout
    secs += [
        {"kind": "broll", "shot": "vs", "dur": 1.8},
        {"kind": "broll", "shot": "title", "dur": None, "lead": lead},
        {"kind": "face", "take": "l46", "a": 0.0, "b": cut["l46"]["dur"], "overlay": "cta"},
        {"kind": "face", "take": "l51", "a": 0.0, "b": cut["l51"]["dur"], "overlay": "code"},
        {"kind": "loop", "dur": 8 / FPS},
    ]
    # actual B-roll lengths once rendered
    if broll:
        lens = shot_lengths(broll)
        for s in secs:
            if s["kind"] == "broll" and s["dur"] is None:
                s["dur"] = lens[s["shot"]]
    t = 0.0
    for s in secs:
        if s["kind"] == "face":
            s["dur"] = s["b"] - s["a"]
        s["start"] = t
        t += s["dur"] or 0
    # voiceover placements (final-time start, take, take range)
    fr = next(s for s in secs if s.get("shot") == "freeze")
    g0 = next(s for s in secs if s.get("shot") == "gen0")
    br = next(s for s in secs if s.get("shot") == "brains")
    vs = next(s for s in secs if s.get("shot") == "vs")
    ti = next(s for s in secs if s.get("shot") == "title")
    vo = [
        (fr["start"] + lead_in, "l38", 0.0, l38["dur"]),
        (g0["start"] + gap, "l42", 0.0, l42["dur"]),
        (br["start"] + 0.1, "l44", 0.0, l44["dur"]),
        (vs["start"] + 0.15, "l45", 0.0, a45),
        (ti["start"] + ti["lead"] + KO_FRAMES / FPS + 0.22, "l45", a45, l45["dur"]),
    ]
    placed = {v[1] for v in vo}
    for s in secs:                      # face-cam takes not already running as voiceover
        if s["kind"] == "face" and s["take"] not in placed:
            vo.append((s["start"], s["take"], s["a"], s["b"]))
    return secs, vo


def shot_lengths(broll):
    tl = broll["timeline"] + [[broll["frames"] / FPS, "_end"]]
    return {name: tl[i + 1][0] - t for i, (t, name) in enumerate(tl[:-1])}


def cmd_plan(cut_dir, out):
    cut = load_cut(cut_dir)
    secs, _ = sections(cut)
    margin = 0.25
    plan = []
    for s in secs:
        if s["kind"] != "broll":
            continue
        e = {"shot": s["shot"]}
        if s["shot"] == "cold":
            e.update(dur=2.1, tagline=TAGLINE)
        elif s["shot"] == "title":
            e.update(lead=s["lead"], ko_hold=1.7)
        elif s["shot"] == "powerup":
            e.update(dur=s["dur"] - 2 / FPS)          # powerup adds two flash frames
        else:
            e.update(dur=s["dur"] + margin)
        plan.append(e)
    json.dump(plan, open(out, "w"), indent=1)
    for e in plan:
        print(e)


# ---------------------------------------------------------------- captions
def caption_pages(words, t0):
    """Group words into short pages (<=3 words, break on punctuation), with fixes applied."""
    toks = []
    i = 0
    while i < len(words):
        w = dict(words[i])
        n = norm(w["w"])
        # "a hundred and fifty" -> "150"
        if n == "a" and i + 3 < len(words) and [norm(x["w"]) for x in words[i + 1:i + 4]] == ["hundred", "and", "fifty"]:
            toks.append({"w": "150", "t": w["t"], "e": words[i + 3]["e"], "p": False})
            i += 4
            continue
        txt = w["w"]
        if n == "engineer":
            txt = "engineers"
        if n == "for" and i > 0 and norm(words[i - 1]["w"]) == "champion":
            txt = "fought"
        if n == "fortyfour":
            txt = "44" + ("," if txt.endswith(",") else "")
        if n == "base" and i + 1 < len(words) and norm(words[i + 1]["w"]) == "fortyfour":
            nxt = words[i + 1]
            toks.append({"w": "Base44" + ("," if nxt["w"].endswith(",") else ""), "t": w["t"], "e": nxt["e"], "p": nxt["w"].endswith(",")})
            i += 2
            continue
        punct = txt.endswith((",", ".", "?", "!"))
        toks.append({"w": txt, "t": w["t"], "e": w["e"], "p": punct})
        i += 1
    # set phrases stay in one caption (and light up together)
    phrases = [("fruit", "fly", "brain"), ("open", "source"), ("number", "two"), ("fly", "neurons"), ("single", "punch")]
    joined, k = [], 0
    while k < len(toks):
        for ph in phrases:
            n = len(ph)
            if tuple(norm(x["w"]) for x in toks[k:k + n]) == ph:
                grp = toks[k:k + n]
                joined.append({"w": " ".join(x["w"].rstrip(",.") for x in grp) + ("," if grp[-1]["p"] else ""),
                               "t": grp[0]["t"], "e": grp[-1]["e"], "p": grp[-1]["p"]})
                k += n
                break
        else:
            joined.append(toks[k])
            k += 1
    toks = joined
    weak = {"and", "a", "the", "for", "to", "of", "so", "but", "my", "is", "this", "you", "how", "i", "i'll", "if"}
    starters = {"and", "so", "but", "if", "whenever"}
    pages, cur = [], []
    for k, tk in enumerate(toks):
        nw = norm(tk["w"])
        if cur and nw in starters:                      # new thought -> new caption
            pages.append(cur)
            cur = []
        cur.append(tk)
        chars = sum(len(x["w"]) for x in cur) + len(cur)
        full = len(cur) == 3 or chars > 16
        if tk["p"] or full:
            # never end a caption on a little word: carry it over to the next one
            carry = []
            while not tk["p"] and len(cur) > 1 and norm(cur[-1]["w"]) in weak:
                carry.insert(0, cur.pop())
            pages.append(cur)
            cur = carry
    if cur:
        pages.append(cur)
    # fold a dangling one-word caption into its neighbour when it fits
    merged = []
    for pg in pages:
        if merged and len(pg) == 1 and not merged[-1][-1]["p"] and len(merged[-1]) < 3 \
                and sum(len(x["w"]) for x in merged[-1] + pg) + len(merged[-1]) < 22:
            merged[-1] = merged[-1] + pg
        else:
            merged.append(pg)
    pages = merged
    out = []
    for k, pg in enumerate(pages):
        start = t0 + pg[0]["t"]
        end = t0 + (pages[k + 1][0]["t"] if k + 1 < len(pages) else pg[-1]["e"] + 0.25)
        out.append({"start": start, "end": end, "words": [(x["w"].rstrip(",.").replace("Engineer", "engineers"), t0 + x["t"]) for x in pg]})
    return out


def draw_caption(img, T, page, t, y):
    f = T.font(98, 900, 80)
    d = ImageDraw.Draw(img)
    words = [w.upper() for w, _ in page["words"]]
    space = d.textlength(" ", font=f)
    widths = [d.textlength(w, font=f) for w in words]
    total = sum(widths) + space * (len(words) - 1)
    lines = [list(range(len(words)))]
    if total > 980 and len(words) > 1:
        lines = [list(range(len(words) - 1)), [len(words) - 1]]
    age = t - page["start"]
    pop = min(1.0, 0.86 + age / 0.1 * 0.14)
    if pop < 1:
        f = T.font(int(98 * pop), 900, 80)
        widths = [d.textlength(w, font=f) for w in words]
        space = d.textlength(" ", font=f)
    lh = int(f.size * 1.05)
    y0 = y - lh * len(lines) / 2
    active = max([i for i, (_, ts) in enumerate(page["words"]) if ts <= t] or [0])
    for li, idxs in enumerate(lines):
        lw = sum(widths[i] for i in idxs) + space * (len(idxs) - 1)
        x = (W - lw) / 2
        for i in idxs:
            col = ORANGE if i == active else WHITE
            d.text((x, y0 + li * lh), words[i], font=f, fill=col, stroke_width=int(f.size * 0.09), stroke_fill=(0, 0, 0))
            x += widths[i] + space


# ---------------------------------------------------------------- assembly
def face_frames(take, a, b, src, tmp):
    """Decode the kept pieces of a face-cam take between cut-time a and b, 30 fps, 1080x1920."""
    out, acc = [], 0.0
    zoom_flip = False
    for ka, kb in take["keeps"]:
        seg = kb - ka
        lo, hi = max(a, acc), min(b, acc + seg)
        if hi > lo:
            s0, s1 = ka + (lo - acc), ka + (hi - acc)
            path = os.path.join(tmp, f"face_{os.path.basename(src)}_{s0:.3f}.raw")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{s0:.3f}", "-to", f"{s1:.3f}", "-i", src, "-map", "0:v:0",
                            "-vf", f"fps={FPS},scale={W}:{H}:flags=lanczos", "-f", "rawvideo", "-pix_fmt", "rgb24", path], check=True)
            raw = np.fromfile(path, np.uint8)
            n = len(raw) // (W * H * 3)
            frames = raw[: n * W * H * 3].reshape(n, H, W, 3)
            want = int(round((hi - lo) * FPS))
            idx = np.clip(np.round(np.linspace(0, n - 1, max(1, want))).astype(int), 0, n - 1) if n else []
            for j in idx:
                out.append((frames[j], zoom_flip))
            os.remove(path)
            zoom_flip = not zoom_flip          # punch in/out on every jump cut
        acc += seg
    return out


def vo_audio(take, src, tmp):
    """Cut a take's audio to its keep pieces (with tiny crossfades), cleaned up, mono 44.1k."""
    path = os.path.join(tmp, "vo.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", src, "-map", "0:a:0", "-ac", "1", "-ar", str(A.SR), path], check=True)
    with wave.open(path) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
    fade = int(0.012 * A.SR)
    parts = []
    for ka, kb in take["keeps"]:
        seg = x[int(ka * A.SR):int(kb * A.SR)].copy()
        if len(seg) > 2 * fade:
            seg[:fade] *= np.linspace(0, 1, fade)
            seg[-fade:] *= np.linspace(1, 0, fade)
        parts.append(seg)
    y = np.concatenate(parts)
    y = A.filt(y, "high", 85)
    # gentle compression + level
    env = np.sqrt(np.convolve(y ** 2, np.ones(441) / 441, "same") + 1e-9)
    gain = np.minimum(1.0, (0.12 / env) ** 0.35)
    y = y * gain
    return y / (np.abs(y).max() + 1e-9) * 0.92


def cmd_assemble(cut_dir, broll_dir, out, font, code):
    cut = load_cut(cut_dir)
    broll = json.load(open(os.path.join(broll_dir, "broll.json")))
    secs, vo = sections(cut, broll)
    T = Type(font)
    tmp = os.path.join(os.path.dirname(os.path.abspath(out)), "tmp_assemble")
    os.makedirs(tmp, exist_ok=True)
    lens = shot_lengths(broll)
    shot_start = {name: t for t, name in broll["timeline"]}
    total = sum(s["dur"] for s in secs)
    n_frames = int(round(total * FPS))

    # captions for every VO placement
    pages = []
    for t0, tk, a, b in vo:
        take = cut[tk]
        ws = [dict(w, t=w["t"] - a, e=w["e"] - a) for w in take["words"] if a - 0.01 <= w["t"] < b - 0.01]
        pages += caption_pages(ws, t0)
    pages.sort(key=lambda p: p["start"])
    for k in range(len(pages) - 1):
        pages[k]["end"] = min(pages[k]["end"], pages[k + 1]["start"])

    # face-cam frames
    face = {}
    for i, s in enumerate(secs):
        if s["kind"] == "face":
            face[i] = face_frames(cut[s["take"]], s["a"], s["b"], cut[s["take"]]["src"], tmp)

    def broll_frame(shot, t_in):
        f = int(round((shot_start[shot] + min(t_in, lens[shot] - 1 / FPS)) * FPS))
        f = min(f, broll["frames"] - 1)
        return Image.open(os.path.join(broll_dir, "frames", f"{f:05d}.png")).convert("RGBA")

    cta = fx.slam_text(T.font(150, 900, 72), "COMMENT", WHITE, angle=-5, shear=0.15)
    b44 = fx.slam_text(T.font(230, 900, 72), "BASE44", ORANGE, back=WHITE, angle=-6)
    code_card = None
    if code:
        code_card = fx.slam_text(T.font(170, 900, 72), code, GOLD, back=(0, 0, 0), angle=-4)
        code_lab = fx.slam_text(T.font(64, 850, 80), "BASE44 CODE", WHITE, angle=-4, shear=0.12)

    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS),
                           "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p",
                           os.path.join(tmp, "video.mp4")], stdin=subprocess.PIPE)
    first = None
    last = None
    for fi in range(n_frames):
        t = fi / FPS
        si = max(i for i, s in enumerate(secs) if s["start"] <= t + 1e-6)
        s = secs[si]
        lt = t - s["start"]
        cap_y = 1640
        if s["kind"] == "broll":
            img = broll_frame(s["shot"], lt)
            if s["shot"] == "brains":
                cap_y = 260
            if s["shot"] in ("cold", "freeze"):           # re-set the tagline on one line
                ImageDraw.Draw(img).rectangle((0, 140, W, 380), fill=(0, 0, 0, 255))
                tag = T.font(58, 850, 78)
                d = ImageDraw.Draw(img)
                tw = d.textlength(TAGLINE, font=tag)
                if s["shot"] == "freeze" or lt > 6 / FPS:
                    d.text(((W - tw) / 2, 225), TAGLINE, font=tag, fill=WHITE if s["shot"] == "cold" else (205, 205, 205),
                           stroke_width=5, stroke_fill=(0, 0, 0))
            if s["shot"] == "title":                      # captions replace the scoreboard here
                ImageDraw.Draw(img).rectangle((0, 1482, W, H), fill=(0, 0, 0, 255))
            if s["shot"] == "brainspin":                  # lift the brain out of the black
                rgb = img.convert("RGB")
                rgb = ImageEnhance.Brightness(rgb).enhance(1.55)
                rgb = ImageChops.add(rgb, ImageEnhance.Brightness(rgb.filter(ImageFilter.GaussianBlur(14))).enhance(1.1))
                img = rgb.convert("RGBA")
        elif s["kind"] == "face":
            frames = face[si]
            k = min(len(frames) - 1, int(lt * FPS))
            arr, zoomed = frames[k]
            img = Image.fromarray(arr).convert("RGBA")
            if zoomed:                                # alternate framing hides the jump cuts
                bw, bh = W / 1.08, H / 1.08
                img = img.resize((W, H), Image.BICUBIC, box=((W - bw) / 2, (H - bh) / 2 - 40, (W + bw) / 2, (H + bh) / 2 - 40))
            cap_y = 1380
            if s.get("overlay") == "cta":
                fx.paste_center(img, cta, (540, 300), 1.6 - min(1.0, lt / 0.12) * 0.6)
                if lt > 0.15:
                    fx.paste_center(img, b44, (540, 500), 1.7 - min(1.0, (lt - 0.15) / 0.12) * 0.7)
            if s.get("overlay") == "code" and code_card is not None:
                cw = cut["l51"]
                t_code = wt(cw, "code") - 0.1
                if lt >= t_code:
                    dt = lt - t_code
                    box = Image.new("RGBA", (W, 420), (0, 0, 0, 150))
                    img.alpha_composite(box, (0, 170))
                    fx.paste_center(img, code_lab, (540, 260))
                    fx.paste_center(img, code_card, (540, 440), 1.5 - min(1.0, dt / 0.12) * 0.5)
        else:  # loop back into the first frame
            p = (lt + 1 / FPS) / s["dur"]
            img = Image.blend(last, first, min(1.0, p))
        if s["kind"] != "loop":
            for pg in pages:
                if pg["start"] <= t < pg["end"]:
                    draw_caption(img, T, pg, t, cap_y)
                    break
        if first is None:
            first = img.copy()
        last = img
        ff.stdin.write(np.asarray(img.convert("RGB")).tobytes())
    ff.stdin.close()
    ff.wait()

    # ---- audio --------------------------------------------------------------
    rng = np.random.default_rng(11)
    n = int(total * A.SR)
    voice = np.zeros(n)
    for t0, tk, a, b in vo:
        src = cut[tk]["src"]
        y = vo_audio(cut[tk], src, tmp)
        y = y[int(a * A.SR):int(b * A.SR)]
        A.mix_in(voice, y, int(t0 * A.SR))
    # remap B-roll sound cues into the final timeline
    events = []
    for s in secs:
        if s["kind"] != "broll":
            continue
        b0 = shot_start[s["shot"]]
        for ev in broll["events"]:
            if b0 <= ev[0] < b0 + s["dur"]:
                e = list(ev)
                e[0] = s["start"] + (ev[0] - b0)
                events.append(e)
    by = {s.get("shot"): s for s in secs if s["kind"] == "broll"}
    ti = by["title"]
    ko_t = ti["start"] + ti["lead"]
    marks = [(0.0, "full"), (by["freeze"]["start"], "off"), (by["freeze"]["start"] + 0.35, "bed"),
             (by["gen0"]["start"], "muffled"), (by["powerup"]["start"], "off"),
             (by["brains"]["start"], "full"), (ko_t, "off"), (ti["start"] + ti["dur"], "bed")]
    events.append((by["freeze"]["start"], "scratch"))
    ducks = [(ko_t, ko_t + KO_FRAMES / FPS)]
    for d in broll["ducks"]:
        for s in secs:
            if s["kind"] == "broll" and s["shot"] == "cold":
                b0 = shot_start["cold"]
                if b0 <= d[0] < b0 + s["dur"]:
                    ducks.append((s["start"] + d[0] - b0, s["start"] + d[1] - b0))
    mus_fx = A.build_soundtrack(total, events, marks, ducks, rng)
    venv = np.convolve(np.abs(voice), np.ones(2205) / 2205, "same")
    venv = np.minimum(1.0, venv / (venv.max() * 0.25 + 1e-9))
    duck = 1 - 0.72 * np.convolve(venv, np.ones(4410) / 4410, "same")
    mix = mus_fx * duck * 0.8 + voice * 1.0
    mix = np.tanh(mix * 1.05)
    mix = mix / (np.abs(mix).max() + 1e-9) * 0.92
    wav = os.path.join(tmp, "mix.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(A.SR)
        w.writeframes((mix * 32767).astype("<i2").tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.join(tmp, "video.mp4"), "-i", wav, "-c:v", "copy", "-c:a", "aac",
                    "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check=True)
    with open(os.path.splitext(out)[0] + "_timeline.txt", "w") as f:
        for s in secs:
            f.write(f"{s['start']:6.2f}s  {s['kind']:6s} {s.get('shot', s.get('take', ''))}  ({s['dur']:.2f}s)\n")
        f.write(f"{total:6.2f}s  end\n\ncaptions:\n")
        for pg in pages:
            f.write(f"{pg['start']:6.2f}-{pg['end']:6.2f}  {' '.join(w for w, _ in pg['words'])}\n")
    print(f"wrote {out} ({total:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("plan"); p1.add_argument("cut"); p1.add_argument("out")
    p2 = sub.add_parser("assemble"); p2.add_argument("cut"); p2.add_argument("broll"); p2.add_argument("out")
    p2.add_argument("--font", default=""); p2.add_argument("--code", default="")
    a = ap.parse_args()
    if a.cmd == "plan":
        cmd_plan(a.cut, a.out)
    else:
        cmd_assemble(a.cut, a.broll, a.out, a.font, a.code)


if __name__ == "__main__":
    main()
