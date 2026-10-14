"""
Render a vertical (1080x1920) Instagram/TikTok reel of Fly Boxing.

    pip install playwright pillow numpy && playwright install chromium
    python tools/make_reel.py --font Archivo-VF.ttf          # -> reel/fly_boxing_reel.mp4

Drives the real viewer (docs/index.html?capture) frame by frame, so every
shot is the actual simulation: the recorded fights, the 3D ring and the
FlyWire brain view. Captions and a scoreboard are drawn on top, and an
original soundtrack is synthesized (beat, punches, blocks, whooshes, bells)
with every sound placed on the frame where the event happens.

--font: a TTF/OTF to set captions in. The Archivo variable font from
@fontsource-variable/archivo (converted to TTF) matches the site.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import os
import shutil
import subprocess
import threading
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1080, 1920, 30
STAGE_Y = 400                     # stage occupies 400..1480
ORANGE, BLUE, WHITE, MUTED = (255, 122, 26), (46, 155, 214), (242, 241, 238), (140, 144, 152)

CAPTURE_CSS = """
html, body { background:#000 !important; overflow:hidden !important; }
body * { visibility:hidden !important; }
.stage, .stage *, .b3, .b3 * { visibility:visible !important; }
.overlay { transition:none !important; }
.ns.idle { opacity:1 !important; }
.stage { position:fixed !important; left:0; top:0; width:1080px !important; height:1080px !important;
         aspect-ratio:auto !important; border-radius:0 !important; z-index:10; }
.views, .hint { display:none !important; }
#b3A, #b3Z { position:fixed !important; left:0; width:1080px !important; height:400px !important;
             aspect-ratio:auto !important; border-radius:0 !important; z-index:10; margin:0 !important; }
#b3A { top:1080px } #b3Z { top:1480px }
.result b { font-size:190px !important; } .result span { font-size:44px !important; margin-top:24px !important; }
.banner b { font-size:150px !important; }
.b3 .cap { display:none !important; }
"""


# ---------------------------------------------------------------- text
class Type:
    def __init__(self, path):
        self.path = path

    @functools.lru_cache(maxsize=64)
    def font(self, size, weight=900, width=78):
        if not self.path:
            return ImageFont.load_default(size)
        f = ImageFont.truetype(self.path, size)
        try:
            f.set_variation_by_axes([weight, width])
        except Exception:
            pass
        return f


def wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for word in text.split():
        trial = (cur + " " + word).strip()
        if draw.textlength(trial, font=font) <= max_w or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def caption(img, T, text, y_center, size=86, color=WHITE, max_w=960, pop=1.0, weight=900):
    """Big centred caption with a soft shadow; pop scales it for an entrance."""
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = T.font(max(10, int(size * pop)), weight)
    lines = wrap(d, text, f, max_w)
    lh = int(f.size * 1.08)
    y = y_center - lh * len(lines) / 2
    for ln in lines:
        w = d.textlength(ln, font=f)
        d.text(((W - w) / 2, y), ln, font=f, fill=color + (255,), stroke_width=max(2, int(f.size * 0.07)), stroke_fill=(0, 0, 0, 255))
        y += lh
    img.alpha_composite(layer)


def scoreboard(img, T, st):
    d = ImageDraw.Draw(img)
    names, fr = st["names"], st["frame"]
    rd, ti = st["round"]
    nf = T.font(46, 800, 80)
    y = 1520
    d.text((48, y), names[0], font=nf, fill=ORANGE)
    w = d.textlength(names[1], font=nf)
    d.text((W - 48 - w, y), names[1], font=nf, fill=BLUE)
    secs = max(0, int(np.ceil((450 - ti) / 30)))
    clock = f"0:{secs:02d}"
    cf = T.font(64, 850, 75)
    cw = d.textlength(clock, font=cf)
    d.text(((W - cw) / 2, y - 6), clock, font=cf, fill=WHITE)
    rf = T.font(26, 500, 100)
    rt = f"Round {rd + 1} of 3"
    d.text(((W - d.textlength(rt, font=rf)) / 2, y + 66), rt, font=rf, fill=MUTED)
    bw, by = 380, y + 74
    for s, x0, col in ((0, 48, ORANGE), (1, W - 48 - bw, BLUE)):
        d.rounded_rectangle((x0, by, x0 + bw, by + 16), 8, fill=(30, 33, 38))
        hp = max(0.0, fr[s]["hp"]) / 100
        if hp > 0:
            if s == 0:
                d.rounded_rectangle((x0, by, x0 + bw * hp, by + 16), 8, fill=col)
            else:
                d.rounded_rectangle((x0 + bw * (1 - hp), by, x0 + bw, by + 16), 8, fill=col)
    tag = T.font(30, 600, 100)
    t = "fly-boxing.vercel.app"
    d.text(((W - d.textlength(t, font=tag)) / 2, 1850), t, font=tag, fill=MUTED)


# ---------------------------------------------------------------- audio
SR = 44100


def env(n, attack=0.002, decay=0.15):
    t = np.arange(n) / SR
    a = np.minimum(1, t / max(attack, 1e-6))
    return a * np.exp(-t / decay)


def sfx_punch(rng):
    n = int(0.35 * SR); t = np.arange(n) / SR
    thump = np.sin(2 * np.pi * (95 * t - 60 * t * t)) * env(n, 0.001, 0.09)
    crack = rng.normal(0, 1, n) * env(n, 0.0005, 0.018)
    return 0.9 * thump + 0.5 * crack


def sfx_block(rng):
    n = int(0.25 * SR); t = np.arange(n) / SR
    return 0.55 * np.sin(2 * np.pi * 420 * t) * env(n, 0.001, 0.04) + 0.25 * rng.normal(0, 1, n) * env(n, 0.0005, 0.01)


def sfx_whoosh(rng):
    n = int(0.3 * SR)
    noise = rng.normal(0, 1, n)
    k = np.hanning(n)
    out = np.convolve(noise, np.ones(18) / 18, mode="same") * k
    return 0.35 * out / (np.abs(out).max() + 1e-9)


def sfx_bell():
    n = int(1.6 * SR); t = np.arange(n) / SR
    tone = sum(a * np.sin(2 * np.pi * f * t) for f, a in ((880, 1), (1320, 0.5), (2210, 0.25), (3150, 0.12)))
    return 0.35 * tone * env(n, 0.001, 0.5)


def beat_track(seconds, bpm=96, rng=None):
    n = int(seconds * SR); out = np.zeros(n)
    beat = 60 / bpm
    kick_n = int(0.4 * SR); kt = np.arange(kick_n) / SR
    kick = np.sin(2 * np.pi * (110 * kt - 70 * kt * kt)) * env(kick_n, 0.001, 0.12)
    snare_n = int(0.2 * SR)
    snare = rng.normal(0, 1, snare_n) * env(snare_n, 0.001, 0.05) * 0.5
    hat_n = int(0.05 * SR)
    hat = np.diff(rng.normal(0, 1, hat_n + 1)) * env(hat_n, 0.0005, 0.012) * 0.25
    step = 0
    tpos = 0.0
    while tpos < seconds:
        i = int(tpos * SR)
        b = step % 8
        if b in (0, 3, 5):
            mix_in(out, kick * 0.8, i)
        if b in (2, 6):
            mix_in(out, snare, i)
        mix_in(out, hat, i)
        step += 1
        tpos += beat / 2
    return out


def mix_in(buf, clip, start):
    if start >= len(buf):
        return
    end = min(len(buf), start + len(clip))
    buf[start:end] += clip[: end - start]


# ---------------------------------------------------------------- capture
class Server:
    def __init__(self, root):
        class Quiet(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass
        handler = functools.partial(Quiet, directory=root)
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "reel", "fly_boxing_reel.mp4"))
    ap.add_argument("--preview", action="store_true", help="render every 6th frame only, for a quick look")
    args = ap.parse_args()
    from playwright.sync_api import sync_playwright

    T = Type(args.font)
    out_dir = os.path.dirname(args.out)
    frames_dir = os.path.join(out_dir, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    events = []                     # (video_time, kind)
    frame_no = [0]
    last_img = [None]

    def emit(img):
        path = os.path.join(frames_dir, f"{frame_no[0]:05d}.png")
        if not args.preview or frame_no[0] % 6 == 0:
            img.convert("RGB").save(path, compress_level=1)
        frame_no[0] += 1
        last_img[0] = img
        if frame_no[0] % 60 == 0:
            print(f"  {frame_no[0]} frames ({frame_no[0] / FPS:.0f}s)", flush=True)

    srv = Server(os.path.join(ROOT, "docs"))
    with sync_playwright() as p:
        browser = p.chromium.launch(args=["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
        page = browser.new_page(viewport={"width": 1080, "height": 1880})
        if args.font:
            font_bytes = open(args.font, "rb").read()
            page.route("**/fonts.googleapis.com/**", lambda r: r.fulfill(
                status=200, content_type="text/css",
                body="@font-face{font-family:'Archivo';src:url(https://fonts.gstatic.com/archivo.ttf) format('truetype');"
                     "font-weight:100 900;font-stretch:62% 125%;}"))
            page.route("**/fonts.gstatic.com/**", lambda r: r.fulfill(status=200, content_type="font/ttf", body=font_bytes))
        page.goto(f"http://127.0.0.1:{srv.port}/index.html?capture", wait_until="load")
        page.add_style_tag(content=CAPTURE_CSS)
        page.wait_for_function("!!(window.flybox && window.flybox.tick)", polling=200)
        page.wait_for_timeout(1500)
        page.evaluate("flybox.tick(0.05)")

        def js(code):
            return page.evaluate(code)

        def stage_img():
            png = page.screenshot(clip={"x": 0, "y": 0, "width": 1080, "height": 1080})
            return Image.open(__import__("io").BytesIO(png)).convert("RGBA")

        def brains_img():
            png = page.screenshot(clip={"x": 0, "y": 1080, "width": 1080, "height": 800})
            return Image.open(__import__("io").BytesIO(png)).convert("RGBA")

        def cinematic(mind=8.0, maxd=17.0, height=3.6, look=1.0, snap=False, k=0.16):
            js(f"""(()=>{{const a=flybox.flies[0].position,b=flybox.flies[1].position;
              const mx=(a.x+b.x)/2, mz=(a.z+b.z)/2, dx=b.x-a.x, dz=b.z-a.z, L=Math.hypot(dx,dz)||1;
              const C=window.__cine||(window.__cine={{}});
              const dist=Math.min({maxd},Math.max({mind},L*1.1+5));
              // side-on to the fighters, on whichever side has more open ring; ease to the other
              // side only when the current one would put the camera outside the ropes
              const pick=()=>{{ const px=-dz/L, pz=dx/L, s1=Math.hypot(mx+px*dist, mz+pz*dist), s2=Math.hypot(mx-px*dist, mz-pz*dist);
                                return s1<s2 ? [px,pz] : [-px,-pz]; }};
              const outside=d=>Math.max(Math.abs(mx+d[0]*dist), Math.abs(mz+d[1]*dist)) > 13.2;
              if({str(snap).lower()}||!C.dir) C.dir=pick();
              else if(outside(C.dir)) {{ const n=pick(); C.dir=[C.dir[0]+(n[0]-C.dir[0])*0.08, C.dir[1]+(n[1]-C.dir[1])*0.08];
                                        const m=Math.hypot(C.dir[0],C.dir[1])||1; C.dir=[C.dir[0]/m, C.dir[1]/m]; }}
              const c=flybox.controls, cam=flybox.camera.position, k={1 if snap else k};
              c.target.x += (mx-c.target.x)*k; c.target.y={look}; c.target.z += (mz-c.target.z)*k;
              let cx=c.target.x+C.dir[0]*dist, cz=c.target.z+C.dir[1]*dist;
              cx=Math.max(-13.2,Math.min(13.2,cx)); cz=Math.max(-13.2,Math.min(13.2,cz));
              cam.x += (cx-cam.x)*k; cam.y += ({height}+dist*0.12-cam.y)*k; cam.z += (cz-cam.z)*k;}})()""")

        def snap_camera(**kw):
            cinematic(snap=True, **kw)

        def collect_events(prev_i, cur_i, frames_fn):
            """Sound cues for every hit / block / dodge between two sim frames."""
            if cur_i <= prev_i:
                return
            fr = frames_fn(prev_i, cur_i)
            t_now = frame_no[0] / FPS
            for k, (a, b) in enumerate(zip(fr[:-1], fr[1:])):
                for s in (0, 1):
                    if b[s]["hit"] and not a[s]["hit"]:
                        events.append((t_now, "block" if b[s]["blk"] else "punch"))
                    if b[s]["d"] and not a[s]["d"]:
                        events.append((t_now, "whoosh"))

        def frames_between(i0, i1):
            return js(f"(()=>{{const F=FLYBOX.fights[__fi]; return F.frames.slice({i0},{i1}+1);}})()".replace("__fi", "window.__fi"))

        def segment(fight, start, seconds, speed, cam, cap, bottom="score", cap2=None, cap2_at=None, hold_result=0.0):
            js(f"window.__fi={fight}; flybox.load({fight}); flybox.seek({start}); flybox.setSpeed({speed}); flybox.play(true)")
            if cam == "ringside":
                js("flybox.view('ringside')")
            js("flybox.tick(1/30)")
            cam_kw = {"close": {}, "medium": {"mind": 10.5, "maxd": 18.0, "height": 5.2, "k": 0.22},
                      "wide": {"mind": 12.0, "maxd": 21.0, "height": 7.0}}.get(cam)
            if cam_kw is not None:
                js("flybox.noTween()")
                js("flybox.tick(1/30)")
                snap_camera(**cam_kw)
            n = int(seconds * FPS)
            prev = start
            for k in range(n):
                if cam_kw is not None:
                    cinematic(**cam_kw)
                js("flybox.tick(1/30)")
                st = js("flybox.state()")
                cur = int(st["t"])
                collect_events(prev, cur, frames_between)
                prev = cur
                img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                if bottom == "brains":
                    # cropped ring on top, the two fighters' real fly brains stacked below
                    img.alpha_composite(stage_img().crop((0, 200, 1080, 920)), (0, STAGE_Y))
                    img.alpha_composite(brains_img(), (0, 1120))
                    d = ImageDraw.Draw(img)
                    lab = T.font(40, 800, 85)
                    d.text((40, 1136), f"{st['names'][0]}'s brain", font=lab, fill=ORANGE)
                    d.text((40, 1536), f"{st['names'][1]}'s brain", font=lab, fill=BLUE)
                else:
                    img.alpha_composite(stage_img(), (0, STAGE_Y))
                if bottom != "brains":
                    scoreboard(img, T, st)
                text = cap2 if (cap2 and cap2_at is not None and k >= cap2_at * FPS) else cap
                k0 = int(cap2_at * FPS) if (cap2 and cap2_at is not None and k >= cap2_at * FPS) else 0
                pop = min(1.0, 0.82 + (k - k0) / 6 * 0.18)
                caption(img, T, text, 200, pop=pop)
                emit(img)
                if not st["playing"] and st["t"] >= st["n"] - 1:
                    break
            if hold_result > 0:
                events.append((frame_no[0] / FPS, "bell3"))
                for k in range(int(hold_result * FPS)):
                    js("flybox.tick(1/30)")
                    img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                    img.alpha_composite(stage_img(), (0, STAGE_Y))
                    scoreboard(img, T, js("flybox.state()"))
                    caption(img, T, cap2 or cap, 200)
                    emit(img)

        def card(text, seconds, sub=None, bell=True, bg=None):
            if bell:
                events.append((frame_no[0] / FPS, "bell"))
            for k in range(int(seconds * FPS)):
                img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                if bg is not None:
                    blur = bg.filter(ImageFilter.GaussianBlur(14))
                    dark = Image.new("RGBA", (W, H), (0, 0, 0, 170))
                    img.alpha_composite(blur); img.alpha_composite(dark)
                caption(img, T, text, 860 if sub else 960, size=104, pop=min(1, 0.8 + k / 6 * 0.2))
                if sub:
                    for j, (line, col, size) in enumerate(sub):
                        caption(img, T, line, 1180 + j * 120, size=size, color=col, weight=800)
                emit(img)

        # ---- find the moments --------------------------------------------------
        fights = js("FLYBOX.fights.map(f=>({title:f.title,n:f.frames.length,winner:f.winner,ko:f.ko}))")
        idx = {f["title"]: i for i, f in enumerate(fights)}
        title, spar, newb = idx["Title Fight"], idx["Sparring Session"], idx["Newborns"]
        hit_t = js(f"(()=>{{const f=FLYBOX.fights[{title}].frames; for(let i=400;i<f.length;i++) if(f[i][1].hit&&!f[i][1].blk&&!f[i][0].hit) return i; return 300;}})()")
        spar_hit = js(f"(()=>{{const f=FLYBOX.fights[{spar}].frames; for(let i=30;i<f.length;i++) if(f[i][1].hit&&!f[i][1].blk) return i; return 60;}})()")
        n_title = fights[title]["n"]

        # ---- the reel ----------------------------------------------------------
        events.append((0.0, "bell"))
        segment(title, hit_t - 14, 3.2, 0.3, "close", "I gave fruit flies a real brain and taught them to box")
        segment(newb, 40, 4.6, 2, "wide", "Generation 0 has absolutely no idea what it's doing")
        card("150 generations of evolution later...", 1.6)
        segment(spar, max(0, spar_hit - 45), 5.6, 1, "medium", "Now they lunge, block and dodge", cap2="Nobody coded a single punch", cap2_at=2.8)
        segment(title, 200, 4.8, 1, "wide", "Every punch is real neurons firing in a real fly brain map", bottom="brains")
        segment(title, max(0, n_title - 1 - 150), 5.2, 1, "close", "Then the champion fought the #2 contender",
                cap2="...and got KNOCKED OUT", cap2_at=4.7, hold_result=1.6)
        card("Pick your fighter", 3.2, sub=[("fly-boxing.vercel.app", ORANGE, 64), ("Orange or blue? Comment below", WHITE, 52)],
             bg=last_img[0])
        browser.close()
    srv.close()

    # ---- soundtrack -------------------------------------------------------------
    total = frame_no[0] / FPS
    audio = beat_track(total, rng=rng) * 0.35
    bank = {"punch": lambda: sfx_punch(rng), "block": lambda: sfx_block(rng), "whoosh": lambda: sfx_whoosh(rng), "bell": sfx_bell}
    last_t = {}
    for t, kind in sorted(events):
        if kind == "bell3":
            for k in range(3):
                mix_in(audio, sfx_bell(), int((t + k * 0.22) * SR))
            continue
        if t - last_t.get(kind, -1) < 0.08:
            continue
        last_t[kind] = t
        mix_in(audio, bank[kind](), int(t * SR))
    audio = np.tanh(audio * 1.2) * 0.85
    wav = os.path.join(out_dir, "soundtrack.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype("<i2").tobytes())

    if args.preview:
        print(f"preview: {frame_no[0]} frames ({total:.1f}s), every 6th saved in {frames_dir}")
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "%05d.png"),
                    "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    shutil.copy(os.path.join(frames_dir, "00010.png"), os.path.join(out_dir, "cover.png"))
    print(f"wrote {args.out} ({total:.1f}s, {frame_no[0]} frames, {len(events)} sound cues)")


if __name__ == "__main__":
    main()
