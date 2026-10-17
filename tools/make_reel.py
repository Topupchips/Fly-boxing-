"""
Render a vertical (1080x1920) Instagram/TikTok reel of Fly Boxing with
anime-style fight edits.

    pip install playwright pillow numpy scipy && playwright install chromium
    python tools/make_reel.py --font Archivo-VF.ttf          # -> reel/fly_boxing_reel.mp4
    python tools/make_reel.py --shots hook,vs --preview      # quick look at some shots

Drives the real viewer (docs/index.html?capture) frame by frame, so every
shot is the actual simulation: the recorded fights, the 3D ring and the
FlyWire brain view. On top of that it adds anime-style edits aimed at the
real action (impact frames and hit-stop freezes on clean hits, speed lines
on lunges, afterimages on dodges, a power-up aura, a VS screen and a K.O.
slam) and an original synthesized soundtrack with every sound on its frame.
See tools/anime_fx.py and tools/anime_audio.py.

--font: a TTF/OTF for captions. The Archivo variable font from
@fontsource-variable/archivo (converted to TTF) matches the site.
"""
from __future__ import annotations

import argparse
import collections
import functools
import http.server
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import threading
import wave

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import anime_audio  # noqa: E402
import anime_fx as fx  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, FPS = 1080, 1920, 30
STAGE_Y = 400                     # ring render occupies 400..1480
ORANGE, BLUE, WHITE, MUTED = (255, 122, 26), (46, 155, 214), (242, 241, 238), (140, 144, 152)
GOLD = (255, 206, 64)

CAPTURE_CSS = """
html, body { background:#000 !important; overflow:hidden !important; }
body * { visibility:hidden !important; }
.stage, .stage *, .b3, .b3 * { visibility:visible !important; }
.overlay { transition:none !important; }
.result { display:none !important; }
.ns.idle { opacity:1 !important; }
.stage { position:fixed !important; left:0; top:0; width:1080px !important; height:1080px !important;
         aspect-ratio:auto !important; border-radius:0 !important; z-index:10; }
.views, .hint { display:none !important; }
#b3A, #b3Z { position:fixed !important; left:0; width:1080px !important; height:400px !important;
             aspect-ratio:auto !important; border-radius:0 !important; z-index:10; margin:0 !important; }
#b3A { top:1080px } #b3Z { top:1480px }
.banner b { font-size:150px !important; }
.b3 .cap { display:none !important; }
"""


# ---------------------------------------------------------------- text
class Type:
    def __init__(self, path):
        self.path = path

    @functools.lru_cache(maxsize=96)
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
    """Big centred caption with a heavy outline; pop scales it for an entrance."""
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
    d.text(((W - d.textlength(clock, font=cf)) / 2, y - 6), clock, font=cf, fill=WHITE)
    rf = T.font(26, 500, 100)
    rt = f"Round {rd + 1} of 3"
    d.text(((W - d.textlength(rt, font=rf)) / 2, y + 66), rt, font=rf, fill=MUTED)
    bw, by = 380, y + 74
    for s, x0, col in ((0, 48, ORANGE), (1, W - 48 - bw, BLUE)):
        d.rounded_rectangle((x0, by, x0 + bw, by + 16), 8, fill=(30, 33, 38))
        hp = max(0.0, fr[s]["hp"]) / 100
        if hp > 0:
            box = (x0, by, x0 + bw * hp, by + 16) if s == 0 else (x0 + bw * (1 - hp), by, x0 + bw, by + 16)
            d.rounded_rectangle(box, 8, fill=col)
    tag = T.font(30, 600, 100)
    t = "fly-boxing.vercel.app"
    d.text(((W - d.textlength(t, font=tag)) / 2, 1850), t, font=tag, fill=MUTED)


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


SHOTS = ["cold", "freeze", "facecam", "gen0", "powerup", "brains", "vs", "title", "end", "loop"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--font", default="")
    ap.add_argument("--out", default=os.path.join(ROOT, "reel", "fly_boxing_reel.mp4"))
    ap.add_argument("--preview", action="store_true", help="save every 6th frame only and skip encoding")
    ap.add_argument("--shots", default=",".join(SHOTS), help="comma list; default cut: " + ",".join(SHOTS) + " (also: hook, spar)")
    ap.add_argument("--plan", default="", help="JSON list of B-roll shots with durations (voiceover cut); "
                    "renders frames + broll.json for tools/make_vo_reel.py instead of a finished reel")
    args = ap.parse_args()
    shots = [s for s in args.shots.split(",") if s]
    from playwright.sync_api import sync_playwright

    T = Type(args.font)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    frames_dir = os.path.join(out_dir, "frames")
    shutil.rmtree(frames_dir, ignore_errors=True)
    os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.default_rng(7)
    events, marks, ducks = [], [], []     # sound cues, music sections, hit-stop spans
    timeline = []                         # (time, shot name) for the voiceover sheet
    frame_no = [0]
    last_img = [None]
    cover = [None]
    first_img = [None]
    ui = {"tag": None, "freeze_src": None}      # tagline drawn on every frame while set; best hit-stop frame
    words = itertools.cycle(["WHAM!", "BAM!", "SMASH!", "POW!"])

    def now():
        return frame_no[0] / FPS

    def emit(img):
        if not args.preview or frame_no[0] % 6 == 0:
            img.convert("RGB").save(os.path.join(frames_dir, f"{frame_no[0]:05d}.png"), compress_level=1)
        if first_img[0] is None:
            first_img[0] = img
        frame_no[0] += 1
        last_img[0] = img
        if frame_no[0] % 60 == 0:
            print(f"  {frame_no[0]} frames ({now():.0f}s)", flush=True)

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

        def shot_png(y, h):
            return Image.open(io.BytesIO(page.screenshot(clip={"x": 0, "y": y, "width": 1080, "height": h}))).convert("RGBA")

        def stage_img():
            return shot_png(0, 1080)

        def brains_img():
            return shot_png(1080, 800)

        def project():
            """Screen position (x, y) and apparent radius of each fly in the ring render."""
            return js("""(()=>{const cam=flybox.camera, half=Math.tan(cam.fov*Math.PI/360), out=[];
              for(const f of flybox.flies){const p=f.position.clone(); p.y+=1.15+f.userData.body.position.y*1.9;
                const d=cam.position.distanceTo(p), c=p.clone().project(cam);
                out.push([(c.x+1)*540,(1-c.y)*540, 2.6/(d*half)*540]);} return out;})()""")

        def cinematic(mind=8.0, maxd=17.0, height=3.6, look=1.0, snap=False, k=0.16):
            js(f"""(()=>{{const a=flybox.flies[0].position,b=flybox.flies[1].position;
              const mx=(a.x+b.x)/2, mz=(a.z+b.z)/2, dx=b.x-a.x, dz=b.z-a.z, L=Math.hypot(dx,dz)||1;
              const C=window.__cine||(window.__cine={{}});
              const dist=Math.min({maxd},Math.max({mind},L*1.1+5));
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

        def calm_frame(fight, s, start=0):
            """First frame at/after start where fly s has been standing still-ish (no lunge/dodge, off the ropes)."""
            return js(f"""(()=>{{const F=FLYBOX.fights[{fight}].frames;
              for(let i={start};i<F.length;i++){{let ok=true;
                for(let j=Math.max(0,i-10);j<=i;j++){{const a=F[j][{s}]; if(a.d||a.l||Math.abs(a.x)>9.5||Math.abs(a.y)>9.5) ok=false;}}
                if(ok) return i;}} return {start};}})()""")

        def hold_on(fight, frame_i):
            """Seek, pause and let poses (hop, rear-up) settle."""
            js(f"window.__fi={fight}; flybox.load({fight}); flybox.seek({frame_i}); flybox.play(false); flybox.noTween()")
            for _ in range(8):
                js("flybox.tick(1/30)")

        def face_camera(s, dist=6.5, height=2.4, side=2.2, look=1.1):
            """Camera in front of fly s, three-quarter view, kept inside the ropes."""
            js(f"""(()=>{{const f=flybox.flies[{s}], h=f.rotation.y, fx=Math.cos(h), fz=-Math.sin(h), c=flybox.controls;
              c.minDistance=3; c.target.set(f.position.x, {look}, f.position.z);
              let cx=f.position.x+fx*{dist}+fz*{side}, cz=f.position.z+fz*{dist}-fx*{side};
              cx=Math.max(-13.2,Math.min(13.2,cx)); cz=Math.max(-13.2,Math.min(13.2,cz));
              flybox.camera.position.set(cx,{height},cz); c.update();}})()""")

        def frames_between(i0, i1):
            return js(f"FLYBOX.fights[window.__fi].frames.slice({i0},{i1}+1)")

        def detect(fr):
            out = []
            for a, b in zip(fr[:-1], fr[1:]):
                for s in (0, 1):
                    if b[s]["hit"] and not a[s]["hit"]:
                        out.append(("block" if b[s]["blk"] else "clean", s, 1 - s))
                    if b[s]["d"] and not a[s]["d"]:
                        out.append(("dodge", s, s))
                    if b[s]["l"] and not a[s]["l"]:
                        out.append(("lunge", s, s))
            return out

        def compose(stage, st, cap, pop=1.0, bottom="score", brains=None):
            img = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            if bottom == "brains":
                img.alpha_composite(stage.crop((0, 200, 1080, 920)), (0, STAGE_Y))
                img.alpha_composite(brains, (0, 1120))
                d = ImageDraw.Draw(img)
                lab = T.font(40, 800, 85)
                d.text((40, 1136), f"{st['names'][0]}'s brain", font=lab, fill=ORANGE)
                d.text((40, 1536), f"{st['names'][1]}'s brain", font=lab, fill=BLUE)
            else:
                img.alpha_composite(stage, (0, STAGE_Y))
                if st is not None and bottom == "score":
                    scoreboard(img, T, st)
            if cap:
                caption(img, T, cap, 200, pop=pop)
            if ui["tag"]:
                caption(img, T, ui["tag"], 255, size=58, weight=850, max_w=1020)
            return img

        def hitstop(stage, st, cap, focus, attacker, big=False, bottom="score"):
            """Impact frames, a white flash, then a frozen shaking punch-in with a slammed word."""
            tint = ORANGE if attacker == 0 else BLUE
            t0 = now()
            events.append((t0, "ko" if big else "impact"))
            seq = [fx.impact_frame(stage, focus, rng, True, tint), fx.impact_frame(stage, focus, rng, False)]
            if big:
                seq += [fx.impact_frame(stage, focus, rng, True, (255, 255, 255)), fx.impact_frame(stage, focus, rng, False)]
            for im in seq:
                emit(compose(im, st, cap, bottom=bottom))
            emit(compose(Image.blend(stage, Image.new("RGBA", stage.size, (255, 255, 255, 255)), 0.75), st, cap, bottom=bottom))
            hold = 26 if big else 10
            word = "K.O.!!" if big else next(words)
            slam = fx.slam_text(T.font(210 if big else 150, 900, 70), word, GOLD if big else WHITE, back=tint, angle=-6 if big else rng.uniform(-10, 4))
            for j in range(hold):
                pr = j / hold
                zoom = (1.32 if big else 1.2) - pr * (0.14 if big else 0.08)
                amp = (28 if big else 16) * (1 - pr) ** 1.5
                im, f2 = fx.punch_in(stage, zoom, focus, rng.uniform(-amp, amp), rng.uniform(-amp, amp))
                im = fx.chroma(im, (16 if big else 9) * (1 - pr))
                im.alpha_composite(fx.speed_lines(f2, 260 if big else 200, rng, n=100, alpha=int(170 * (1 - pr * 0.5))))
                scale = 1.7 - min(1.0, j / 3) * 0.7
                spot = (540, 520) if big else (min(820, max(260, f2[0])), max(200, f2[1] - 230))
                fx.paste_center(im, slam, spot, scale)
                frame = compose(im, st, cap, bottom=bottom)
                if j == 4 and not big:
                    ui["freeze_src"] = frame
                emit(frame)
            ducks.append((t0, now()))

        def segment(name, fight, start, seconds, speed, cam, cap, bottom="score", cap2=None, cap2_at=None,
                    cap2_on_ko=False, budget=0, lines=True, ghosts=True, qmarks=False, ko_hold=0.0, tagline=None):
            timeline.append((now(), name))
            js(f"window.__fi={fight}; flybox.load({fight}); flybox.seek({start}); flybox.setSpeed({speed}); flybox.play(true); flybox.controls.minDistance=8")
            js("flybox.tick(1/30)")
            cam_kw = {"close": {}, "medium": {"mind": 10.5, "maxd": 18.0, "height": 5.2, "k": 0.22},
                      "wide": {"mind": 12.0, "maxd": 21.0, "height": 7.0}}[cam]
            js("flybox.noTween()")
            js("flybox.tick(1/30)")
            cinematic(snap=True, **cam_kw)
            prev, cur_cap, k0, kod = start, cap, 0, False
            ui["tag"] = None
            buf = collections.deque(maxlen=3)
            for k in range(int(seconds * FPS)):
                cinematic(**cam_kw)
                js("flybox.tick(1/30)")
                if tagline and k == 6:
                    ui["tag"] = tagline
                st = js("flybox.state()")
                cur = int(st["t"])
                hits = detect(frames_between(prev, cur)) if cur > prev else []
                prev = cur
                if cap2 and cap2_at is not None and k == int(cap2_at * FPS):
                    cur_cap, k0 = cap2, k
                stage = stage_img()
                raw = stage.copy()
                pos = project()
                fr = st["frame"]
                clean = [h for h in hits if h[0] == "clean"]
                ko_hit = next((h for h in clean if fr[h[1]]["hp"] <= 0), None)
                for kind, s, a in hits:
                    if kind == "block":
                        events.append((now(), "block"))
                    elif kind == "dodge":
                        events.append((now(), "zip"))
                    elif kind == "lunge":
                        events.append((now(), "whoosh"))
                    elif kind == "clean" and not (ko_hit or budget > 0):
                        events.append((now(), "punch"))
                if ghosts and (fr[0]["d"] or fr[1]["d"]) and buf:
                    for j, old in enumerate(reversed(buf)):
                        stage = Image.blend(stage, old, 0.3 / (j + 1))
                buf.append(raw)
                if lines:
                    for s in (0, 1):
                        if fr[s]["l"]:
                            stage.alpha_composite(fx.speed_lines(pos[s][:2], pos[s][2] * 1.5, rng, n=60, alpha=105))
                if qmarks:
                    for s in (0, 1):
                        x, y, r = pos[s]
                        q = fx.slam_text(T.font(int(max(44, r * 0.75)), 900, 80), "???", WHITE, angle=12 if s else -12, shear=0)
                        fx.paste_center(stage, q, (x, y - r * 1.25 + 8 * np.sin(k * 0.35 + s * 2)))
                if bottom == "brains":
                    frame_img = compose(stage, st, cur_cap, min(1.0, 0.82 + (k - k0) / 6 * 0.18), "brains", brains_img())
                else:
                    frame_img = compose(stage, st, cur_cap, min(1.0, 0.82 + (k - k0) / 6 * 0.18), bottom)
                emit(frame_img)
                if ko_hit:
                    if cap2_on_ko:
                        cur_cap = cap2
                    hitstop(raw, st, cur_cap, pos[ko_hit[1]][:2], ko_hit[2], big=True)
                    kod = True
                elif clean and budget > 0:
                    budget -= 1
                    hitstop(raw, st, cur_cap, pos[clean[0][1]][:2], clean[0][2], bottom=bottom)
                if kod or (not st["playing"] and st["t"] >= st["n"] - 1):
                    break
            ui["tag"] = None
            if kod and ko_hold > 0:
                st = js("flybox.state()")
                win = st["winner"]
                col = ORANGE if win == 0 else BLUE
                pos = project()
                events.append((now(), "bell3"))
                ko_word = fx.slam_text(T.font(170, 900, 72), "KNOCKOUT", col, back=WHITE, angle=-5)
                for j in range(int(ko_hold * FPS)):
                    glow = fx.aura(pos[win][:2], pos[win][2] * 1.05, col, j / FPS, rng, strength=0.95)
                    im = fx.add_glow(raw, glow, dim=0.5)
                    fx.paste_center(im, ko_word, (540, 470), 1.5 - min(1.0, j / 4) * 0.5)
                    sub = fx.slam_text(T.font(64, 850, 80), f"{st['names'][win]} wins", WHITE, angle=-5, shear=0.15)
                    fx.paste_center(im, sub, (540, 640))
                    emit(compose(im, st, cur_cap))

        def powerup(name, fight, frame_i, fly, seconds, cap):
            timeline.append((now(), name))
            marks.append((now(), "off"))
            events.append((now(), "riser", seconds))
            hold_on(fight, calm_frame(fight, fly, frame_i))
            face_camera(fly, dist=8.0, height=2.6, side=2.4)
            js("flybox.tick(1/30)")
            base = stage_img()
            st = js("flybox.state()")
            x, y, r = project()[fly]
            parts = []
            n = int(seconds * FPS)
            for k in range(n):
                pr = k / n
                glow = fx.aura((x, y), min(360, r * (0.7 + 0.45 * pr)), ORANGE, k / FPS, rng, strength=min(0.85, 0.25 + pr))
                for _ in range(5):
                    parts.append([x + rng.uniform(-r, r), y + rng.uniform(0, r * 0.7), rng.uniform(7, 16), 0])
                d = ImageDraw.Draw(glow)
                for pt in parts:
                    pt[1] -= pt[2]
                    pt[3] += 1
                    rad = 3 + 3 * (1 - pt[3] / 20)
                    d.ellipse((pt[0] - rad, pt[1] - rad, pt[0] + rad, pt[1] + rad), fill=(255, 220, 160))
                parts = [pt for pt in parts if pt[3] < 20]
                im = fx.add_glow(base, glow, dim=1 - 0.6 * min(1.0, pr * 2.2))
                amp = 10 * pr ** 2
                im, f2 = fx.punch_in(im, 1.02 + 0.16 * pr, (x, y), rng.uniform(-amp, amp), rng.uniform(-amp, amp), pull=0.8)
                if pr > 0.65:
                    im.alpha_composite(fx.speed_lines(f2, r * 2.2, rng, n=70, alpha=int(200 * (pr - 0.65) / 0.35)))
                emit(compose(im, st, cap, min(1.0, 0.82 + k / 6 * 0.18), bottom="none"))
            events.append((now(), "burst"))
            marks.append((now(), "full"))
            for a in (0.85, 0.45):
                emit(compose(Image.blend(im, Image.new("RGBA", im.size, (255, 255, 255, 255)), a), st, cap, bottom="none"))

        def vs_screen(name, fight, frame_i, seconds, labels):
            timeline.append((now(), name))
            portraits = []
            for s in (0, 1):
                hold_on(fight, calm_frame(fight, s, frame_i))
                face_camera(s, dist=7.0, height=2.3, side=2.0)
                js("flybox.tick(1/30)")
                img = stage_img()
                x, y, r = project()[s]
                half = min(540, r * 2.5)
                cx = min(max(x, half), 1080 - half)
                cy = min(max(y - half * 0.05, half), 1080 - half)
                box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
                portraits.append(img.crop(box).resize((1000, 1000), Image.BICUBIC))
            names = js("flybox.state()")["names"]
            polyA = [(0, 0), (W, 0), (W, 840), (0, 1000)]
            polyB = [(0, 1080), (W, 920), (W, H), (0, H)]

            def panel(poly, portrait, center, tint):
                layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                bg = Image.new("RGBA", (W, H), tuple(int(c * 0.35) for c in tint) + (255,))
                bg.alpha_composite(portrait, (int(center[0] - 500), int(center[1] - 500)))
                wash = Image.new("RGBA", (W, H), tint + (60,))
                bg.alpha_composite(wash)
                mask = Image.new("L", (W, H), 0)
                ImageDraw.Draw(mask).polygon(poly, fill=255)
                layer.paste(bg, (0, 0), mask)
                return layer

            pa = panel(polyA, portraits[0], (540, 450), ORANGE)
            pb = panel(polyB, portraits[1], (540, 1450), BLUE)
            dA = ImageDraw.Draw(pa)
            dB = ImageDraw.Draw(pb)
            nf, lf = T.font(96, 900, 72), T.font(44, 800, 90)
            dA.text((56, 650), labels[0], font=lf, fill=ORANGE, stroke_width=4, stroke_fill=(0, 0, 0))
            dA.text((56, 700), names[0], font=nf, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
            wl, wn = dB.textlength(labels[1], font=lf), dB.textlength(names[1], font=nf)
            dB.text((W - 56 - wl, 1150), labels[1], font=lf, fill=BLUE, stroke_width=4, stroke_fill=(0, 0, 0))
            dB.text((W - 56 - wn, 1200), names[1], font=nf, fill=WHITE, stroke_width=6, stroke_fill=(0, 0, 0))
            vs = fx.slam_text(T.font(300, 900, 70), "VS", WHITE, back=GOLD, angle=-8)
            title = fx.slam_text(T.font(66, 900, 78), "TITLE FIGHT", WHITE, angle=0, shear=0.18)
            events.append((now(), "shing"))
            n = int(seconds * FPS)
            for k in range(n):
                e = 1 - (1 - min(1.0, k / 5)) ** 3
                canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                canvas.alpha_composite(pa, (int(-W * (1 - e)), 0))
                canvas.alpha_composite(pb, (int(W * (1 - e)), 0))
                if k >= 5:
                    if k == 5:
                        events.append((now(), "slam"))
                    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
                    gd = ImageDraw.Draw(glow)
                    pts = fx.lightning(gd, (-20, 1040), (W + 20, 880), rng, 34, GOLD + (255,), jitter=30)
                    glow = glow.filter(ImageFilter.GaussianBlur(14))
                    canvas.alpha_composite(glow)
                    ImageDraw.Draw(canvas).line(pts, fill=(255, 255, 255, 255), width=9, joint="curve")
                    fx.paste_center(canvas, vs, (540, 960), 2.2 - min(1.0, (k - 5) / 4) * 1.2)
                    fx.paste_center(canvas, title, (540, 90))
                    if k in (5, 6):
                        canvas = Image.blend(canvas, Image.new("RGBA", (W, H), (255, 255, 255, 255)), 0.7 if k == 5 else 0.3)
                    amp = 14 * max(0.0, 1 - (k - 5) / 10)
                    if amp > 0.5:
                        shaken = Image.new("RGBA", (W, H), (0, 0, 0, 255))
                        shaken.alpha_composite(canvas, (int(rng.uniform(-amp, amp)), int(rng.uniform(-amp, amp))))
                        canvas = shaken
                if k == 20:
                    cover[0] = canvas.copy()
                emit(canvas)

        def freeze(name, seconds, cap):
            """Record scratch: freeze the last frame, glitch it back, drain the colour, explain."""
            timeline.append((now(), name))
            marks.append((now(), "off"))
            events.append((now(), "scratch"))
            base = (ui["freeze_src"] or last_img[0]).copy()
            gray = base.convert("L").convert("RGBA")
            n = int(seconds * FPS)
            for k in range(n):
                pr = k / n
                im = Image.blend(base, gray, min(1.0, pr * 2.5) * 0.85)
                if k < 4:                                         # rewind glitch: shifted bands
                    arr = np.array(im)
                    for _ in range(6):
                        y0 = int(rng.uniform(0, H - 120))
                        h = int(rng.uniform(20, 120))
                        arr[y0:y0 + h] = np.roll(arr[y0:y0 + h], int(rng.uniform(-80, 80)), axis=1)
                    im = Image.fromarray(arr)
                zoom = 1 + 0.05 * pr
                bw, bh = W / zoom, H / zoom
                im = im.resize((W, H), Image.BICUBIC, box=((W - bw) / 2, (H - bh) / 2, (W + bw) / 2, (H + bh) / 2))
                im.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, int(90 * min(1.0, pr * 3)))))
                if k >= 3 and cap:
                    caption(im, T, cap, 960, size=92, pop=min(1.0, 0.8 + (k - 3) / 5 * 0.2))
                emit(im)

        def brainspin(name, fight, start, seconds):
            """Full-frame shot of the real FlyWire brain, spinning and lighting up as a fight plays."""
            timeline.append((now(), name))
            js(f"window.__fi={fight}; flybox.load({fight}); flybox.seek({start}); flybox.setSpeed(1); flybox.play(true)")
            js("""(()=>{const b=document.getElementById('b3A'); b.dataset.css=b.getAttribute('style')||'';
                  b.setAttribute('style','position:fixed!important;left:0;top:0;width:1080px!important;height:1080px!important;'
                    +'aspect-ratio:auto!important;z-index:40;margin:0!important;border-radius:0!important');})()""")
            page.wait_for_timeout(400)
            for _ in range(3):
                js("flybox.tick(1/30)")
            st = js("flybox.state()")
            for k in range(int(seconds * FPS)):
                js("flybox.tick(1/30)")
                img = stage_img().convert("RGB")
                img = ImageEnhance.Contrast(ImageEnhance.Brightness(img).enhance(2.3)).enhance(1.25)
                glow = img.filter(ImageFilter.GaussianBlur(9))
                img = ImageChops.add(img, ImageEnhance.Brightness(glow).enhance(0.9)).convert("RGBA")
                im, _ = fx.punch_in(img, 1.06 + 0.1 * k / max(1, seconds * FPS), (540, 540))
                emit(compose(im, st, None, bottom="none"))
            js("(()=>{const b=document.getElementById('b3A'); b.setAttribute('style', b.dataset.css||'');})()")
            page.wait_for_timeout(300)

        def loop_back(name, frames=8):
            """Cross-fade the last frame into the first so the reel replays seamlessly."""
            timeline.append((now(), name))
            a, b = last_img[0], first_img[0]
            for k in range(frames):
                emit(Image.blend(a, b, (k + 1) / frames))

        def end_card(name, seconds):
            timeline.append((now(), name))
            marks.append((now(), "outro"))
            events.append((now(), "shing"))
            bg = Image.new("RGBA", (W, H), (0, 0, 0, 255))
            if last_img[0] is not None:
                bg.alpha_composite(last_img[0].filter(ImageFilter.GaussianBlur(16)))
            bg.alpha_composite(Image.new("RGBA", (W, H), (0, 0, 0, 185)))
            word = fx.slam_text(T.font(250, 900, 72), "BASE44", ORANGE, back=WHITE, angle=-6)
            for k in range(int(seconds * FPS)):
                im = bg.copy()
                caption(im, T, "Comment", 700, size=110, pop=min(1.0, 0.8 + k / 6 * 0.2))
                if k >= 4:
                    if k == 4:
                        events.append((now(), "slam"))
                    amp = 12 * max(0.0, 1 - (k - 4) / 8)
                    fx.paste_center(im, word, (540 + rng.uniform(-amp, amp), 920 + rng.uniform(-amp, amp)), 2.0 - min(1.0, (k - 4) / 4))
                if k >= 10:
                    caption(im, T, "and I'll send you the exact workflow to build something like this", 1200, size=60, weight=800, max_w=900)
                    caption(im, T, "fly-boxing.vercel.app", 1720, size=44, color=MUTED, weight=700)
                emit(im)

        # ---- find the moments --------------------------------------------------
        fights = js("FLYBOX.fights.map(f=>({title:f.title,n:f.frames.length,winner:f.winner,ko:f.ko}))")
        idx = {f["title"]: i for i, f in enumerate(fights)}
        title, spar, newb = idx["Title Fight"], idx["Sparring Session"], idx["Newborns"]
        hit_t = js(f"(()=>{{const f=FLYBOX.fights[{title}].frames; for(let i=400;i<f.length;i++) if(f[i][1].hit&&!f[i][1].blk&&!f[i][0].hit) return i; return 300;}})()")
        spar_hit = js(f"(()=>{{const f=FLYBOX.fights[{spar}].frames; for(let i=30;i<f.length;i++) if(f[i][1].hit&&!f[i][1].blk) return i; return 60;}})()")
        n_title = fights[title]["n"]

        # ---- the reel ----------------------------------------------------------
        marks.append((0.0, "full"))
        plan = {
            # 0-3s: cinematic cold open, no explanation, one tagline
            "cold": lambda: segment("cold", title, hit_t - 8, 2.1, 0.3, "close", None, bottom="none", budget=1,
                                    tagline="fruit flies. real brains."),
            # record scratch + freeze
            "freeze": lambda: freeze("freeze", 1.4, "You're probably wondering how fruit flies learned to box"),
            # your face-cam goes here; clean slow-mo fight footage underneath, music as a quiet bed
            "facecam": lambda: (marks.append((now(), "bed")),
                                segment("facecam", spar, max(0, spar_hit - 20), 5.0, 0.6, "medium", None, bottom="none",
                                        lines=True)),
            "gen0": lambda: (marks.append((now(), "muffled")), events.append((now() + 0.1, "wah")),
                             segment("gen0", newb, 40, 2.8, 2, "wide", "Generation 0: no idea what it's doing",
                                     lines=False, ghosts=False, qmarks=True)),
            "powerup": lambda: powerup("powerup", spar, 2, 0, 2.4, "150 generations of evolution later..."),
            "brains": lambda: segment("brains", title, 200, 4.4, 1, "wide", "Every punch is real neurons firing in a real fly brain map",
                                      bottom="brains", lines=False, ghosts=False),
            "vs": lambda: vs_screen("vs", title, 6, 1.8, ["CHAMPION", "#2 CONTENDER"]),
            "title": lambda: segment("title", title, max(0, n_title - 1 - 150), 6.0, 1, "close", "Title fight: champion vs #2 contender",
                                     cap2="...and the champion got KNOCKED OUT", cap2_on_ko=True, budget=1, ko_hold=1.7),
            "end": lambda: end_card("end", 4.6),
            "loop": lambda: loop_back("loop"),
            # extra shots kept for other cuts
            "hook": lambda: segment("hook", title, hit_t - 14, 3.0, 0.3, "close",
                                    "I wired fruit flies with real brain data and taught them to box", budget=1),
            "spar": lambda: segment("spar", spar, max(0, spar_hit - 45), 5.4, 1, "medium", "Now they lunge, block and dodge",
                                    cap2="Nobody coded a single punch", cap2_at=2.8, budget=2),
        }
        if args.plan:
            spec = json.load(open(args.plan))
            for sh in spec:
                kind, dur = sh["shot"], sh.get("dur", 0)
                if kind == "cold":
                    segment("cold", title, hit_t - 8, dur, 0.3, "close", None, bottom="none", budget=1, tagline=sh.get("tagline"))
                elif kind == "freeze":
                    freeze("freeze", dur, None)
                elif kind == "brainspin":
                    brainspin("brainspin", title, 220, dur)
                elif kind == "fightcut":
                    segment("fightcut", spar, max(0, spar_hit - 24), dur, 1, "close", None, bottom="none", budget=0)
                elif kind == "gen0":
                    events.append((now() + 0.1, "wah"))
                    segment("gen0", newb, 40, dur, 2, "wide", None, bottom="none", lines=False, ghosts=False, qmarks=True)
                elif kind == "powerup":
                    powerup("powerup", spar, 2, 0, dur, None)
                elif kind == "brains":
                    segment("brains", title, 200, dur, 1, "wide", None, bottom="brains", lines=False, ghosts=False)
                elif kind == "vs":
                    vs_screen("vs", title, 6, dur, ["CHAMPION", "#2 CONTENDER"])
                elif kind == "title":
                    lead = sh["lead"]
                    segment("title", title, max(0, n_title - 1 - int(lead * FPS)), lead + 1.0, 1, "close", None,
                            bottom="none", budget=0, ko_hold=sh.get("ko_hold", 1.7))
                else:
                    raise SystemExit(f"unknown shot {kind}")
            shots = []
        for s in shots:
            before = len(events)
            plan[s]()
            if s == "title":
                ko = next((e[0] for e in events[before:] if e[1] == "ko"), None)
                if ko is not None:
                    marks.append((ko, "off"))
        browser.close()
    srv.close()

    # ---- soundtrack + encode ----------------------------------------------------
    total = frame_no[0] / FPS
    if args.plan:
        json.dump({"frames": frame_no[0], "fps": FPS, "timeline": timeline, "events": events, "ducks": ducks},
                  open(os.path.join(out_dir, "broll.json"), "w"), indent=1)
        print(f"B-roll: {frame_no[0]} frames ({total:.1f}s) -> {frames_dir}")
        return
    audio = anime_audio.build_soundtrack(total, events, marks, ducks, rng)
    wav = os.path.join(out_dir, "soundtrack.wav")
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(anime_audio.SR)
        w.writeframes((audio * 32767).astype("<i2").tobytes())
    with open(os.path.join(out_dir, "timeline.txt"), "w") as f:
        for t, name in timeline:
            f.write(f"{t:6.2f}s  {name}\n")
        f.write(f"{total:6.2f}s  end\n")
    if cover[0] is not None:
        cover[0].convert("RGB").save(os.path.join(out_dir, "cover.png"))
    if args.preview:
        print(f"preview: {frame_no[0]} frames ({total:.1f}s), every 6th saved in {frames_dir}")
        return
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", os.path.join(frames_dir, "%05d.png"),
                    "-i", wav, "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", args.out], check=True)
    print(f"wrote {args.out} ({total:.1f}s, {frame_no[0]} frames, {len(events)} sound cues)")


if __name__ == "__main__":
    main()
