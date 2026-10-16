"""
Anime-style fight effects for the Fly Boxing reel.

All effects work on the 1080x1080 ring render (RGBA PIL images) and take
screen positions of the flies, so they can be aimed at the real action:

  speed_lines   radial focus lines (lunges, hit-stops, power-ups)
  impact_frame  the high-contrast inverted flash frame on a big hit
  aura          flickering energy flames around a fighter
  punch_in      zoom + shake toward a point
  chroma        red/blue channel split for impact wobble
  slam_text     big slanted onomatopoeia ("WHAM!", "K.O.!!")
  lightning     jagged bolt for the VS screen divider

Everything here is drawn from scratch; no frames, logos or artwork from any
existing show are used.
"""
from __future__ import annotations

import math

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps

S = 1080


def speed_lines(center, inner, rng, n=80, alpha=160, color=(255, 255, 255), size=S):
    """Radial focus lines that leave a clear zone of radius ~inner around center."""
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    cx, cy = center
    far = size * 1.6
    for _ in range(n):
        a = rng.uniform(0, 2 * math.pi)
        hw = rng.uniform(0.003, 0.016)
        r0 = inner * rng.uniform(1.0, 1.6)
        tip = (cx + math.cos(a) * r0, cy + math.sin(a) * r0)
        p1 = (cx + math.cos(a - hw) * far, cy + math.sin(a - hw) * far)
        p2 = (cx + math.cos(a + hw) * far, cy + math.sin(a + hw) * far)
        d.polygon([tip, p1, p2], fill=tuple(color) + (int(alpha * rng.uniform(0.45, 1.0)),))
    return layer


def impact_frame(stage, center, rng, invert=True, tint=(255, 255, 255)):
    """Posterized, optionally inverted flash frame with heavy focus lines."""
    g = ImageOps.autocontrast(ImageOps.grayscale(stage.convert("RGB")), cutoff=2)
    g = g.point(lambda v: 0 if v < 85 else (110 if v < 165 else 255))
    if invert:
        g = ImageOps.invert(g)
        img = ImageOps.colorize(g, black=(0, 0, 0), white=tint).convert("RGBA")
        line_col = (255, 255, 255)
    else:
        img = Image.merge("RGB", (g, g, g)).convert("RGBA")
        line_col = (0, 0, 0)
    img.alpha_composite(speed_lines(center, 120, rng, n=120, alpha=255, color=line_col))
    return img


def aura(center, radius, color, t, rng, strength=1.0, size=S):
    """Flickering flame aura (RGB, black background) for additive blending."""
    cx, cy = center
    layer = Image.new("RGB", (size, size), (0, 0, 0))
    d = ImageDraw.Draw(layer)
    col = tuple(int(c * strength) for c in color)
    hot = tuple(min(255, int((c * 0.45 + 255 * 0.55) * strength)) for c in color)
    for k in range(38):
        a = rng.uniform(0, 2 * math.pi)
        base_r = radius * rng.uniform(0.6, 0.95)
        bx, by = cx + math.cos(a) * base_r, cy + math.sin(a) * base_r * 0.75
        rise = 1.5 if math.sin(a) < 0.2 else 0.7          # flames on top climb higher
        length = radius * rng.uniform(0.45, 1.25) * rise
        dx, dy = math.cos(a) * 0.4 + math.sin(t * 17 + k) * 0.18, -1.0
        n = math.hypot(dx, dy)
        dx, dy = dx / n, dy / n
        tip = (bx + dx * length, by + dy * length)
        hw = radius * rng.uniform(0.07, 0.15)
        d.polygon([(bx - dy * hw, by + dx * hw), tip, (bx + dy * hw, by - dx * hw)], fill=col)
    d.ellipse((cx - radius, cy - radius * 0.8, cx + radius, cy + radius * 0.8), outline=hot, width=max(2, int(radius * 0.1)))
    glow = layer.filter(ImageFilter.GaussianBlur(max(2, radius * 0.14)))
    core = layer.filter(ImageFilter.GaussianBlur(max(1, radius * 0.035)))
    out = ImageChops.add(glow, ImageEnhance.Brightness(core).enhance(0.8))
    # keep the fighter visible: fade the aura out toward its centre
    hole = Image.new("L", (size, size), 255)
    ImageDraw.Draw(hole).ellipse((cx - radius * 0.55, cy - radius * 0.45, cx + radius * 0.55, cy + radius * 0.45), fill=40)
    hole = hole.filter(ImageFilter.GaussianBlur(max(2, radius * 0.18)))
    return Image.composite(out, Image.new("RGB", (size, size)), hole)


def add_glow(stage, glow_rgb, dim=1.0):
    """Darken the scene (power-up mood) and add a glow layer on top."""
    base = stage.convert("RGB")
    if dim < 1:
        base = ImageEnhance.Brightness(base).enhance(dim)
    return ImageChops.add(base, glow_rgb).convert("RGBA")


def punch_in(img, zoom, focus, dx=0.0, dy=0.0, pull=0.55, size=S):
    """Zoom toward focus (pulled part-way from centre) with a shake offset.
    Returns the image and where focus lands in it."""
    bw = size / zoom
    fx = size / 2 + (focus[0] - size / 2) * pull + dx
    fy = size / 2 + (focus[1] - size / 2) * pull + dy
    fx = min(max(fx, bw / 2), size - bw / 2)
    fy = min(max(fy, bw / 2), size - bw / 2)
    box = (fx - bw / 2, fy - bw / 2, fx + bw / 2, fy + bw / 2)
    out = img.resize((size, size), Image.BICUBIC, box=box)
    return out, ((focus[0] - box[0]) * zoom, (focus[1] - box[1]) * zoom)


def chroma(img, shift):
    """Red/blue channel split."""
    s = int(round(shift))
    if s < 1:
        return img
    r, g, b, a = img.convert("RGBA").split()
    return Image.merge("RGBA", (ImageChops.offset(r, s, 0), g, ImageChops.offset(b, -s, 0), a))


def slam_text(font, text, fill, stroke=(0, 0, 0), angle=-7, shear=0.22, back=None):
    """Big slanted text with a thick outline and optional coloured back-outline."""
    probe = ImageDraw.Draw(Image.new("RGBA", (8, 8)))
    size = font.size
    sw, bw = max(3, int(size * 0.09)), max(6, int(size * 0.17))
    bb = probe.textbbox((0, 0), text, font=font, stroke_width=bw)
    w, h = bb[2] - bb[0], bb[3] - bb[1]
    pad = int(size * 0.4)
    layer = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    org = (pad - bb[0], pad - bb[1])
    if back:
        d.text(org, text, font=font, fill=back, stroke_width=bw, stroke_fill=back)
    d.text(org, text, font=font, fill=fill, stroke_width=sw, stroke_fill=stroke)
    W_, H_ = layer.size
    layer = layer.transform((int(W_ + H_ * shear), H_), Image.AFFINE, (1, shear, -H_ * shear, 0, 1, 0), Image.BICUBIC)
    return layer.rotate(angle, expand=True, resample=Image.BICUBIC)


def paste_center(img, layer, center, scale=1.0):
    if scale != 1.0:
        layer = layer.resize((max(1, int(layer.width * scale)), max(1, int(layer.height * scale))), Image.BICUBIC)
    img.alpha_composite(layer, (int(center[0] - layer.width / 2), int(center[1] - layer.height / 2)))


def lightning(draw, p0, p1, rng, width, color, segs=16, jitter=38):
    pts = [p0]
    vx, vy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.hypot(vx, vy) or 1
    nx, ny = -vy / L, vx / L
    for i in range(1, segs):
        t = i / segs
        j = rng.uniform(-jitter, jitter)
        pts.append((p0[0] + vx * t + nx * j, p0[1] + vy * t + ny * j))
    pts.append(p1)
    draw.line(pts, fill=color, width=width, joint="curve")
    return pts
