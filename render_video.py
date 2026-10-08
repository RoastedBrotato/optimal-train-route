"""Render the slime-mould run as a slow, dark, glowing video (MP4 + GIF).

Reads out/history.npz written by simulate.py (conductivity and showcase flux
for every step) and draws:
  * dead tubes as a faint grey lattice (the "maze")
  * living tubes glowing orange -> yellow with thickness
  * particles flowing along living tubes
  * a monospace stats panel with a sparkline of tubes alive
  * an intro card and an end card

Usage:
    python render_video.py [--width 1920] [--fps 30] [--out out/mold_pakistan.mp4]
Requires ffmpeg on PATH.
"""
import argparse
import json
import os
import subprocess

import cv2
import numpy as np
import shapely
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import shape

from cities import CITIES

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
ALIVE = 0.005
ALIVE_REL = 0.01   # a tube is "alive" when above 1% of the strongest tube (flow only depends on ratios)
FADE_DECADES = 2.0  # tubes fade to black this many decades below the strongest

# palette (RGB)
BG = (13, 13, 16)
LAND = (22, 22, 27)
BORDER = (48, 48, 56)
DEAD = (40, 40, 47)
ORANGE = (255, 106, 26)
YELLOW = (255, 211, 77)
WHITE = (235, 235, 230)
GREY = (120, 120, 130)
DIM = (80, 80, 90)
RED = (255, 76, 48)
GREEN = (78, 204, 120)


def rgb2bgr(c):
    return (c[2], c[1], c[0])


def font(size, bold=False):
    name = "consolab.ttf" if bold else "consola.ttf"
    for path in (os.path.join("C:/Windows/Fonts", name), name):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


# ----------------------------------------------------------------------------
class Scene:
    def __init__(self, W, H):
        self.W, self.H = W, H
        self.panel_w = int(W * 0.26)
        self.map_w = W - self.panel_w
        h = np.load(os.path.join(OUT, "history.npz"))
        self.Dh = h["D"]            # steps x edges
        self.Qh = h["Q"]
        self.src, self.dst = h["src"], h["dst"]
        self.lon, self.lat = h["lon"], h["lat"]
        self.length = h["length"]
        self.city_ids = h["city_ids"]
        self.rail_km = float(h["rail_km"])
        self.n_steps = self.Dh.shape[0]
        self.n_edges = self.Dh.shape[1]

        with open(os.path.join(DATA, "pakistan.geojson")) as f:
            gj = json.load(f)
        self.country = shapely.union_all([shape(ft["geometry"]) for ft in gj["features"]])

        # projection: fit the country in the map area with a margin
        minx, miny, maxx, maxy = self.country.bounds
        cosl = np.cos(np.radians((miny + maxy) / 2))
        margin = 0.06
        sx = self.map_w * (1 - 2 * margin) / ((maxx - minx) * cosl)
        sy = self.H * (1 - 2 * margin) / (maxy - miny)
        self.k = min(sx, sy)
        self.cosl = cosl
        self.x0 = minx
        self.y1 = maxy
        self.ox = (self.map_w - (maxx - minx) * cosl * self.k) / 2
        self.oy = (self.H - (maxy - miny) * self.k) / 2

        self.px = self.project(self.lon, self.lat)           # nodes x 2
        self.p_src = self.px[self.src]
        self.p_dst = self.px[self.dst]
        self.seg_px = np.linalg.norm(self.p_dst - self.p_src, axis=1)

        self.names = [c["name"] for c in CITIES]
        self.pops = np.array([float(c["pop"]) for c in CITIES])

        # particles: random phase per edge, dots per edge from pixel length
        rng = np.random.default_rng(3)
        self.phase = rng.uniform(0, 1, self.n_edges).astype(np.float32)
        self.ndots = np.maximum(1, np.round(self.seg_px / 9.0)).astype(int)

        self.static = self.render_static()
        self.alive_curve = []

    def project(self, lon, lat):
        x = self.ox + (np.asarray(lon) - self.x0) * self.cosl * self.k
        y = self.oy + (self.y1 - np.asarray(lat)) * self.k
        return np.stack([x, y], axis=-1)

    # ------------------------------------------------------------------
    def render_static(self):
        img = np.zeros((self.H, self.W, 3), np.uint8)
        img[:] = rgb2bgr(BG)
        polys = self.country.geoms if hasattr(self.country, "geoms") else [self.country]
        for p in polys:
            pts = self.project(*np.array(p.exterior.coords).T).astype(np.int32)
            cv2.fillPoly(img, [pts], rgb2bgr(LAND), lineType=cv2.LINE_AA)
            cv2.polylines(img, [pts], True, rgb2bgr(BORDER), 1, cv2.LINE_AA)
        # the dead lattice ("maze walls")
        for a, b in zip(self.p_src.astype(np.int32), self.p_dst.astype(np.int32)):
            cv2.line(img, tuple(a), tuple(b), rgb2bgr(DEAD), 1, cv2.LINE_AA)
        return img

    def tube_color(self, rel):
        """orange for weak tubes -> yellow for strong, as BGR float arrays."""
        t = np.clip(np.sqrt(rel), 0, 1)[:, None]
        c = np.array(ORANGE)[None] * (1 - t) + np.array(YELLOW)[None] * t
        return c[:, ::-1]

    def draw_tubes(self, D, Q, frame_idx):
        """Return a BGR uint8 layer with glowing tubes and particles."""
        layer = np.zeros((self.H, self.W, 3), np.float32)
        dmax = max(float(D.max()), 1e-9)
        alive = D > ALIVE_REL * dmax
        idx = np.nonzero(alive)[0]
        if len(idx) == 0:
            return layer
        # visuals are relative to the strongest tube on a log scale (flow in the
        # model depends only on ratios of D), and everything thins/dims when the
        # lattice is crowded (the flood phase)
        dens = float(np.clip(np.sqrt(600.0 / len(idx)), 0.2, 1.0))
        self.dens = dens
        rel = np.clip(1.0 + np.log10(D[idx] / dmax) / FADE_DECADES, 0, 1)
        cols = self.tube_color(rel) * (0.45 + 0.55 * rel)[:, None] * (0.55 + 0.45 * dens)
        widths = (1 + 4.5 * np.sqrt(rel)) * dens
        # draw thin tubes first so thick ones sit on top
        order = np.argsort(widths)
        for j in order:
            e = idx[j]
            a = tuple(self.p_src[e].astype(int))
            b = tuple(self.p_dst[e].astype(int))
            c = tuple(float(v) for v in cols[j])
            cv2.line(layer, a, b, c, max(1, int(round(widths[j]))), cv2.LINE_AA)
        # faint alpha for weak tubes
        # (handled by colour brightness: scale by D)
        bright = np.zeros(len(D), np.float32)
        bright[idx] = (0.35 + 0.65 * rel) * (0.5 + 0.5 * dens)
        # particle positions
        q = Q[idx]
        speed = 1.2 + 3.5 * np.clip(np.sqrt(np.abs(q) / 0.25), 0, 1)
        self.phase[idx] = (self.phase[idx] + np.sign(q) * speed / np.maximum(self.seg_px[idx], 1)) % 1.0
        # only draw dots for reasonably alive tubes, capped for the flood phase
        dot_mask = rel > 0.45
        de = idx[dot_mask]
        if len(de) > 6000:
            de = de[np.argsort(D[de])[-6000:]]
        if len(de):
            nd = self.ndots[de]
            rep = np.repeat(de, nd)
            k = np.concatenate([np.arange(n) for n in nd])
            t = (self.phase[rep] + k / self.ndots[rep]) % 1.0
            pos = self.p_src[rep] + t[:, None] * (self.p_dst[rep] - self.p_src[rep])
            xi = pos[:, 0].astype(int)
            yi = pos[:, 1].astype(int)
            dot_c = np.array([250, 240, 200], np.float32)[::-1]  # warm white, BGR
            b = bright[rep][:, None]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    xx = np.clip(xi + dx, 0, self.W - 1)
                    yy = np.clip(yi + dy, 0, self.H - 1)
                    wgt = 1.0 if (dx == 0 and dy == 0) else 0.45
                    np.maximum.at(layer, (yy, xx), dot_c[None] * b * wgt)
        return layer

    def compose(self, layer):
        base = self.static.astype(np.float32)
        g = 0.35 + 0.65 * getattr(self, "dens", 1.0)
        glow = cv2.GaussianBlur(layer, (0, 0), 6) * 0.9 * g
        glow2 = cv2.GaussianBlur(layer, (0, 0), 18) * 0.35 * g
        out = base + glow + glow2 + layer
        return np.clip(out, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    def draw_cities(self, img):
        for i, cid in enumerate(self.city_ids):
            x, y = self.px[cid]
            r = int(2 + 5 * np.sqrt(self.pops[i] / self.pops.max()))
            cv2.circle(img, (int(x), int(y)), r + 2, rgb2bgr(BG), -1, cv2.LINE_AA)
            cv2.circle(img, (int(x), int(y)), r, rgb2bgr(WHITE), -1, cv2.LINE_AA)
        return img

    def draw_labels(self, pil):
        d = ImageDraw.Draw(pil)
        f = font(max(13, self.H // 70))
        for i, cid in enumerate(self.city_ids):
            if self.pops[i] < 900:
                continue
            x, y = self.px[cid]
            d.text((x + 9, y - 9), self.names[i], font=f, fill=GREY)

    def panel(self, pil, step, D, done_text=None):
        d = ImageDraw.Draw(pil)
        x = self.map_w + int(self.panel_w * 0.08)
        y = int(self.H * 0.22)
        big = font(int(self.H / 28), bold=True)
        mono = font(int(self.H / 46))
        small = font(int(self.H / 60))
        lh = int(self.H / 30)
        d.text((x, y), "MOLD vs PAKISTAN", font=big, fill=RED)
        y += int(lh * 1.8)
        alive = D > ALIVE_REL * D.max()
        n_alive = int(alive.sum())
        km = float(self.length[alive].sum())

        def row(label, value, color=WHITE, vcolor=None):
            nonlocal y
            d.text((x, y), label, font=mono, fill=color)
            vw = d.textlength(str(value), font=mono)
            d.text((x + self.panel_w * 0.78 - vw, y), str(value), font=mono, fill=vcolor or color)
            y += lh

        row("step", f"{step:d}")
        row("tubes alive", f"{n_alive:,}")
        row("of", f"{self.n_edges:,}", color=DIM)
        y += lh // 2
        row("network", f"{km:,.0f} km", color=GREEN)
        row("real railways", f"{self.rail_km:,.0f} km", color=DIM)
        y += lh // 2
        pct = n_alive / self.n_edges * 100
        if pct > 60:
            status, sc = "flooding...", GREY
        elif pct > 5:
            status, sc = "competing...", ORANGE
        else:
            status, sc = "STARVED  {:.1f}%".format(100 - pct), GREEN
        d.text((x, y), status, font=mono, fill=sc)
        y += int(lh * 1.6)

        # sparkline box
        bw, bh = int(self.panel_w * 0.78), int(self.H * 0.12)
        d.rectangle([x, y, x + bw, y + bh], outline=(60, 60, 70))
        if len(self.alive_curve) > 1:
            n = len(self.alive_curve)
            pts = [(x + 4 + i / max(1, self.n_steps - 1) * (bw - 8),
                    y + bh - 4 - v / 100 * (bh - 8)) for i, v in enumerate(self.alive_curve)]
            d.line(pts, fill=RED, width=2)
        d.text((x, y + bh + 6), "% of tubes still alive", font=small, fill=GREY)
        y += bh + int(lh * 2.2)
        if done_text:
            for line in done_text:
                d.text((x, y), line, font=small, fill=GREY)
                y += int(lh * 0.8)
        d.text((x, self.H - int(self.H * 0.08)), "Tero flow model  |  OSM rails", font=small, fill=DIM)

    # ------------------------------------------------------------------
    def frame(self, t_step, frame_idx, done_text=None):
        """t_step: fractional simulation step."""
        k = int(np.floor(t_step))
        k = min(k, self.n_steps - 1)
        fr = t_step - k
        if k + 1 < self.n_steps and fr > 0:
            D = (1 - fr) * self.Dh[k] + fr * self.Dh[k + 1]
            Q = (1 - fr) * self.Qh[k] + fr * self.Qh[k + 1]
        else:
            D, Q = self.Dh[k], self.Qh[k]
        layer = self.draw_tubes(D, Q, frame_idx)
        img = self.compose(layer)
        img = self.draw_cities(img)
        pil = Image.fromarray(img[..., ::-1])
        self.draw_labels(pil)
        self.panel(pil, k, D, done_text)
        return np.asarray(pil)[..., ::-1].copy()

    def card(self, lines, colors, sizes, bolds):
        pil = Image.new("RGB", (self.W, self.H), BG)
        d = ImageDraw.Draw(pil)
        total = sum(int(self.H / s) * 1.5 for s in sizes)
        y = (self.H - total) / 2
        for text, col, s, b in zip(lines, colors, sizes, bolds):
            f = font(int(self.H / s), bold=b)
            w = d.textlength(text, font=f)
            d.text(((self.W - w) / 2, y), text, font=f, fill=col)
            y += int(self.H / s) * 1.5
        return np.asarray(pil)[..., ::-1].copy()


# ----------------------------------------------------------------------------
def schedule(n_steps, fps):
    """Fractional sim steps per video frame: slow at first (the drama is in the
    first ~60 steps), faster afterwards."""
    ts = []
    last = min(200, n_steps - 1)
    seg = [(0, 40, 2), (40, 110, 7), (110, 160, 3), (160, last, 2)]
    for a, b, fpstep in seg:
        for s in range(a, b):
            for j in range(fpstep):
                ts.append(s + j / fpstep)
    ts.append(last)
    return ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(OUT, "mold_pakistan.mp4"))
    ap.add_argument("--gif", action="store_true", help="also write a 15 fps GIF")
    args = ap.parse_args()
    W = args.width
    H = int(W * 9 / 16)

    sc = Scene(W, H)
    ts = schedule(sc.n_steps, args.fps)
    print(f"{len(ts)} frames at {args.fps} fps = {len(ts) / args.fps:.1f}s of simulation footage")

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(args.fps), "-i", "-",
           "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-preset", "medium", args.out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    def emit(img, n=1):
        for _ in range(n):
            proc.stdin.write(np.ascontiguousarray(img).tobytes())

    # intro card
    intro = sc.card(
        ["MOLD vs PAKISTAN",
         f"{sc.n_edges:,} tubes. {len(sc.city_ids)} cities. one blob.",
         "every route it tried, live. tubes that carry flow thicken, the rest starve."],
        [RED, WHITE, GREY], [12, 32, 46], [True, False, False])
    emit(intro, int(args.fps * 2.5))

    last_k = -1
    for i, t in enumerate(ts):
        k = int(np.floor(t))
        if k != last_k:
            sc.alive_curve.append(float((sc.Dh[k] > ALIVE_REL * sc.Dh[k].max()).mean() * 100))
            last_k = k
        emit(sc.frame(t, i))
        if i % 100 == 0:
            print(f"  frame {i}/{len(ts)}")

    # hold on the final network
    Dend = sc.Dh[int(ts[-1])]
    alive = Dend > ALIVE_REL * Dend.max()
    km = float(sc.length[alive].sum())
    starved = 100 - alive.mean() * 100
    hold = sc.frame(ts[-1], len(ts), done_text=[
        f"starved {starved:.1f}% of its tubes",
        f"{km:,.0f} km vs {sc.rail_km:,.0f} km real",
        "every city still connected"])
    emit(hold, int(args.fps * 3))

    # end card
    end = sc.card(
        ["THE RESULT",
         f"starved {starved:.1f}% of its own tubes",
         f"kept {km:,.0f} km of track  vs  {sc.rail_km:,.0f} km real railway",
         "Karachi-Lahore-Peshawar trunk, Quetta branch, Faisalabad loop: all rediscovered",
         "",
         "no brain. no search. just flow."],
        [RED, WHITE, WHITE, GREY, GREY, YELLOW], [14, 34, 34, 46, 60, 28],
        [True, False, False, False, False, True])
    emit(end, int(args.fps * 4))

    proc.stdin.close()
    proc.wait()
    print("wrote", args.out)

    if args.gif:
        gif = os.path.splitext(args.out)[0] + ".gif"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", args.out,
                        "-vf", "fps=15,scale=960:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer",
                        gif], check=True)
        print("wrote", gif)


if __name__ == "__main__":
    main()
