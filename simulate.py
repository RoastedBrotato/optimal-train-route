"""Physarum-style adaptive transport network over Pakistan.

Implements the Tero et al. (2010, Science) flow model used in the Tokyo rail
experiment:

  * a lattice of tubes covers the land area of the country
  * cities are food sources; each step a random pair is chosen and unit flow
    is pushed from one to the other (pressures from a sparse Poisson solve)
  * tube conductivity D adapts:  dD/dt = f(|Q|) - D,  f(Q) = Q^mu / (1 + Q^mu)
    so tubes carrying flow thicken and idle tubes starve

Terrain (elevation and slope) makes tubes "longer", mimicking the light the
researchers used to discourage the mould from crossing mountains and water.

Usage:
    python simulate.py [--res 0.1] [--steps 300] [--seed 1]
Outputs in ./out:  network.png, comparison.png, overlay.png, pruning.gif, stats.txt
"""
import argparse
import json
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse.csgraph import connected_components
import shapely
from shapely.geometry import shape

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from PIL import Image  # noqa: E402

from cities import CITIES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

KM_PER_DEG = 111.2
ALIVE = 0.005  # conductivity below which a tube counts as starved


# ----------------------------------------------------------------------------
# Geometry: land mask, lattice, terrain cost
# ----------------------------------------------------------------------------
def load_country():
    with open(os.path.join(DATA, "pakistan.geojson")) as f:
        gj = json.load(f)
    geoms = [shape(ft["geometry"]) for ft in gj["features"]]
    return shapely.union_all(geoms)


def load_elevation():
    z = np.load(os.path.join(DATA, "elevation.npz"))
    elev, lat, lon = z["elev"], z["lat"], z["lon"]
    # block-average 4x4 (~1.2 km px -> ~5 km) to tame noise
    b = 4
    H, W = elev.shape[0] // b * b, elev.shape[1] // b * b
    e = elev[:H, :W].reshape(H // b, b, W // b, b).mean(axis=(1, 3))
    la = lat[:H].reshape(H // b, b).mean(axis=1)
    lo = lon[:W].reshape(W // b, b).mean(axis=1)
    # lat is descending in a mercator mosaic; interpolator needs ascending
    interp = RegularGridInterpolator((la[::-1], lo), e[::-1], bounds_error=False, fill_value=0.0)
    return interp


def build_lattice(country, elev_interp, res, max_elev, jitter=0.35, rng=None):
    rng = rng or np.random.default_rng(0)
    minx, miny, maxx, maxy = country.bounds
    lons = np.arange(minx, maxx + res, res)
    lats = np.arange(miny, maxy + res, res)
    LON, LAT = np.meshgrid(lons, lats)
    # jitter node positions so the lattice has no families of equal-length
    # staircase paths (a regular grid produces parallel-strand artefacts)
    LON = LON + rng.uniform(-jitter, jitter, LON.shape) * res
    LAT = LAT + rng.uniform(-jitter, jitter, LAT.shape) * res
    inside = shapely.contains_xy(country, LON.ravel(), LAT.ravel()).reshape(LON.shape)
    elev = elev_interp(np.c_[LAT.ravel(), LON.ravel()]).reshape(LON.shape)
    inside &= elev < max_elev

    ny, nx = LON.shape
    idx = -np.ones((ny, nx), dtype=np.int64)
    idx[inside] = np.arange(inside.sum())
    n = inside.sum()

    # 8-neighbour edges (each undirected edge once)
    src, dst = [], []
    for dy, dx in [(0, 1), (1, 0), (1, 1), (1, -1)]:
        a = idx[max(0, -dy):ny - max(0, dy), max(0, -dx):nx - max(0, dx)]
        b = idx[max(0, dy):ny - max(0, -dy) if dy else ny, max(0, dx):nx - max(0, -dx) if dx else nx]
        # a and b are same-shape views offset by (dy, dx)
        m = (a >= 0) & (b >= 0)
        src.append(a[m])
        dst.append(b[m])
    src = np.concatenate(src)
    dst = np.concatenate(dst)

    node_lon = LON[inside]
    node_lat = LAT[inside]
    node_elev = elev[inside]

    # edge length in km
    dlon = (node_lon[dst] - node_lon[src]) * np.cos(np.radians((node_lat[src] + node_lat[dst]) / 2))
    dlat = node_lat[dst] - node_lat[src]
    length = KM_PER_DEG * np.sqrt(dlon ** 2 + dlat ** 2)

    # terrain cost: altitude + slope penalty (the "light" in the experiment)
    emean = (node_elev[src] + node_elev[dst]) / 2
    slope = np.abs(node_elev[dst] - node_elev[src]) / length  # m per km
    cost = 1.0 + 0.5 * np.clip(emean - 300, 0, None) / 1000.0 + 0.05 * slope
    eff_len = length * cost

    return dict(n=n, src=src, dst=dst, lon=node_lon, lat=node_lat, elev=node_elev,
                length=length, eff_len=eff_len, cost=cost)


def snap_cities(lat_):
    lon, lat = lat_["lon"], lat_["lat"]
    ids, names, pops = [], [], []
    for c in CITIES:
        d2 = ((lon - c["lon"]) * np.cos(np.radians(c["lat"]))) ** 2 + (lat - c["lat"]) ** 2
        ids.append(int(np.argmin(d2)))
        names.append(c["name"])
        pops.append(float(c["pop"]))
    return np.array(ids), names, np.array(pops)


def keep_main_component(g, city_ids):
    A = sp.coo_matrix((np.ones(len(g["src"])), (g["src"], g["dst"])), shape=(g["n"], g["n"]))
    _, labels = connected_components(A, directed=False)
    main = np.bincount(labels[city_ids]).argmax()
    keep = labels == main
    lost = [i for i in city_ids if labels[i] != main]
    if lost:
        raise RuntimeError(f"{len(lost)} cities fall outside the main land component; lower --res")
    remap = -np.ones(g["n"], dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    em = keep[g["src"]] & keep[g["dst"]]
    out = dict(n=int(keep.sum()), src=remap[g["src"][em]], dst=remap[g["dst"][em]])
    for k in ("lon", "lat", "elev"):
        out[k] = g[k][keep]
    for k in ("length", "eff_len", "cost"):
        out[k] = g[k][em]
    return out, remap[city_ids]


# ----------------------------------------------------------------------------
# The mould
# ----------------------------------------------------------------------------
class Physarum:
    def __init__(self, g, city_ids, pops, I0=1.0, mu=1.8, Q0=0.5, dt=0.1, rng=None):
        self.g = g
        self.n = g["n"]
        self.src, self.dst = g["src"], g["dst"]
        self.L = g["eff_len"]
        self.city_ids = city_ids
        self.rng = rng or np.random.default_rng(0)
        # wide random start, as in Tero et al.: tubes then starve at different times
        self.D = self.rng.uniform(0.1, 1.0, len(self.src))
        self.I0, self.mu, self.Q0, self.dt = I0, mu, Q0, dt
        self.floor = 1e-6
        # pair weight ~ sqrt(pop_a * pop_b): a soft gravity model of demand
        w = np.sqrt(pops)
        W = np.outer(w, w)
        np.fill_diagonal(W, 0.0)
        self.pair_w = W / W.sum()
        self.ground = int(city_ids[np.argmax(pops)])
        self.last_dP = None
        # showcase flow for animation: biggest city sources, all others sink ~ sqrt(pop)
        self.show_src = int(np.argmax(pops))
        sw = w.copy(); sw[self.show_src] = 0.0
        self.show_w = sw / sw.sum()

    def showcase_flux(self):
        """Signed flow per tube for 'largest city -> everyone else' (by linearity)."""
        if self.last_dP is None:
            return np.zeros(len(self.D))
        return self.last_dP[:, self.show_src] - self.last_dP @ self.show_w

    def laplacian(self):
        w = self.D / self.L
        n = self.n
        rows = np.concatenate([self.src, self.dst, self.src, self.dst])
        cols = np.concatenate([self.dst, self.src, self.src, self.dst])
        vals = np.concatenate([-w, -w, w, w])
        # ground one node (zero its row, unit diagonal) so the system is non-singular
        keep = rows != self.ground
        rows = np.append(rows[keep], self.ground)
        cols = np.append(cols[keep], self.ground)
        vals = np.append(vals[keep], 1.0)
        return sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsc()

    def step(self):
        """One adaptation step using the demand-weighted average of f(|Q|)
        over every city pair.  By linearity the pressure field for a pair
        (a, b) is P_a - P_b where P_a is the field for a unit source at a
        drained at the ground node, so only one solve per city is needed."""
        lu = spla.splu(self.laplacian())
        k = len(self.city_ids)
        rhs = np.zeros((self.n, k))
        rhs[self.city_ids, np.arange(k)] = self.I0
        rhs[self.ground, :] = 0.0
        P = lu.solve(rhs)
        w = self.D / self.L
        dP = w[:, None] * (P[self.src] - P[self.dst])  # edges x cities
        self.last_dP = dP
        f = np.zeros(len(self.D))
        for a in range(k - 1):
            Q = np.abs(dP[:, a:a + 1] - dP[:, a + 1:])  # edges x (pairs a,b>a)
            q = (Q / self.Q0) ** self.mu
            f += (q / (1.0 + q)) @ (2.0 * self.pair_w[a, a + 1:])
        self.D += self.dt * (f - self.D)
        self.D = np.maximum(self.D, self.floor)

    def alive(self, thr=ALIVE):
        return self.D > thr


# ----------------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------------
def draw_country(ax, country):
    polys = country.geoms if hasattr(country, "geoms") else [country]
    for p in polys:
        x, y = p.exterior.xy
        ax.fill(x, y, color="#f3efe6", zorder=0)
        ax.plot(x, y, color="#8a8378", lw=0.6, zorder=1)
    ax.set_aspect(1 / np.cos(np.radians(30)))
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def draw_network(ax, g, D, thr=ALIVE, color="#c8102e", max_lw=4.0):
    m = D > thr
    segs = np.stack([np.c_[g["lon"][g["src"][m]], g["lat"][g["src"][m]]],
                     np.c_[g["lon"][g["dst"][m]], g["lat"][g["dst"][m]]]], axis=1)
    lw = max_lw * np.sqrt(np.clip(D[m], 0, 1))
    alpha = np.clip(0.25 + 0.75 * np.sqrt(D[m]), 0, 1)
    lc = LineCollection(segs, linewidths=lw, colors=color, alpha=None, zorder=3, capstyle="round")
    lc.set_alpha(None)
    rgba = np.zeros((m.sum(), 4))
    rgba[:, :3] = matplotlib.colors.to_rgb(color)
    rgba[:, 3] = alpha
    lc.set_color(rgba)
    ax.add_collection(lc)


def draw_cities(ax, names, pops, lons, lats, label=True):
    s = 6 + 60 * np.sqrt(pops / pops.max())
    ax.scatter(lons, lats, s=s, color="#1a1a1a", zorder=5, edgecolor="white", linewidth=0.6)
    if label:
        for n, p, x, y in zip(names, pops, lons, lats):
            if p >= 400:
                ax.annotate(n, (x, y), xytext=(4, 3), textcoords="offset points",
                            fontsize=6.5, color="#222", zorder=6)


def clip_rails(lines, country):
    """Keep only the parts of the OSM rail polylines inside the border."""
    out = []
    for l in lines:
        if len(l) < 2:
            continue
        geom = shapely.LineString(l).intersection(country)
        if geom.is_empty:
            continue
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for p in parts:
            if p.geom_type == "LineString" and len(p.coords) > 1:
                out.append(list(p.coords))
    return out


def draw_rails(ax, lines, color="#1f4e9c", lw=0.7):
    segs = [np.array(l) for l in lines if len(l) > 1]
    ax.add_collection(LineCollection(segs, colors=color, linewidths=lw, zorder=2, alpha=0.9))


def render_frame(country, g, D, names, pops, clon, clat, step, dpi=110):
    fig, ax = plt.subplots(figsize=(7.2, 6.6), dpi=dpi)
    draw_country(ax, country)
    draw_network(ax, g, D)
    draw_cities(ax, names, pops, clon, clat, label=False)
    alive = (D > ALIVE).mean() * 100
    ax.set_title(f"step {step:4d}    tubes alive: {alive:5.1f}%", loc="left", fontsize=10, family="monospace")
    minx, miny, maxx, maxy = country.bounds
    ax.set_xlim(minx - 0.3, maxx + 0.3)
    ax.set_ylim(miny - 0.3, maxy + 0.3)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=float, default=0.1, help="lattice spacing in degrees")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-elev", type=float, default=4500.0)
    ap.add_argument("--frame-every", type=int, default=5)
    ap.add_argument("--history-steps", type=int, default=200, help="steps of D/flux saved for render_video.py")
    args = ap.parse_args()

    t0 = time.time()
    country = load_country()
    elev_interp = load_elevation()
    rng = np.random.default_rng(args.seed)
    g = build_lattice(country, elev_interp, args.res, args.max_elev, rng=rng)
    city_ids, names, pops = snap_cities(g)
    g, city_ids = keep_main_component(g, city_ids)
    clon, clat = g["lon"][city_ids], g["lat"][city_ids]
    print(f"lattice: {g['n']} nodes, {len(g['src'])} tubes, {len(city_ids)} cities "
          f"({time.time() - t0:.1f}s)")

    model = Physarum(g, city_ids, pops, rng=rng)

    frames = []
    hist_D, hist_Q = [], []
    t1 = time.time()
    for step in range(args.steps + 1):
        if step <= args.history_steps:
            hist_D.append(model.D.astype(np.float32))
            hist_Q.append(model.showcase_flux().astype(np.float32))
        if step % args.frame_every == 0:
            frames.append(render_frame(country, g, model.D, names, pops, clon, clat, step))
        if step % 50 == 0:
            alive = model.alive().mean() * 100
            print(f"step {step:4d}  alive {alive:5.1f}%  elapsed {time.time() - t1:5.1f}s")
        if step < args.steps:
            model.step()

    with open(os.path.join(DATA, "railways.json")) as f:
        rails = json.load(f)
    rails["lines"] = clip_rails(rails["lines"], country)
    rail_km = 0.0
    for l in rails["lines"]:
        a = np.array(l)
        dlon = np.diff(a[:, 0]) * np.cos(np.radians(a[:-1, 1]))
        rail_km += KM_PER_DEG * np.sqrt(dlon ** 2 + np.diff(a[:, 1]) ** 2).sum()

    D = model.D
    alive = D > ALIVE
    total_km = g["length"][alive].sum()
    # a "backbone" = tubes that are reasonably thick
    backbone = D > 0.2
    # are all cities still linked by living tubes?
    A = sp.coo_matrix((np.ones(alive.sum()), (g["src"][alive], g["dst"][alive])), shape=(g["n"], g["n"]))
    _, lab = connected_components(A, directed=False)
    ncomp_cities = len(set(lab[city_ids]))
    stats = [
        f"all cities connected:     {'yes' if ncomp_cities == 1 else f'no ({ncomp_cities} components)'}",
        f"lattice nodes:            {g['n']}",
        f"tubes at start:           {len(D)}",
        f"tubes alive (D>{ALIVE}):     {alive.sum()}  ({alive.mean() * 100:.1f}%)",
        f"tubes starved:            {(~alive).sum()}  ({(~alive).mean() * 100:.1f}%)",
        f"network length (alive):   {total_km:,.0f} km",
        f"backbone length (D>0.2):  {g['length'][backbone].sum():,.0f} km",
        f"OSM rail ways inside PK:  {rail_km:,.0f} km (sums every mapped track, so double track counts twice)",
        f"steps: {args.steps}   res: {args.res} deg   seed: {args.seed}",
    ]
    with open(os.path.join(OUT, "stats.txt"), "w") as f:
        f.write("\n".join(stats) + "\n")
    print("\n".join(stats))

    # GIF: hold the last frame a while
    frames += [frames[-1]] * 15
    frames[0].save(os.path.join(OUT, "pruning.gif"), save_all=True, append_images=frames[1:],
                   duration=80, loop=0, optimize=False)

    # Final network figure
    fig, ax = plt.subplots(figsize=(9, 8.4), dpi=150)
    draw_country(ax, country)
    draw_network(ax, g, D, max_lw=5)
    draw_cities(ax, names, pops, clon, clat)
    ax.set_title("Physarum transport network for Pakistan (Tero flow model)", loc="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "network.png"))
    plt.close(fig)

    # Comparison with the real railways
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2), dpi=150)
    draw_country(axes[0], country)
    draw_network(axes[0], g, D, max_lw=4)
    draw_cities(axes[0], names, pops, clon, clat)
    axes[0].set_title("slime mould", loc="left", fontsize=12)
    draw_country(axes[1], country)
    draw_rails(axes[1], rails["lines"])
    draw_cities(axes[1], names, pops, clon, clat)
    axes[1].set_title(f"Pakistan Railways (OpenStreetMap, {rails['source']})", loc="left", fontsize=12)
    for ax in axes:
        minx, miny, maxx, maxy = country.bounds
        ax.set_xlim(minx - 0.3, maxx + 0.3)
        ax.set_ylim(miny - 0.3, maxy + 0.3)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, "comparison.png"))
    plt.close(fig)

    # Overlay figure: mould on top of rails
    fig, ax = plt.subplots(figsize=(9, 8.4), dpi=150)
    draw_country(ax, country)
    draw_rails(ax, rails["lines"], color="#1f4e9c", lw=1.0)
    draw_network(ax, g, D, max_lw=4, color="#c8102e")
    draw_cities(ax, names, pops, clon, clat)
    ax.set_title("overlay: mould (red) vs real railways (blue)", loc="left", fontsize=11)
    minx, miny, maxx, maxy = country.bounds
    ax.set_xlim(minx - 0.3, maxx + 0.3)
    ax.set_ylim(miny - 0.3, maxy + 0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "overlay.png"))
    plt.close(fig)

    np.savez_compressed(os.path.join(OUT, "network.npz"), D=D, src=g["src"], dst=g["dst"],
                        lon=g["lon"], lat=g["lat"], city_ids=city_ids)
    np.savez_compressed(os.path.join(OUT, "history.npz"), D=np.stack(hist_D), Q=np.stack(hist_Q),
                        src=g["src"], dst=g["dst"], lon=g["lon"], lat=g["lat"], length=g["length"],
                        city_ids=city_ids, rail_km=rail_km)
    print(f"total time {time.time() - t0:.1f}s, outputs in {OUT}")


if __name__ == "__main__":
    main()
