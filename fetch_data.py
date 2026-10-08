"""Download the inputs for the Pakistan slime-mould rail experiment.

Writes into ./data:
  pakistan.geojson   country outline (geoBoundaries ADM0)
  elevation.npz      elevation grid in metres (AWS Terrarium tiles, zoom 7)
  railways.json      OpenStreetMap railway=rail ways as lon/lat polylines
"""
import io
import json
import math
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA, exist_ok=True)

# Bounding box that comfortably covers Pakistan (lon_min, lat_min, lon_max, lat_max)
BBOX = (60.5, 23.3, 78.0, 37.5)
HEADERS = {"User-Agent": "physarum-pakistan-rail-experiment/0.1"}


def fetch_boundary():
    out = os.path.join(DATA, "pakistan.geojson")
    if os.path.exists(out):
        print("boundary: cached")
        return
    meta = requests.get(
        "https://www.geoboundaries.org/api/current/gbOpen/PAK/ADM0/",
        headers=HEADERS, timeout=60,
    ).json()
    url = meta["gjDownloadURL"]
    gj = requests.get(url, headers=HEADERS, timeout=120).json()
    with open(out, "w") as f:
        json.dump(gj, f)
    print("boundary: saved", out)


def _tile_xy(lon, lat, z):
    n = 2 ** z
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1.0 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _tile_bounds(x, y, z):
    n = 2 ** z
    lon0 = x / n * 360.0 - 180.0
    lon1 = (x + 1) / n * 360.0 - 180.0
    lat0 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    lat1 = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon0, lat0, lon1, lat1


def fetch_elevation(z=7):
    out = os.path.join(DATA, "elevation.npz")
    if os.path.exists(out):
        print("elevation: cached")
        return
    x0, y1 = _tile_xy(BBOX[0], BBOX[1], z)
    x1, y0 = _tile_xy(BBOX[2], BBOX[3], z)
    xs = range(int(math.floor(x0)), int(math.floor(x1)) + 1)
    ys = range(int(math.floor(y0)), int(math.floor(y1)) + 1)
    W, H = 256 * len(xs), 256 * len(ys)
    mosaic = np.zeros((H, W), dtype=np.float32)
    for j, ty in enumerate(ys):
        for i, tx in enumerate(xs):
            url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{tx}/{ty}.png"
            for attempt in range(3):
                try:
                    r = requests.get(url, headers=HEADERS, timeout=60)
                    r.raise_for_status()
                    break
                except Exception as e:  # noqa: BLE001
                    print("  retry", url, e)
                    time.sleep(2)
            else:
                raise RuntimeError("elevation tile download failed: " + url)
            img = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB")).astype(np.float32)
            elev = img[..., 0] * 256 + img[..., 1] + img[..., 2] / 256 - 32768
            mosaic[j * 256:(j + 1) * 256, i * 256:(i + 1) * 256] = elev
        print(f"  elevation row {j + 1}/{len(ys)} done")
    # Web-mercator mosaic extent
    lon0, _, _, lat1 = _tile_bounds(xs[0], ys[0], z)
    _, lat0, lon1, _ = _tile_bounds(xs[-1], ys[-1], z)
    # Row -> latitude is non-linear in mercator; store per-row latitude so the
    # simulation can interpolate correctly.
    n = 2 ** z
    row_y = ys[0] + (np.arange(H) + 0.5) / 256.0
    row_lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * row_y / n))))
    col_lon = lon0 + (np.arange(W) + 0.5) / W * (lon1 - lon0)
    np.savez_compressed(out, elev=mosaic, lat=row_lat.astype(np.float32), lon=col_lon.astype(np.float32))
    print("elevation: saved", out, mosaic.shape)


# Manual fallback: major Pakistan Railways corridors as city-to-city sequences
# (used only if Overpass is unreachable). Coordinates are filled in from CITIES.
FALLBACK_LINES = [
    # ML-1 Karachi -> Peshawar
    ["Karachi", "Hyderabad", "Nawabshah", "Sukkur", "Rahim Yar Khan", "Bahawalpur",
     "Multan", "Khanewal", "Sahiwal", "Lahore", "Gujranwala", "Gujrat", "Jhelum",
     "Rawalpindi", "Attock", "Peshawar"],
    # ML-2 Kotri -> Attock via the west bank of the Indus
    ["Hyderabad", "Dadu", "Larkana", "Jacobabad", "Dera Ghazi Khan", "Kundian", "Attock"],
    # ML-3 Rohri -> Quetta -> Chaman
    ["Sukkur", "Jacobabad", "Sibi", "Quetta", "Chaman"],
    # Branches
    ["Khanewal", "Faisalabad", "Sargodha", "Kundian"],
    ["Lahore", "Faisalabad"],
    ["Lahore", "Sialkot"],
    ["Sukkur", "Mirpur Khas", "Hyderabad"],
    ["Quetta", "Zhob"],
]


def fetch_railways():
    out = os.path.join(DATA, "railways.json")
    if os.path.exists(out):
        print("railways: cached")
        return
    query = f"""
    [out:json][timeout:180];
    (
      way["railway"="rail"]({BBOX[1]},{BBOX[0]},{BBOX[3]},{BBOX[2]});
    );
    out geom;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    lines = []
    for ep in endpoints:
        try:
            r = requests.post(ep, data={"data": query}, headers=HEADERS, timeout=300)
            r.raise_for_status()
            js = r.json()
            for el in js.get("elements", []):
                if el.get("type") != "way" or "geometry" not in el:
                    continue
                tags = el.get("tags", {})
                if tags.get("service") in ("yard", "siding", "spur", "crossover"):
                    continue
                lines.append([[p["lon"], p["lat"]] for p in el["geometry"]])
            if lines:
                break
        except Exception as e:  # noqa: BLE001
            print("  overpass failed at", ep, "->", e)
    source = "osm"
    if not lines:
        print("  using hardcoded fallback corridors")
        from cities import CITIES
        lookup = {c["name"]: (c["lon"], c["lat"]) for c in CITIES}
        for seq in FALLBACK_LINES:
            lines.append([list(lookup[n]) for n in seq])
        source = "fallback"
    with open(out, "w") as f:
        json.dump({"source": source, "lines": lines}, f)
    print(f"railways: saved {out} ({len(lines)} polylines, source={source})")


if __name__ == "__main__":
    fetch_boundary()
    fetch_elevation()
    fetch_railways()
    print("done")
