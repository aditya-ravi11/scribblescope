"""Day-0 data probe for ScribbleScope.

Samples Quick, Draw! simplified ndjson files via HTTP range requests (no full
downloads) to:
  1. verify the countrycode field and stroke format,
  2. estimate per-class drawing counts and country distributions,
  3. validate stroke -> 64x64 grayscale rasterization,
  4. emit the locked 50-class list and the achievable country list.

Outputs land in ml/probe/.
"""

import io
import json
import random
import urllib.parse
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
OUT = Path(__file__).parent / "probe"
CHUNK = 131072  # 128 KiB per range request
OFFSETS_PER_FILE = 6
SEED = 20260809

# 60 candidates -> probe picks 50. Bias toward classes with documented or
# plausible cross-cultural drawing variation (Google's own Quick Draw research:
# bread, power outlet, mailbox, snowman, chair, house...) plus visually fun,
# highly drawable objects.
CANDIDATES = [
    # documented cultural variation (keep all)
    "bread", "power outlet", "mailbox", "snowman", "chair", "house", "teapot",
    "mug", "coffee cup", "telephone", "television", "traffic light", "bus",
    "train", "castle", "fan", "ice cream", "umbrella", "shoe", "hat",
    # everyday objects, plausible variation
    "door", "table", "bed", "couch", "key", "scissors", "clock", "eyeglasses",
    "backpack", "book", "pencil", "laptop", "cell phone", "camera",
    "headphones", "envelope", "ladder", "candle", "bridge", "church",
    "lighthouse", "windmill", "tent",
    # food
    "apple", "banana", "pizza", "hamburger", "hot dog", "birthday cake",
    "donut",
    # nature / scenes
    "sun", "moon", "star", "cloud", "rainbow", "mountain", "tree", "flower",
    "palm tree", "cactus", "beach",
    # animals & vehicles (fun to draw)
    "cat", "dog", "fish", "bird", "airplane", "bicycle", "car", "sailboat",
    "hot air balloon", "guitar",
]


def url_for(cls: str) -> str:
    return BASE.format(urllib.parse.quote(cls))


def fetch_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{start + length - 1}"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def head_size(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def probe_class(cls: str) -> dict:
    url = url_for(cls)
    size = head_size(url)
    rng = random.Random(f"{SEED}:{cls}")
    # Evenly spaced offsets with jitter; skip the tail chunk.
    step = max((size - CHUNK) // OFFSETS_PER_FILE, 1)
    offsets = [min(i * step + rng.randint(0, step // 2), size - CHUNK - 1) for i in range(OFFSETS_PER_FILE)]
    offsets[0] = 0  # include file start (complete first line, catches format drift)

    countries: Counter = Counter()
    line_bytes: list[int] = []
    recognized = 0
    total = 0
    stroke_counts: list[int] = []
    samples: list[dict] = []

    for off in offsets:
        try:
            blob = fetch_range(url, off, CHUNK)
        except Exception:
            continue
        text = blob.decode("utf-8", errors="ignore")
        lines = text.split("\n")
        # Drop partial first line unless we started at byte 0; last line is partial too.
        body = lines[1:-1] if off else lines[:-1]
        for ln in body:
            if not ln.strip():
                continue
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            total += 1
            line_bytes.append(len(ln) + 1)
            countries[d.get("countrycode", "??")] += 1
            recognized += bool(d.get("recognized"))
            stroke_counts.append(len(d.get("drawing", [])))
            if len(samples) < 3 and d.get("recognized"):
                samples.append(d)

    avg_line = float(np.mean(line_bytes)) if line_bytes else float("nan")
    return {
        "class": cls,
        "file_bytes": size,
        "sampled_lines": total,
        "est_total_drawings": int(size / avg_line) if line_bytes else 0,
        "recognized_frac": round(recognized / total, 4) if total else 0.0,
        "avg_strokes": round(float(np.mean(stroke_counts)), 2) if stroke_counts else 0.0,
        "countries": dict(countries.most_common()),
        "samples": samples,
    }


def rasterize(drawing: list, size: int = 64, src: int = 256, line_w: int = 2) -> np.ndarray:
    """Simplified-format strokes ([[x...],[y...]] in 0..255) -> size x size grayscale."""
    img = Image.new("L", (src, src), 0)
    d = ImageDraw.Draw(img)
    for xs, ys in drawing:
        pts = list(zip(xs, ys))
        if len(pts) == 1:
            d.point(pts, fill=255)
        else:
            d.line(pts, fill=255, width=line_w * (src // size))
    return np.asarray(img.resize((size, size), Image.LANCZOS))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=12) as ex:
        futs = {ex.submit(probe_class, c): c for c in CANDIDATES}
        for fut in as_completed(futs):
            cls = futs[fut]
            try:
                r = fut.result()
                results[cls] = r
                print(f"  {cls:16s} est={r['est_total_drawings']:>9,}  rec={r['recognized_frac']:.2f}  countries={len(r['countries'])}")
            except Exception as e:
                print(f"  {cls:16s} FAILED: {e}")

    # --- montage: validate rasterization on 3 classes x 3 samples ---
    montage_classes = [c for c in ("bread", "power outlet", "snowman") if c in results and results[c]["samples"]]
    tiles = []
    for c in montage_classes:
        tiles.extend(rasterize(s["drawing"]) for s in results[c]["samples"][:3])
    if tiles:
        n = len(tiles)
        m = Image.new("L", (64 * n, 64))
        for i, t in enumerate(tiles):
            m.paste(Image.fromarray(t), (64 * i, 0))
        m.resize((128 * n, 128), Image.NEAREST).save(OUT / "raster_montage.png")

    # --- aggregate country distribution (weighted by est class size) ---
    agg: Counter = Counter()
    for r in results.values():
        tot = sum(r["countries"].values()) or 1
        for cc, n in r["countries"].items():
            agg[cc] += (n / tot) * r["est_total_drawings"]
    total_est = sum(agg.values())
    country_share = {cc: n / total_est for cc, n in agg.most_common()}

    # --- pick the 50 classes ---
    # Keep every documented-variation class; fill the rest by sampled-line
    # health (probe succeeded), recognized fraction, and country diversity.
    must_keep = CANDIDATES[:20]
    rest = [c for c in CANDIDATES[20:] if c in results]
    rest.sort(key=lambda c: (results[c]["recognized_frac"], len(results[c]["countries"])), reverse=True)
    chosen = [c for c in must_keep if c in results] + rest
    chosen = chosen[:50]

    # --- achievable country table under the sampling plan ---
    # Plan: 12k drawings/class x 50 classes = 600k, 10% test = 60k test rows.
    # A country with global share s gets ~60000*s test rows WITHOUT stratification.
    plan_test_rows = 60000
    kept_countries = {cc: round(share * plan_test_rows) for cc, share in country_share.items() if share * plan_test_rows >= 1000}

    for r in results.values():
        r.pop("samples", None)  # keep the stats file lean

    (OUT / "class_stats.json").write_text(json.dumps(results, indent=1))
    (OUT / "country_distribution.json").write_text(json.dumps(country_share, indent=1))
    (OUT / "locked_choices.json").write_text(json.dumps({
        "classes": chosen,
        "countries_ge_1k_test_natural": kept_countries,
        "note": "natural (unstratified) estimate; stratified sampling caps US share and lifts the tail",
    }, indent=1))

    print(f"\nProbed {len(results)}/{len(CANDIDATES)} classes OK")
    print(f"Total est drawings across candidates: {int(total_est):,}")
    print(f"\nTop 15 countries by share: " + ", ".join(f"{cc}:{s:.1%}" for cc, s in list(country_share.items())[:15]))
    print(f"\nCountries with >=1k test rows under natural 60k test split: {len(kept_countries)}")
    print(f"\nLocked 50 classes: {chosen}")


if __name__ == "__main__":
    main()
