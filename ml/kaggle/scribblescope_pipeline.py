"""ScribbleScope end-to-end pipeline — runs top-to-bottom on a Kaggle GPU notebook.

Stages (toggle below):
  DATA    — stream 50 Quick, Draw! simplified ndjson files from GCS, split by
            md5(key_id) BEFORE sampling (train pool 85% / calib 5% / test 10%),
            build the 600k stratified train set (US capped), the full ~650k
            test set, a 60k calib set, and rasterize everything to 64x64 uint8.
  TRAIN   — train CNN seeds 1-3 (baseline sampling) + seed 101 (country-balanced
            resampling) with AMP; ~30-40 min/model on a T4.
  EVAL    — per-seed + 3-seed-ensemble metrics on the full test set: top-1/3,
            50x50 confusion, per-class, per-country accuracy with bootstrap CIs,
            temperature scaling on calib, ECE + reliability bins. Emits the
            eval JSONs the web app renders.
  EXPORT  — ONNX (dynamic batch) + int8 dynamic quantization + parity check.

Design notes (also in the model card):
  * Only `recognized == True` drawings are used (label-noise control).
  * Splits are by key_id hash, fixed before any sampling — no drawing can leak
    across splits, and the test pool is NEVER subsampled, so the fairness audit
    covers every country with >=2k test rows (~45 countries).
  * Baseline train set caps the US at 25%/class (natural share ~44%) so the
    tail is represented; the "balanced" variant resamples with weights
    proportional to n_country^-0.5 for the Fixing-Bias lab.
"""

import hashlib
import json
import math
import os
import random
import urllib.parse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

try:
    import orjson as fastjson

    def jloads(b):
        return fastjson.loads(b)
except ImportError:
    def jloads(b):
        return json.loads(b)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RUN_DATA = True
RUN_TRAIN = True
RUN_EVAL = True
RUN_EXPORT = True

# SMOKE mode: tiny end-to-end run (3 classes, ~20k lines/file, 1 epoch, CPU-ok)
# to validate the full pipeline before spending GPU hours. Never for real metrics.
SMOKE = os.environ.get("SCRIBBLE_SMOKE") == "1"

CLASSES = [
    "bread", "power outlet", "mailbox", "snowman", "chair", "house", "teapot",
    "mug", "coffee cup", "telephone", "television", "traffic light", "bus",
    "train", "castle", "fan", "ice cream", "umbrella", "shoe", "hat",
    "rainbow", "clock", "cloud", "donut", "apple", "camera", "sun", "door",
    "sailboat", "ladder", "star", "headphones", "flower", "palm tree",
    "mountain", "tent", "bicycle", "hot air balloon", "table", "envelope",
    "tree", "hamburger", "fish", "candle", "pizza", "birthday cake", "couch",
    "cactus", "dog", "windmill",
]
assert len(CLASSES) == 50 and len(set(CLASSES)) == 50

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{}.ndjson"
WORK = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("work")
DATA = WORK / "data"
ARTI = WORK / "artifacts"   # checkpoints, onnx, eval json — download these
IMG = 64
SUPERSAMPLE = 256           # rasterize here, then box-down to IMG (web app matches)
STROKE_W = 8                # stroke width at SUPERSAMPLE scale
BIN_THRESH = 32             # >thresh -> ink. Binary keeps train/serve parity exact
TRAIN_PER_CLASS = 12_000
CALIB_TOTAL = 60_000
US_CAP_FRAC = 0.25          # baseline: max share of any single country per class
SEEDS = [1, 2, 3]
BALANCED_SEED = 101         # trained on the balanced resample
EPOCHS = 8
BATCH = 512
LR = 3e-3
SPLIT_SEED = 20260809

MAX_LINES_PER_FILE = None   # SMOKE only: cap lines streamed per class file
if SMOKE:
    CLASSES = CLASSES[:3]
    TRAIN_PER_CLASS = 400
    CALIB_TOTAL = 150
    EPOCHS = 1
    BATCH = 128
    SEEDS = [1, 2]
    MAX_LINES_PER_FILE = 20_000
    print("*** SMOKE MODE — tiny run, results meaningless ***")

# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------

def bucket(key_id: str) -> int:
    """Stable 0-99 bucket from key_id. <85 train pool, 85-89 calib, >=90 test."""
    return int(hashlib.md5(key_id.encode()).hexdigest()[:8], 16) % 100


def rasterize(drawing, size=IMG, src=SUPERSAMPLE, width=STROKE_W, thresh=BIN_THRESH):
    """Simplified strokes (0..255 coords) -> binary {0,1} uint8 [size, size].

    Supersample at `src`, box-downsample to `size`, threshold. The web app
    reproduces this exact path on a canvas (offscreen 256 -> drawImage 64 ->
    threshold), so training and serving see the same pixels.
    """
    from PIL import Image, ImageDraw
    img = Image.new("L", (src, src), 0)
    d = ImageDraw.Draw(img)
    for stroke in drawing:
        pts = list(zip(stroke[0], stroke[1]))
        if len(pts) == 1:
            r = width // 2
            d.ellipse([pts[0][0] - r, pts[0][1] - r, pts[0][0] + r, pts[0][1] + r], fill=255)
        else:
            d.line(pts, fill=255, width=width, joint="curve")
    a = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.uint8)
    return (a > thresh).astype(np.uint8)


def pack(imgs):
    """[N, 64, 64] binary -> [N, 512] bit-packed uint8 (8x smaller on disk)."""
    return np.packbits(imgs.reshape(len(imgs), -1), axis=1)


def unpack(packed):
    """[N, 512] -> [N, 64, 64] uint8 in {0,1}."""
    return np.unpackbits(packed, axis=1).reshape(-1, IMG, IMG)


def _raster_chunk(drawings):
    return pack(np.stack([rasterize(dr) for dr in drawings]))


def rasterize_packed(drawings, pool=None):
    """Parallel rasterize + bit-pack -> [N, 512] uint8."""
    if not drawings:
        return np.zeros((0, IMG * IMG // 8), dtype=np.uint8)
    chunk = 2000
    chunks = [drawings[i:i + chunk] for i in range(0, len(drawings), chunk)]
    if pool is None:
        return np.concatenate([_raster_chunk(c) for c in chunks])
    return np.concatenate(list(pool.map(_raster_chunk, chunks)))


# ----------------------------------------------------------------------------
# Stage DATA
# ----------------------------------------------------------------------------

def stage_data():
    """Stream + split + sample + rasterize, one class at a time.

    Memory is bounded by ONE class's stroke pool (~130k drawings, a few hundred
    MB): each class is rasterized and bit-packed immediately, then its stroke
    vectors are dropped. Packed output for all four splits is ~1 GB total.
    """
    import requests

    DATA.mkdir(parents=True, exist_ok=True)
    ARTI.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SPLIT_SEED)
    calib_per_class = CALIB_TOTAL // len(CLASSES)
    cap = int(TRAIN_PER_CLASS * US_CAP_FRAC)

    acc = {t: {"x": [], "y": [], "country": []} for t in ("train", "balanced", "calib", "test")}
    test_keys = []
    showcase = {}

    with ProcessPoolExecutor(max_workers=max(os.cpu_count() - 1, 1)) as pool:
        for ci, cls in enumerate(CLASSES):
            url = BASE.format(urllib.parse.quote(cls))
            by_cc = defaultdict(list)     # train-bucket drawings, keyed by country
            calib_pool, test_rows = [], []
            n_pool = 0
            with requests.get(url, stream=True, timeout=180) as r:
                r.raise_for_status()
                for ln_no, line in enumerate(r.iter_lines()):
                    if MAX_LINES_PER_FILE and ln_no >= MAX_LINES_PER_FILE:
                        break
                    if not line:
                        continue
                    d = jloads(line)
                    if not d.get("recognized"):
                        continue
                    b = bucket(d["key_id"])
                    cc = d.get("countrycode") or "??"
                    if b >= 90:
                        test_rows.append((d["drawing"], cc, d["key_id"]))
                    elif b >= 85:
                        calib_pool.append(d["drawing"])
                    else:
                        by_cc[cc].append(d["drawing"])
                        n_pool += 1

            # --- baseline sample: TRAIN_PER_CLASS, no country above US_CAP_FRAC ---
            capped, overflow = [], []
            for cc, items in by_cc.items():
                rng.shuffle(items)
                capped.extend((dr, cc) for dr in items[:cap])
                overflow.extend((dr, cc) for dr in items[cap:])
            rng.shuffle(capped)
            if len(capped) >= TRAIN_PER_CLASS:
                base = capped[:TRAIN_PER_CLASS]
            else:
                rng.shuffle(overflow)
                base = capped + overflow[: TRAIN_PER_CLASS - len(capped)]

            # --- balanced sample: draw countries with p ~ n^0.5 (flattens the n^1 skew) ---
            remaining = {cc: list(v) for cc, v in by_cc.items()}
            ccs = list(remaining)
            w = np.array([len(remaining[c]) ** 0.5 for c in ccs], dtype=np.float64)
            w /= w.sum()
            bal = []
            grng = np.random.default_rng(SPLIT_SEED + ci)
            for idx in grng.choice(len(ccs), size=min(TRAIN_PER_CLASS * 4, 200_000), p=w):
                cc = ccs[idx]
                if remaining[cc]:
                    bal.append((remaining[cc].pop(), cc))
                    if len(bal) == TRAIN_PER_CLASS:
                        break
            if len(bal) < TRAIN_PER_CLASS:      # top off from whatever is left
                flat = [(dr, cc) for cc, v in remaining.items() for dr in v]
                rng.shuffle(flat)
                bal.extend(flat[: TRAIN_PER_CLASS - len(bal)])

            rng.shuffle(calib_pool)
            calib = calib_pool[:calib_per_class]

            # --- rasterize + pack now, then drop this class's stroke vectors ---
            for tag, rows in (("train", base), ("balanced", bal)):
                acc[tag]["x"].append(rasterize_packed([dr for dr, _ in rows], pool))
                acc[tag]["y"].extend([ci] * len(rows))
                acc[tag]["country"].extend(cc for _, cc in rows)
            acc["calib"]["x"].append(rasterize_packed(calib, pool))
            acc["calib"]["y"].extend([ci] * len(calib))
            acc["test"]["x"].append(rasterize_packed([dr for dr, _, _ in test_rows], pool))
            acc["test"]["y"].extend([ci] * len(test_rows))
            acc["test"]["country"].extend(cc for _, cc, _ in test_rows)
            test_keys.extend(k for _, _, k in test_rows)
            showcase[cls] = [dr for dr, _, _ in test_rows[:12]]

            del by_cc, remaining, calib_pool, test_rows, base, bal, capped, overflow
            print(f"[{ci + 1:02d}/{len(CLASSES)}] {cls:16s} pool={n_pool:>7,} "
                  f"test={len(acc['test']['y']) :>7,} (cum)", flush=True)

    for tag, d in acc.items():
        x = np.concatenate(d["x"]) if d["x"] else np.zeros((0, IMG * IMG // 8), np.uint8)
        np.save(DATA / f"{tag}_x.npy", x)
        np.save(DATA / f"{tag}_y.npy", np.asarray(d["y"], dtype=np.int64))
        meta = {"n": len(d["y"]), "packed": True, "img": IMG}
        if d["country"]:
            meta["country"] = d["country"]
        if tag == "test":
            meta["key_id"] = test_keys
        (DATA / f"{tag}_meta.json").write_text(json.dumps(meta))
        print(f"  saved {tag}: packed{x.shape} = {x.nbytes / 1e6:.0f} MB, n={len(d['y']):,}")

    (ARTI / "showcase_strokes.json").write_text(json.dumps(showcase))
    (ARTI / "classes.json").write_text(json.dumps(CLASSES))
    (ARTI / "preprocess.json").write_text(json.dumps({
        "img": IMG, "supersample": SUPERSAMPLE, "stroke_width": STROKE_W,
        "binarize_threshold": BIN_THRESH,
        "note": "web app must reproduce: draw at supersample with stroke_width, "
                "downsample to img, ink = value > binarize_threshold",
    }))


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

def build_model():
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    return nn.Sequential(
        block(1, 32), block(32, 64), block(64, 128), block(128, 192),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Dropout(0.2), nn.Linear(192, len(CLASSES)),
    )


# ----------------------------------------------------------------------------
# Stage TRAIN
# ----------------------------------------------------------------------------

def stage_train():
    ARTI.mkdir(parents=True, exist_ok=True)
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")

    def load(tag):
        x = np.load(DATA / f"{tag}_x.npy", mmap_mode="r")
        y = np.load(DATA / f"{tag}_y.npy")
        return x, y

    ca_x, ca_y = load("calib")

    def run_one(seed, tag):
        torch.manual_seed(seed)
        np.random.seed(seed)
        x, y = load(tag)
        n = len(y)
        model = build_model().to(dev)
        nparams = sum(p.numel() for p in model.parameters())
        print(f"\n=== seed {seed} on '{tag}' ({n:,} rows, {nparams / 1e6:.2f}M params) ===")
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
        steps = math.ceil(n / BATCH) * EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=steps, pct_start=0.05)
        scaler = torch.amp.GradScaler(dev)
        lossf = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

        idx = np.arange(n)
        for ep in range(EPOCHS):
            model.train()
            np.random.shuffle(idx)
            tot = correct = seen = 0
            for i in range(0, n, BATCH):
                bidx = np.sort(idx[i:i + BATCH])  # sorted gather is faster on memmap
                xb = torch.from_numpy(unpack(x[bidx])).to(dev, non_blocking=True)
                xb = xb.float().unsqueeze(1)
                # light augmentation: random +-3px shift
                if ep < EPOCHS - 1:
                    sx, sy = np.random.randint(-3, 4, 2)
                    xb = torch.roll(xb, shifts=(int(sx), int(sy)), dims=(2, 3))
                yb = torch.from_numpy(y[bidx]).to(dev)
                with torch.amp.autocast(dev):
                    out = model(xb)
                    loss = lossf(out, yb)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                tot += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                seen += len(yb)
            # quick calib-set accuracy as val proxy
            model.eval()
            cacc = evaluate_logits(model, ca_x, ca_y, dev)[1]
            print(f"  ep{ep + 1}: train_loss={tot / seen:.4f} train_acc={correct / seen:.4f} calib_acc={cacc:.4f}")
        torch.save(model.state_dict(), ARTI / f"cnn_seed{seed}.pt")
        return model

    for s in SEEDS:
        run_one(s, "train")
    run_one(BALANCED_SEED, "balanced")


def evaluate_logits(model, x, y, dev, return_logits=False):
    import torch
    logits = []
    with torch.no_grad():
        for i in range(0, len(y), 2048):
            xb = torch.from_numpy(unpack(x[i:i + 2048])).to(dev).float().unsqueeze(1)
            logits.append(model(xb).float().cpu())
    logits = torch.cat(logits)
    acc = (logits.argmax(1).numpy() == y).mean()
    if return_logits:
        return logits, acc
    return logits, acc


# ----------------------------------------------------------------------------
# Stage EVAL
# ----------------------------------------------------------------------------

def stage_eval():
    ARTI.mkdir(parents=True, exist_ok=True)
    import torch

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    te_x = np.load(DATA / "test_x.npy", mmap_mode="r")
    te_y = np.load(DATA / "test_y.npy")
    te_meta = json.loads((DATA / "test_meta.json").read_text())
    countries = np.asarray(te_meta["country"])
    ca_x = np.load(DATA / "calib_x.npy", mmap_mode="r")
    ca_y = np.load(DATA / "calib_y.npy")

    def load_model(seed):
        m = build_model().to(dev)
        m.load_state_dict(torch.load(ARTI / f"cnn_seed{seed}.pt", map_location=dev))
        m.eval()
        return m

    report = {"classes": CLASSES, "n_test": int(len(te_y)), "seeds": {}}
    probs_by_seed = {}
    for seed in SEEDS + [BALANCED_SEED]:
        m = load_model(seed)
        te_logits, te_acc = evaluate_logits(m, te_x, te_y, dev)
        ca_logits, _ = evaluate_logits(m, ca_x, ca_y, dev)
        T = fit_temperature(ca_logits, ca_y)
        probs = torch.softmax(te_logits / T, 1).numpy()
        probs_by_seed[seed] = probs
        report["seeds"][str(seed)] = {
            "test_top1": round(float(te_acc), 4),
            "test_top3": round(topk(te_logits.numpy(), te_y, 3), 4),
            "temperature": round(float(T), 4),
            "ece_before": round(ece(torch.softmax(te_logits, 1).numpy(), te_y), 4),
            "ece_after": round(ece(probs, te_y), 4),
        }
        print(seed, report["seeds"][str(seed)])

    # 3-seed ensemble (baseline seeds only)
    ens = np.mean([probs_by_seed[s] for s in SEEDS], axis=0)
    pred = ens.argmax(1)
    report["ensemble"] = {
        "test_top1": round(float((pred == te_y).mean()), 4),
        "test_top3": round(topk(ens, te_y, 3), 4),
        "ece": round(ece(ens, te_y), 4),
        "reliability_bins": reliability(ens, te_y),
    }
    print("ensemble", {k: v for k, v in report["ensemble"].items() if k != "reliability_bins"})

    # confusion + per-class
    NC = len(CLASSES)
    conf = np.zeros((NC, NC), dtype=int)
    for t, p in zip(te_y, pred):
        conf[t, p] += 1
    report["confusion"] = conf.tolist()
    report["per_class_acc"] = {CLASSES[i]: round(float(conf[i, i] / conf[i].sum()), 4) for i in range(NC)}

    # ---- fairness audit: per-country accuracy + bootstrap CIs, baseline vs balanced ----
    fair = {}
    correct_base = (pred == te_y)
    pred_bal = probs_by_seed[BALANCED_SEED].argmax(1)
    correct_bal = (pred_bal == te_y)
    rng = np.random.default_rng(SPLIT_SEED)
    min_cc_rows = 50 if SMOKE else 2000
    for cc in sorted(set(countries.tolist())):
        mask = countries == cc
        n = int(mask.sum())
        if n < min_cc_rows or cc == "??":
            continue
        accs_b = bootstrap_acc(correct_base[mask], rng)
        accs_l = bootstrap_acc(correct_bal[mask], rng)
        # per-class acc for this country (classes with >=30 rows)
        pc = {}
        for i in range(50):
            m2 = mask & (te_y == i)
            if m2.sum() >= 30:
                pc[CLASSES[i]] = round(float(correct_base[m2].mean()), 4)
        fair[cc] = {
            "n": n,
            "acc": round(float(correct_base[mask].mean()), 4),
            "ci95": [round(float(np.percentile(accs_b, 2.5)), 4), round(float(np.percentile(accs_b, 97.5)), 4)],
            "acc_balanced": round(float(correct_bal[mask].mean()), 4),
            "ci95_balanced": [round(float(np.percentile(accs_l, 2.5)), 4), round(float(np.percentile(accs_l, 97.5)), 4)],
            "per_class": pc,
        }
    report["fairness"] = fair
    if fair:
        accs = [v["acc"] for v in fair.values()]
        report["fairness_summary"] = {
            "n_countries": len(fair),
            "max_min_gap": round(max(accs) - min(accs), 4),
            "std": round(float(np.std(accs)), 4),
            "gap_balanced": round(max(v["acc_balanced"] for v in fair.values()) - min(v["acc_balanced"] for v in fair.values()), 4),
        }
        print("fairness:", report["fairness_summary"])
    else:
        report["fairness_summary"] = {"n_countries": 0}

    (ARTI / "eval_report.json").write_text(json.dumps(report))
    # compact app payload (no confusion matrix rows the UI never shows)
    app = {k: report[k] for k in ["classes", "ensemble", "per_class_acc", "fairness", "fairness_summary", "n_test"]}
    (ARTI / "app_eval.json").write_text(json.dumps(app))


def fit_temperature(logits, y):
    import torch
    T = torch.nn.Parameter(torch.ones(1) * 1.0)
    yt = torch.from_numpy(y)
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(logits / T.clamp(0.05, 10.0), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(0.05, 10.0))


def topk(scores, y, k):
    if k >= scores.shape[1]:
        return 1.0
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    return float(np.mean([y[i] in part[i] for i in range(len(y))]))


def ece(probs, y, bins=15):
    conf = probs.max(1)
    pred = probs.argmax(1)
    correct = pred == y
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def reliability(probs, y, bins=15):
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() >= 50:
            out.append({"bin": round(float(edges[i + 1]), 3), "conf": round(float(conf[m].mean()), 4),
                        "acc": round(float(correct[m].mean()), 4), "n": int(m.sum())})
    return out


def bootstrap_acc(correct, rng, iters=1000):
    n = len(correct)
    idx = rng.integers(0, n, size=(iters, n))
    return correct[idx].mean(axis=1)


# ----------------------------------------------------------------------------
# Stage EXPORT
# ----------------------------------------------------------------------------

def stage_export():
    ARTI.mkdir(parents=True, exist_ok=True)
    import torch

    for seed in SEEDS + [BALANCED_SEED]:
        m = build_model()
        m.load_state_dict(torch.load(ARTI / f"cnn_seed{seed}.pt", map_location="cpu"))
        m.eval()
        dummy = torch.zeros(1, 1, IMG, IMG)
        fp = ARTI / f"cnn_seed{seed}.onnx"
        torch.onnx.export(m, dummy, fp, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
            qp = ARTI / f"cnn_seed{seed}.int8.onnx"
            quantize_dynamic(str(fp), str(qp), weight_type=QuantType.QUInt8)
        except Exception as e:
            print(f"quantization skipped for seed {seed}: {e}")
            qp = fp
        # parity check on 256 random test images
        import onnxruntime as ort
        te_x = np.load(DATA / "test_x.npy", mmap_mode="r")
        xb = unpack(te_x[:256]).astype(np.float32)[:, None]
        with torch.no_grad():
            ref = torch.softmax(m(torch.from_numpy(xb)), 1).numpy()
        sess = ort.InferenceSession(str(qp), providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": xb})[0]
        out = np.exp(out) / np.exp(out).sum(1, keepdims=True)
        agree = (ref.argmax(1) == out.argmax(1)).mean()
        print(f"seed {seed}: onnx={fp.stat().st_size / 1e6:.2f}MB int8={qp.stat().st_size / 1e6:.2f}MB "
              f"argmax-agree={agree:.4f} max|dp|={np.abs(ref - out).max():.4f}")


if __name__ == "__main__":
    if RUN_DATA:
        stage_data()
    if RUN_TRAIN:
        stage_train()
    if RUN_EVAL:
        stage_eval()
    if RUN_EXPORT:
        stage_export()
    print("\nDone. Download /kaggle/working/artifacts/*")
