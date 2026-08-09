# ScribbleScope

**Google's game guesses your doodle. ScribbleScope shows you *why*, *how sure*, and *who it fails*.**

An interactive AI-literacy lab. You draw; a real convolutional neural network ensemble — trained
from scratch on ~600,000 Quick, Draw! doodles from 195 countries — thinks out loud, doubts itself,
explains which pen strokes it was actually using, and then shows you the people it serves worst.

Everything runs **in your browser**. No server, no upload, no account. It works offline after first
load and on a school Chromebook.

> Built solo for the [ML Empowerment Build Challenge 2.0](https://ml-empowerment-2.devpost.com/).

**Live demo:** _pending deploy_ · **Instructor guide:** [docs/instructor-guide.md](docs/instructor-guide.md) · **Parity harness:** `#parity`

---

## Problem statement

Students use AI daily and are taught about it with slides. They can define "confidence" and
"bias" without ever having seen either one *happen*. That gap matters: the misconception that a
model's 92% means "92% likely correct" is exactly the belief that makes people trust a system
outside the conditions it was built for.

The obstacle is practical. Demonstrating calibration, uncertainty, explainability or dataset bias
normally means a Python environment, a GPU, a dataset download, and an hour of setup — which rules
out most classrooms, and rules out every student without a capable laptop.

**Target users:** high-school and early-undergraduate students in free AI courses — the ML
Empowerment Foundation's own audience — and the volunteer instructors teaching them, including in
low-bandwidth classrooms and on shared devices.

## Solution overview

Four labs, each turning one abstract concept into something measured live on the student's own
drawing:

| Lab | Concept | What actually happens |
|---|---|---|
| **1 · What does a network see?** | Models output distributions | 50 probabilities update as you draw; a Doubt Meter reads entropy and inter-model disagreement |
| **2 · Erase a stroke** | Explainability by intervention | Each stroke is deleted and the network genuinely re-run; strokes are coloured by what their removal costs |
| **3 · When AI doesn't know** | Limits of a closed world | Draw something never in the 50 classes and watch it answer confidently anyway |
| **4 · Does AI understand everyone?** | Dataset skew becomes measurable harm | Per-country accuracy across 44 countries, with bootstrap intervals, and a rebalanced model to compare |

**Ada**, a built-in tutor, narrates each result from the network's real outputs. She works with no
language model configured at all — see [Ada](#ada-the-tutor-that-still-works-when-it-breaks).

## Key features

- **Real trained model, not an API call.** A 4-block CNN (~850k parameters) trained from scratch;
  three independently-seeded members form a deep ensemble.
- **Entirely client-side inference.** Three quantized ONNX models (~0.87 MB each) run via
  onnxruntime-web at ~20 ms per prediction. Nothing a student draws leaves their device.
- **Measured explainability.** Leave-one-out stroke ablation, not gradient approximations.
- **Honest uncertainty.** Entropy plus ensemble disagreement, on temperature-calibrated outputs.
- **A fairness audit with real statistical weight** — 44 countries, each with ≥2,000 held-out
  drawings, reported with 95% bootstrap confidence intervals.
- **A classroom artifact:** a 40-minute lesson plan mapping each lab to a standard AI curriculum.

---

## Technical implementation

### Data

[Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) simplified `ndjson`
(CC BY 4.0), streamed directly from Google's public bucket — 3.16 GB across 50 hand-picked classes,
chosen for drawability and for documented cross-cultural variation (bread, power outlet, mailbox,
teapot, traffic light…). Only `recognized: true` drawings are used, as a label-noise control.

**Splits are assigned by hashing each drawing's `key_id`, before any sampling** — 85% train pool,
5% calibration, 10% test. Nothing can leak between splits, and because the test pool is *never*
subsampled it stays large enough for the fairness audit to be statistically meaningful.

| Split | Drawings | Purpose |
|---|---|---|
| Train (baseline) | 600,000 | Uniform draw — inherits the corpus's real skew (**US 44.6%**) |
| Train (mitigated) | 600,000 | Same size, any one country capped at 25%/class (**US 5.3%**) |
| Calibration | 60,000 | Temperature scaling only — never trained on |
| **Test** | **664,923** | Held out; 44 countries with ≥2,000 drawings each |

Images are rasterized at 256×256, downsampled to 64×64, thresholded to binary, and bit-packed to
512 bytes each — the whole corpus is ~1 GB, and binary keeps browser preprocessing exactly
reproducible (below).

### Model

4 blocks of (Conv-BN-ReLU ×2 → MaxPool) at 32/64/128/192 channels, global average pooling, dropout,
linear. ~850k parameters — deliberately small, so three copies fit in a browser tab. AdamW,
OneCycle schedule, label smoothing 0.05, ±3px translation augmentation that pads rather than wraps.

### Evaluation

- Top-1 / top-3 on the full 664,923-drawing test set, plus a 50×50 confusion matrix.
- **Calibration:** temperature scaling fitted on the held-out calibration split; ECE over 15 bins
  with reliability diagrams, reported before and after.
- **Uncertainty:** predictive entropy and disagreement across the three ensemble members.
- **Fairness:** per-country accuracy with 1,000-sample bootstrap 95% intervals, baseline vs
  mitigated model.

### Results

> **Status: the final training run is in progress.** These cells are filled from committed
> notebook output — no number appears here that isn't reproducible from the repo. Interim signal:
> a single member reached 94.6% calibration accuracy at epoch 4 of 6.

| Metric | Value |
|---|---|
| Ensemble top-1 (664,923 held-out) | _pending_ |
| Ensemble top-3 | _pending_ |
| ECE before / after temperature scaling | _pending_ |
| Best-to-worst country gap, baseline | _pending_ |
| Best-to-worst country gap, mitigated | _pending_ |

### Train/serve parity — the bug worth reading about

A browser must reproduce the training rasterizer *exactly*, or the model silently sees a different
distribution at serve time than the one every reported metric was measured on. Three real
mismatches, each found by comparing against 100 Python-generated fixtures:

1. **`drawImage` downsampling is browser-defined.** A judge on Safari could have seen different
   predictions than one on Chrome. Replaced with a port of PIL's bilinear resampling filter, so the
   pipeline is now deterministic across browsers.
2. **Canvas anti-aliases strokes; PIL's `ImageDraw` does not.** The grey fringe survived the ink
   threshold and inflated ink by 4.8%. Fixed by binarizing at the supersample stage.
3. **`lineCap: 'round'` vs PIL's flat polyline ends.**

Result: **98.3% pixel agreement, and predictions matching Python on 98/100 fixtures.** The harness
ships in the app at `#parity` as a permanent regression guard.

### Ada, the tutor that still works when it breaks

Ada receives the network's actual outputs — top-5 probabilities, entropy, member votes, ablation
drops — and explains *that* result rather than the concept in general.

She is explicitly **not** a dependency. Judging is precisely when an expired key or exhausted quota
would hurt most, so the rule-based path is the real product: it reads the same numbers and produces
specific prose ("its runner-up, table, is only 0.3% — it is not remotely torn"), plus deterministic
answers on the concepts students actually ask about. Verified with no key configured. A language
model, when available, adds depth and nothing else.

Server side: the key stays server-only, with per-IP rate limiting, capped output tokens, bounded
question length, and only whitelisted model-derived fields forwarded. The student's message is
treated as a question, never as an instruction that can change the tutor's rules.

---

## Technologies used

**ML:** PyTorch · NumPy · Pillow · ONNX + onnxruntime dynamic int8 quantization
**Web:** React · Vite · Tailwind CSS · onnxruntime-web (WASM)
**Backend:** one Vercel serverless function (the tutor); everything else is static
**Data:** Google Quick, Draw! (CC BY 4.0)

## Social impact statement

The most useful thing this project does is make a specific misconception falsifiable in about
thirty seconds. Students are taught that AI "can be biased" and that confidence "isn't the same as
accuracy". Here they draw a giraffe the model was never taught, watch it answer *chair* at 92%, and
discover the difference themselves — then open Lab 4 and see, with confidence intervals, that the
same model is measurably worse at understanding some countries' drawings than others.

It is free, needs no account, uploads nothing, and runs on hardware schools already own. The
[instructor guide](docs/instructor-guide.md) makes it adoptable as a lesson rather than a demo —
including by the foundation running this challenge, whose own curriculum covers exactly these
topics.

## Limitations

We would rather state these than have them found.

- **50 classes is a closed world.** Anything outside it is answered wrongly by construction. That's
  Lab 3's lesson, but it is still a limitation.
- **Country is not culture.** A per-country gap can reflect device, screen size, prompt
  translation, or how long people spend on a 20-second sketch. The defensible claim is narrower and
  still worth making: this model serves some countries' drawings measurably worse, and the training
  mix is one lever that moves it.
- **Quick, Draw! is not a representative sample of humanity** — it is people who found a Google
  game, on the devices they owned, in 2016–17.
- **Ensemble members are not identically trained.** Seed 1 trained 3 epochs at batch 512, then
  warm-restarted for 3 more at batch 256 after a memory bug was fixed mid-run; seeds 2 and 3 ran
  straight through. Members of a deep ensemble need not match, but the asymmetry is disclosed.
- **Stroke ablation measures what the model uses, not what it "understands".**
- Rasterization parity is 98.3%, not 100%. The residual is edge pixels.

## Reproduce it

```bash
python ml/data_probe.py                       # class/country audit (HTTP range reads, no download)
python ml/kaggle/scribblescope_pipeline.py    # data -> train -> evaluate -> export ONNX
npm --prefix web install && npm --prefix web run dev
```

`SCRIBBLE_SMOKE=1` runs the whole pipeline end-to-end on 3 classes in a couple of minutes. The
pipeline runs unmodified on a Kaggle/Colab GPU notebook or a laptop; on Apple silicon it selects
MPS automatically.

## Engineering notes

Things that broke, because the log is more useful than the highlight reel:

- **Training silently ran on CPU** — the device check tested for CUDA and never MPS. 7× slower.
- **Epoch time went 22 → 89 minutes.** It looked like thermal throttling; it wasn't. PyTorch's MPS
  caching allocator grew unbounded to 7 GB RSS and pushed a 16 GB machine 4.7 GB into swap. Freeing
  the cache periodically brought it back to 0.5 GB.
- **The fairness experiment was inverted.** A per-country cap had been applied to the *baseline*,
  making it flatter (US 5.3%) than the "mitigated" variant (US 10.1%) — the before/after would have
  shown nothing. Caught by validating the data before training on it.
- **The data build needed 17 GB of RAM** on a machine with 16, because it retained every sampled
  drawing. Now streams one class at a time, bounded at ~2 GB.
- **Live predictions could go stale.** Inference requests arriving mid-draw were dropped rather than
  queued, so the panel described an earlier version of the drawing — a house logged as "television"
  at 37%, which is exactly what its half-drawn square looks like.

## Licence

Code MIT. Quick, Draw! data is CC BY 4.0 from Google Creative Lab.
