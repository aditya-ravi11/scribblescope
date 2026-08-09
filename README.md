# ScribbleScope

**Google's game guesses your doodle. ScribbleScope shows you *why*, *how sure*, and *who it fails*.**

An interactive AI-literacy lab: draw for a real convolutional neural network ensemble — trained from scratch on ~600,000 Quick, Draw! doodles from 100+ countries — and watch it think, doubt itself, explain its reasoning stroke-by-stroke, and reveal (then partially fix) its cultural biases. All inference runs in your browser: no server, no upload, works offline.

Built solo for the [ML Empowerment Build Challenge 2.0](https://ml-empowerment-2.devpost.com/).

> 🚧 Build in progress — full technical writeup (metrics, fairness audit, model card) lands here by Aug 14.

## Repository layout

- `ml/` — data probe, Kaggle training pipeline, evaluation notebooks, exported eval artifacts
- `web/` — React + Vite + Tailwind app (onnxruntime-web in-browser inference)
- `docs/` — model card, instructor one-pager, figures

## Data

[Google Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) simplified `ndjson` (CC BY 4.0): 50 curated classes, stroke vectors + country codes. Splits are drawing-disjoint via `key_id` hash, fixed **before** any sampling; the fairness audit evaluates on a full 10% held-out pool (~650k drawings, 45+ countries with ≥2k test rows).
