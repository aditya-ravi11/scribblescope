# Model assets

**`cnn_seed1.int8.onnx` here is currently a 1-EPOCH DEV EXPORT** (91.6% top-1),
used to build the UI while the real training run finishes. Before submission,
replace it with the final seeds and add:

- `cnn_seed2.int8.onnx`, `cnn_seed3.int8.onnx` — the other ensemble members
- `temperatures.json` — per-seed temperature-scaling constants from the calib split
- `app_eval.json` — fairness tables + reliability bins for Lab 4

`preprocess.json` records the exact rasterization parameters the training set was
built with. `src/lib/preprocess.js` reimplements that path (including PIL's
bilinear resampling filter) and `#parity` asserts the two agree — re-run it after
any change to either side.
