// Deep-ensemble inference in the browser.
//
// Three independently-seeded CNNs vote. Their mean gives the prediction; their
// disagreement is the honest uncertainty signal the Doubt Meter shows — a
// single model's softmax is confident even when it is wrong, which is the whole
// pedagogical point of the "When AI Doesn't Know" lab.

import * as ort from 'onnxruntime-web'

const MODEL_FILES = ['cnn_seed1.int8.onnx', 'cnn_seed2.int8.onnx', 'cnn_seed3.int8.onnx']

let state = null

export async function loadEnsemble(onProgress = () => {}) {
  if (state) return state
  const sessions = []
  for (const f of MODEL_FILES) {
    try {
      onProgress(`loading ${f}`)
      sessions.push(await ort.InferenceSession.create(`/models/${f}`, { executionProviders: ['wasm'] }))
    } catch {
      // Seeds 2/3 may not be exported yet during development, and a judge on a
      // slow link may lose one. One model is enough to run; the UI degrades to
      // "single model — no disagreement signal" rather than breaking.
      break
    }
  }
  if (!sessions.length) throw new Error('no ONNX models could be loaded')
  const [classes, temps] = await Promise.all([
    fetch('/models/classes.json').then(r => r.json()),
    fetch('/models/temperatures.json').then(r => r.json()).catch(() => null),
  ])
  state = { sessions, classes, temps: temps ?? MODEL_FILES.map(() => 1) }
  return state
}

export function isEnsemble() {
  return (state?.sessions.length ?? 0) > 1
}

function softmax(logits, T = 1) {
  let max = -Infinity
  for (const v of logits) if (v > max) max = v
  const exp = Array.from(logits, v => Math.exp((v - max) / T))
  const sum = exp.reduce((a, b) => a + b, 0)
  return exp.map(v => v / sum)
}

function entropy(p) {
  let h = 0
  for (const v of p) if (v > 1e-12) h -= v * Math.log(v)
  return h / Math.log(p.length) // normalized to 0..1
}

/**
 * Run every member on one 64x64 tensor.
 * Returns mean probabilities plus the two uncertainty signals.
 */
export async function predict(tensor) {
  const { sessions, classes, temps } = state
  const input = new ort.Tensor('float32', tensor, [1, 1, 64, 64])
  const perModel = []
  for (let i = 0; i < sessions.length; i++) {
    const out = await sessions[i].run({ input })
    perModel.push(softmax(out.logits.data, temps[i] ?? 1))
  }
  const mean = new Array(classes.length).fill(0)
  for (const p of perModel) for (let i = 0; i < p.length; i++) mean[i] += p[i] / perModel.length
  // disagreement: how often members pick different winners, plus spread on the
  // consensus class. Both are zero for a single model — reported honestly.
  const argmaxes = perModel.map(p => p.indexOf(Math.max(...p)))
  const top = mean.indexOf(Math.max(...mean))
  const votes = argmaxes.filter(a => a === top).length
  const spread = perModel.length > 1
    ? Math.max(...perModel.map(p => p[top])) - Math.min(...perModel.map(p => p[top]))
    : 0
  const order = mean.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0])
  return {
    top5: order.slice(0, 5).map(([p, i]) => ({ label: classes[i], p, index: i })),
    mean,
    entropy: entropy(mean),
    disagreement: 1 - votes / perModel.length,
    spread,
    nModels: perModel.length,
    perModelTop: argmaxes.map(i => classes[i]),
  }
}

/** Batch helper for stroke ablation: one run per leave-one-stroke-out variant. */
export async function predictMany(tensors) {
  const out = []
  for (const t of tensors) out.push(await predict(t))
  return out
}

export function doubtLevel({ entropy: h, disagreement, top5 }) {
  const conf = top5[0]?.p ?? 0
  const score = 0.5 * h + 0.3 * disagreement + 0.2 * (1 - conf)
  if (score < 0.15) return { level: 'confident', score, blurb: 'The network is sure.' }
  if (score < 0.32) return { level: 'fairly sure', score, blurb: 'Reasonably confident, with alternatives in play.' }
  if (score < 0.5) return { level: 'unsure', score, blurb: 'Genuinely torn between classes.' }
  return { level: 'lost', score, blurb: "It has no idea — and it can't tell you that unless you measure it." }
}
