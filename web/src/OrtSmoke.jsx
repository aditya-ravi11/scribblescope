// Day-0/1 fail-fast gate: prove onnxruntime-web runs our int8 CNN in-browser
// and agrees with the Python reference. Temporary component, removed later.
import { useEffect, useState } from 'react'
import * as ort from 'onnxruntime-web'

export default function OrtSmoke() {
  const [status, setStatus] = useState('loading onnxruntime-web…')

  useEffect(() => {
    (async () => {
      try {
        const t0 = performance.now()
        const session = await ort.InferenceSession.create('/models/cnn_seed1.int8.onnx', {
          executionProviders: ['wasm'],
        })
        const tLoad = performance.now() - t0
        const fx = await fetch('/models/parity_check.json').then(r => r.json())
        const results = []
        const t1 = performance.now()
        for (let i = 0; i < fx.images.length; i++) {
          const data = Float32Array.from(fx.images[i], v => v / 255.0)
          const input = new ort.Tensor('float32', data, [1, 1, 64, 64])
          const out = await session.run({ input })
          const logits = out.logits.data
          let arg = 0
          for (let j = 1; j < logits.length; j++) if (logits[j] > logits[arg]) arg = j
          results.push({ pred: arg, want: fx.labels[i] })
        }
        const tInfer = (performance.now() - t1) / fx.images.length
        const agree = results.filter(r => r.pred === r.want).length
        setStatus(
          `ORT-WEB OK — load ${tLoad.toFixed(0)}ms, infer ${tInfer.toFixed(1)}ms/img, ` +
          `pred-vs-label ${agree}/${results.length} (smoke model, label match not required) — ` +
          `preds=${JSON.stringify(results.map(r => r.pred))}`
        )
      } catch (e) {
        setStatus(`ORT-WEB FAILED: ${e.message}`)
      }
    })()
  }, [])

  return <pre data-testid="ort-status" style={{ padding: 20, fontSize: 14 }}>{status}</pre>
}
