import { useCallback, useEffect, useRef, useState } from 'react'
import DrawCanvas from '../components/DrawCanvas'
import { TopK, DoubtMeter } from '../components/Readouts'
import { strokesToTensor } from '../lib/preprocess'
import { predict } from '../lib/ensemble'

export default function LabPredict({ classes }) {
  const canvas = useRef(null)
  const [result, setResult] = useState(null)
  const [ms, setMs] = useState(null)
  const pending = useRef(false)
  const queued = useRef(null)

  // Inference is fast but not free; coalesce bursts of pointer events so the
  // canvas never blocks on a backlog of stale predictions.
  const run = useCallback(async strokes => {
    if (!strokes.length) { setResult(null); return }
    if (pending.current) { queued.current = strokes; return }
    pending.current = true
    const t0 = performance.now()
    const r = await predict(strokesToTensor(strokes))
    setMs(performance.now() - t0)
    setResult(r)
    pending.current = false
    if (queued.current) {
      const next = queued.current
      queued.current = null
      run(next)
    }
  }, [])

  useEffect(() => { canvas.current?.clear() }, [])

  return (
    <div className="grid gap-8 lg:grid-cols-[420px_1fr]">
      <div>
        <DrawCanvas ref={canvas} onChange={run} />
        <p className="mt-3 text-xs text-slate-500">
          Runs entirely on your device — nothing is uploaded.
          {ms != null && <> Last inference: <span className="tabular-nums">{ms.toFixed(0)} ms</span>.</>}
        </p>
      </div>

      <div className="space-y-5">
        <section>
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">
            What the network sees
          </h3>
          <TopK result={result} />
        </section>
        <DoubtMeter result={result} />
        <details className="rounded-xl border border-slate-800 p-4 text-sm text-slate-400">
          <summary className="cursor-pointer text-slate-300">What am I looking at?</summary>
          <p className="mt-2">
            Every bar is the ensemble's average probability for one of {classes?.length ?? 50} classes.
            The network never says "I don't know" — it always splits 100% across the classes it was
            taught. The Doubt Meter is what turns that into an honest signal: high entropy means the
            probability is spread thin, and disagreement means the three independently-trained
            members picked different answers.
          </p>
        </details>
      </div>
    </div>
  )
}
