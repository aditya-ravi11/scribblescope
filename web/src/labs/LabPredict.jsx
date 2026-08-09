import { useEffect, useRef } from 'react'
import DrawCanvas from '../components/DrawCanvas'
import { TopK, DoubtMeter } from '../components/Readouts'
import { useLivePredict } from '../lib/useLivePredict'
import AdaPanel from '../components/AdaPanel'
import { explainPrediction } from '../lib/adaFallback'

export default function LabPredict({ classes }) {
  const canvas = useRef(null)
  const { result, ms, run } = useLivePredict()

  useEffect(() => { canvas.current?.clear() }, [])

  return (
    <div className="grid gap-8 lg:grid-cols-[420px_1fr_320px]">
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

      <div className="h-[520px]">
        <AdaPanel
          summary={explainPrediction(result)}
          context={result && {
            lab: 'live prediction',
            top5: result.top5.map(t => ({ label: t.label, p: +t.p.toFixed(4) })),
            entropy: +result.entropy.toFixed(4),
            models: result.nModels,
            disagreement: +result.disagreement.toFixed(3),
            memberVotes: result.perModelTop,
          }}
        />
      </div>
    </div>
  )
}
