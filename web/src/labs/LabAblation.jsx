import { useCallback, useRef, useState } from 'react'
import DrawCanvas from '../components/DrawCanvas'
import { bboxOf, normalizeStrokes, strokesToTensor } from '../lib/preprocess'
import { predict } from '../lib/ensemble'
import AdaPanel from '../components/AdaPanel'
import { explainAblation } from '../lib/adaFallback'

const HEAT = c => {
  // c in 0..1 -> slate (irrelevant) through amber to rose (load-bearing)
  if (c <= 0) return '#475569'
  const t = Math.min(c, 1)
  const r = Math.round(148 + t * (244 - 148))
  const g = Math.round(163 - t * (163 - 63))
  const b = Math.round(184 - t * (184 - 94))
  return `rgb(${r},${g},${b})`
}

export default function LabAblation() {
  const canvas = useRef(null)
  const [raw, setRaw] = useState([])
  const [base, setBase] = useState(null)
  const [contribs, setContribs] = useState(null)
  const [busy, setBusy] = useState(false)

  const onChange = useCallback((_norm, rawStrokes) => {
    setRaw(rawStrokes.map(([xs, ys]) => [[...xs], [...ys]]))
    setContribs(null)
    setBase(null)
  }, [])

  const analyse = async () => {
    if (raw.length < 2) return
    setBusy(true)
    // Pin every variant to the FULL drawing's box so removing a stroke cannot
    // rescale the survivors — otherwise the drop conflates two effects.
    const box = bboxOf(raw)
    const full = await predict(strokesToTensor(normalizeStrokes(raw, 8, box)))
    const target = full.top5[0]
    const rows = []
    for (let i = 0; i < raw.length; i++) {
      const without = raw.filter((_, j) => j !== i)
      const r = await predict(strokesToTensor(normalizeStrokes(without, 8, box)))
      rows.push({
        i,
        drop: target.p - r.mean[target.index],
        becomes: r.top5[0].label,
        flipped: r.top5[0].index !== target.index,
      })
    }
    setBase({ label: target.label, p: target.p })
    setContribs(rows)
    setBusy(false)
  }

  const maxDrop = contribs ? Math.max(...contribs.map(r => Math.abs(r.drop)), 1e-6) : 1

  return (
    <div className="grid gap-8 lg:grid-cols-[420px_1fr_320px]">
      <div className="space-y-3">
        <DrawCanvas ref={canvas} onChange={onChange} />
        <button
          onClick={analyse}
          disabled={raw.length < 2 || busy}
          className="w-full rounded-lg bg-sky-500/90 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-sky-400 disabled:opacity-40"
        >
          {busy ? 'Re-running the network…' : `Erase each stroke (${raw.length} runs)`}
        </button>
        {raw.length < 2 && <p className="text-xs text-slate-500">Draw at least two strokes.</p>}

        {contribs && (
          <svg viewBox="0 0 420 420" className="w-full max-w-[420px] rounded-2xl border border-slate-700/70 bg-[#0b0f19]">
            {raw.map((s, i) => {
              const c = contribs.find(r => r.i === i)
              const pts = s[0].map((x, k) => `${x},${s[1][k]}`).join(' ')
              return (
                <polyline
                  key={i} points={pts} fill="none"
                  stroke={HEAT(Math.max(c.drop, 0) / maxDrop)}
                  strokeWidth="13" strokeLinejoin="round" strokeLinecap="round"
                />
              )
            })}
          </svg>
        )}
      </div>

      <div className="space-y-4">
        {!contribs ? (
          <p className="text-sm text-slate-500">
            Draw something, then erase each stroke in turn. Whatever the network loses when a
            stroke disappears is what that stroke was worth to it.
          </p>
        ) : (
          <>
            <p className="text-sm text-slate-300">
              Whole drawing: <span className="font-semibold text-slate-100">{base.label}</span>{' '}
              at {(base.p * 100).toFixed(1)}%. Removing each stroke costs:
            </p>
            <ul className="space-y-2">
              {[...contribs].sort((a, b) => b.drop - a.drop).map(c => (
                <li key={c.i} className="rounded-lg border border-slate-800 p-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="flex items-center gap-2 text-slate-300">
                      <span className="inline-block h-3 w-3 rounded-full"
                            style={{ background: HEAT(Math.max(c.drop, 0) / maxDrop) }} />
                      Stroke {c.i + 1}
                    </span>
                    <span className="tabular-nums text-slate-400">
                      {c.drop >= 0 ? '−' : '+'}{Math.abs(c.drop * 100).toFixed(1)} pts
                    </span>
                  </div>
                  {c.flipped && (
                    <p className="mt-1 text-xs text-rose-300">
                      Without it the network says “{c.becomes}” instead.
                    </p>
                  )}
                </li>
              ))}
            </ul>
            <details className="rounded-xl border border-slate-800 p-4 text-sm text-slate-400">
              <summary className="cursor-pointer text-slate-300">Why this is real explainability</summary>
              <p className="mt-2">
                This is a leave-one-out ablation: the network is genuinely re-run once per stroke,
                so the numbers are measured, not estimated from gradients. Every variant is framed
                using the complete drawing's bounding box — otherwise erasing a stroke would also
                resize what's left, and you couldn't tell which effect you were seeing. A negative
                value means the network became <em>more</em> confident without that stroke.
              </p>
            </details>
          </>
        )}
      </div>

      <div className="h-[520px]">
        <AdaPanel
          summary={base && contribs ? explainAblation(base, contribs) : null}
          context={base && contribs && {
            lab: 'stroke ablation',
            whole: { label: base.label, p: +base.p.toFixed(4) },
            strokes: contribs.map(c => ({
              stroke: c.i + 1, probabilityDrop: +c.drop.toFixed(4),
              flipsTo: c.flipped ? c.becomes : null,
            })),
          }}
        />
      </div>
    </div>
  )
}
