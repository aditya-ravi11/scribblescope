import { doubtLevel } from '../lib/ensemble'

export function TopK({ result }) {
  if (!result) return <p className="text-sm text-slate-500">Draw something to see the network's guesses.</p>
  return (
    <ul className="space-y-2">
      {result.top5.map((r, i) => (
        <li key={r.label}>
          <div className="flex justify-between text-sm">
            <span className={i === 0 ? 'font-semibold text-slate-100' : 'text-slate-400'}>{r.label}</span>
            <span className="tabular-nums text-slate-400">{(r.p * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-1 h-2 rounded-full bg-slate-800">
            <div
              className={`h-2 rounded-full transition-[width] duration-200 ${i === 0 ? 'bg-sky-400' : 'bg-slate-600'}`}
              style={{ width: `${Math.max(r.p * 100, 1)}%` }}
            />
          </div>
        </li>
      ))}
    </ul>
  )
}

const DOUBT_COLOR = {
  confident: 'text-emerald-300 border-emerald-500/40 bg-emerald-500/10',
  'fairly sure': 'text-sky-300 border-sky-500/40 bg-sky-500/10',
  unsure: 'text-amber-300 border-amber-500/40 bg-amber-500/10',
  lost: 'text-rose-300 border-rose-500/40 bg-rose-500/10',
}

export function DoubtMeter({ result }) {
  if (!result) return null
  const d = doubtLevel(result)
  const pct = Math.min(d.score / 0.65, 1) * 100
  return (
    <div className={`rounded-xl border p-4 ${DOUBT_COLOR[d.level]}`}>
      <div className="flex items-baseline justify-between">
        <span className="text-xs uppercase tracking-wider opacity-80">Doubt meter</span>
        <span className="text-sm font-semibold">{d.level}</span>
      </div>
      <div className="mt-2 h-2 rounded-full bg-black/30">
        <div className="h-2 rounded-full bg-current transition-[width] duration-300" style={{ width: `${pct}%` }} />
      </div>
      <p className="mt-2 text-xs opacity-90">{d.blurb}</p>
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-[11px] opacity-80">
        <dt>Entropy</dt><dd className="text-right tabular-nums">{result.entropy.toFixed(3)}</dd>
        <dt>Model disagreement</dt>
        <dd className="text-right tabular-nums">
          {result.nModels > 1 ? `${(result.disagreement * 100).toFixed(0)}%` : 'n/a (1 model)'}
        </dd>
        {result.nModels > 1 && (<>
          <dt>Spread on top class</dt>
          <dd className="text-right tabular-nums">{(result.spread * 100).toFixed(1)} pts</dd>
        </>)}
      </dl>
      {result.nModels > 1 && (
        <p className="mt-2 text-[11px] opacity-70">
          Members voted: {result.perModelTop.join(' · ')}
        </p>
      )}
    </div>
  )
}
