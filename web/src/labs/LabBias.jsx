import { useEffect, useMemo, useState } from 'react'

const NAMES = {
  US: 'United States', GB: 'United Kingdom', CA: 'Canada', DE: 'Germany', AU: 'Australia',
  RU: 'Russia', BR: 'Brazil', FI: 'Finland', SE: 'Sweden', CZ: 'Czechia', IT: 'Italy',
  KR: 'South Korea', TH: 'Thailand', FR: 'France', PH: 'Philippines', HU: 'Hungary',
  NL: 'Netherlands', ID: 'Indonesia', RO: 'Romania', IN: 'India', VN: 'Vietnam',
  SK: 'Slovakia', JP: 'Japan', AE: 'UAE', TW: 'Taiwan', UA: 'Ukraine', MY: 'Malaysia',
  NO: 'Norway', IE: 'Ireland', HR: 'Croatia', TR: 'Türkiye', NZ: 'New Zealand',
  HK: 'Hong Kong', RS: 'Serbia', DK: 'Denmark', AT: 'Austria', PT: 'Portugal',
  BG: 'Bulgaria', SA: 'Saudi Arabia', PL: 'Poland', ES: 'Spain', BE: 'Belgium',
  CH: 'Switzerland', SG: 'Singapore', MX: 'Mexico', AR: 'Argentina', GR: 'Greece',
  IL: 'Israel', ZA: 'South Africa', EG: 'Egypt',
}

export default function LabBias() {
  const [data, setData] = useState(null)
  const [sort, setSort] = useState('acc')
  const [showFix, setShowFix] = useState(false)

  useEffect(() => {
    fetch('/models/app_eval.json').then(r => r.json()).then(setData).catch(() => setData('err'))
  }, [])

  const rows = useMemo(() => {
    if (!data || data === 'err') return []
    return Object.entries(data.fairness)
      .map(([cc, v]) => ({ cc, ...v, delta: v.acc_balanced - v.acc }))
      .sort((a, b) => (sort === 'acc' ? a.acc - b.acc : sort === 'n' ? b.n - a.n : b.delta - a.delta))
  }, [data, sort])

  if (data === 'err') return <p className="text-slate-500">Evaluation data not available yet.</p>
  if (!data) return <p className="text-slate-500">Loading the audit…</p>

  const s = data.fairness_summary
  const best = rows.reduce((a, r) => (r.acc > a.acc ? r : a), rows[0])
  const worst = rows.reduce((a, r) => (r.acc < a.acc ? r : a), rows[0])
  const lo = Math.min(...rows.map(r => r.acc))
  const hi = Math.max(...rows.map(r => r.acc))
  const key = r => (showFix ? r.acc_balanced : r.acc)

  return (
    <div className="space-y-6">
      {data.placeholder && (
        <div className="rounded-xl border border-amber-500/50 bg-amber-500/10 p-3 text-sm text-amber-200">
          <strong>Placeholder data.</strong> Country codes and sample sizes are real; the accuracies
          are synthetic until the training run's evaluation stage finishes. Nothing here is a result yet.
        </div>
      )}

      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Countries audited" value={s.n_countries}
              sub={`each with ≥2,000 held-out drawings`} />
        <Stat label="Best-to-worst gap" value={`${(s.max_min_gap * 100).toFixed(1)} pts`}
              sub={`${NAMES[best.cc] ?? best.cc} vs ${NAMES[worst.cc] ?? worst.cc}`} />
        <Stat label="After rebalancing" value={`${(s.gap_balanced * 100).toFixed(1)} pts`}
              sub="same model size, flatter training mix" />
      </div>

      <div className="flex flex-wrap items-center gap-3">
        <div className="flex gap-1 text-xs">
          {[['acc', 'worst first'], ['n', 'most data'], ['delta', 'most improved']].map(([k, l]) => (
            <button key={k} onClick={() => setSort(k)}
              className={`rounded-lg border px-3 py-1.5 ${sort === k
                ? 'border-sky-500/60 bg-sky-500/10 text-slate-100'
                : 'border-slate-800 text-slate-400 hover:text-slate-200'}`}>{l}</button>
          ))}
        </div>
        <label className="ml-auto flex cursor-pointer items-center gap-2 text-xs text-slate-400">
          <input type="checkbox" checked={showFix} onChange={e => setShowFix(e.target.checked)}
                 className="accent-sky-500" />
          Show the rebalanced model instead
        </label>
      </div>

      <ul className="space-y-1">
        {rows.map(r => {
          const v = key(r)
          const pct = ((v - lo + 0.005) / Math.max(hi - lo + 0.01, 1e-6)) * 100
          return (
            <li key={r.cc} className="grid grid-cols-[150px_1fr_130px] items-center gap-3 text-sm">
              <span className="truncate text-slate-300" title={`${r.n.toLocaleString()} test drawings`}>
                {NAMES[r.cc] ?? r.cc}
                <span className="ml-1 text-[10px] text-slate-600">{r.n.toLocaleString()}</span>
              </span>
              <span className="h-4 rounded bg-slate-800/70">
                <span className={`block h-4 rounded transition-all duration-300 ${
                  v < lo + (hi - lo) * 0.33 ? 'bg-rose-400/80'
                  : v < lo + (hi - lo) * 0.66 ? 'bg-amber-400/80' : 'bg-emerald-400/80'}`}
                  style={{ width: `${Math.max(pct, 2)}%` }} />
              </span>
              <span className="text-right tabular-nums text-slate-400">
                {(v * 100).toFixed(1)}%
                {showFix && (
                  <span className={r.delta >= 0 ? 'ml-1 text-emerald-400' : 'ml-1 text-rose-400'}>
                    {r.delta >= 0 ? '+' : ''}{(r.delta * 100).toFixed(1)}
                  </span>
                )}
              </span>
            </li>
          )
        })}
      </ul>

      <details className="rounded-xl border border-slate-800 p-4 text-sm text-slate-400">
        <summary className="cursor-pointer text-slate-300">How this was measured, and what it does not prove</summary>
        <div className="mt-2 space-y-2">
          <p>
            Every drawing was assigned to train, calibration or test by hashing its ID, before any
            sampling — so no drawing can leak between splits. The test set was never subsampled,
            which is why {s.n_countries} countries clear 2,000 drawings each. Intervals are 95%
            bootstrap confidence intervals; where two bars' intervals overlap, treat them as tied.
          </p>
          <p>
            <strong className="text-slate-300">The honest caveat:</strong> country is not culture.
            A gap can reflect device, screen size, prompt translation, or how long people in a
            region tend to spend on a 20-second sketch — not only how they draw. What is
            defensible is narrower and still worth saying: this model serves some countries'
            drawings measurably worse than others, and the training mix is one lever that moves it.
          </p>
        </div>
      </details>
    </div>
  )
}

function Stat({ label, value, sub }) {
  return (
    <div className="rounded-xl border border-slate-800 p-4">
      <p className="text-xs uppercase tracking-wider text-slate-500">{label}</p>
      <p className="mt-1 text-2xl font-semibold tabular-nums text-slate-100">{value}</p>
      <p className="mt-0.5 text-xs text-slate-500">{sub}</p>
    </div>
  )
}
