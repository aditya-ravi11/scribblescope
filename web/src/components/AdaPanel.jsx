import { useEffect, useRef, useState } from 'react'
import { answerQuestion } from '../lib/adaFallback'

const bold = t => t.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
  p.startsWith('**') ? <strong key={i} className="text-slate-100">{p.slice(2, -2)}</strong> : p)

/**
 * Ada. `summary` is the rule-based reading of whatever the student just did —
 * always present, always derived from real outputs. The LLM only ever adds to
 * that, so losing it degrades depth rather than function.
 */
export default function AdaPanel({ summary, context }) {
  const [log, setLog] = useState([])
  const [q, setQ] = useState('')
  const [busy, setBusy] = useState(false)
  const [degraded, setDegraded] = useState(false)
  const end = useRef(null)

  useEffect(() => { end.current?.scrollIntoView({ block: 'nearest' }) }, [log, summary])

  const ask = async e => {
    e.preventDefault()
    const question = q.trim()
    if (!question || busy) return
    setQ('')
    setLog(l => [...l, { who: 'you', text: question }])
    setBusy(true)
    try {
      const r = await fetch('/api/ada', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, context }),
      })
      if (!r.ok) throw new Error(String(r.status))
      const { answer } = await r.json()
      setLog(l => [...l, { who: 'ada', text: answer }])
      setDegraded(false)
    } catch {
      setDegraded(true)
      setLog(l => [...l, { who: 'ada', text: answerQuestion(question), offline: true }])
    } finally {
      setBusy(false)
    }
  }

  return (
    <aside className="flex h-full flex-col rounded-2xl border border-slate-800 bg-slate-900/40">
      <header className="flex items-center gap-2 border-b border-slate-800 px-4 py-3">
        <span className="grid h-7 w-7 place-items-center rounded-full bg-sky-500/20 text-sm text-sky-300">A</span>
        <div>
          <p className="text-sm font-medium text-slate-200">Ada</p>
          <p className="text-[11px] text-slate-500">
            {degraded ? 'offline mode — reading the numbers directly' : 'reads the network\'s live output'}
          </p>
        </div>
      </header>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 py-3 text-sm">
        {summary ? (
          <p className="text-slate-300">{bold(summary)}</p>
        ) : (
          <p className="text-slate-500">Draw something and I'll talk you through what the network did.</p>
        )}
        {log.map((m, i) => (
          <p key={i} className={m.who === 'you'
            ? 'ml-6 rounded-lg bg-slate-800/70 px-3 py-2 text-slate-200'
            : 'text-slate-300'}>
            {bold(m.text)}
            {m.offline && <span className="ml-1 text-[10px] text-amber-400/80">(offline)</span>}
          </p>
        ))}
        {busy && <p className="text-slate-500">thinking…</p>}
        <div ref={end} />
      </div>

      <form onSubmit={ask} className="border-t border-slate-800 p-3">
        <input
          value={q}
          onChange={e => setQ(e.target.value)}
          maxLength={500}
          placeholder="Ask why…"
          className="w-full rounded-lg border border-slate-700 bg-slate-950/60 px-3 py-2 text-sm text-slate-200 placeholder:text-slate-600 focus:border-sky-500/60 focus:outline-none"
        />
      </form>
    </aside>
  )
}
