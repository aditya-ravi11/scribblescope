import { useRef, useState } from 'react'
import DrawCanvas from '../components/DrawCanvas'
import { DoubtMeter } from '../components/Readouts'
import { doubtLevel } from '../lib/ensemble'
import { useLivePredict } from '../lib/useLivePredict'

// Deliberately paired: things the network was taught, and things it never was.
const KNOWN = ['house', 'bicycle', 'pizza', 'umbrella', 'snowman']
const UNKNOWN = ['giraffe', 'guitar', 'octopus', 'helicopter', 'your own face']

export default function LabOOD() {
  const canvas = useRef(null)
  const [log, setLog] = useState([])
  const [prompt, setPrompt] = useState(null)
  const { result, run, reset, settled } = useLivePredict()

  const record = async () => {
    if (!result || !prompt) return
    await settled()               // never log a prediction that is still catching up
    const d = doubtLevel(result)
    setLog(l => [{
      asked: prompt.word,
      known: prompt.known,
      said: result.top5[0].label,
      p: result.top5[0].p,
      entropy: result.entropy,
      level: d.level,
    }, ...l].slice(0, 8))
    canvas.current.clear()
    reset()
    setPrompt(null)
  }

  const known = log.filter(r => r.known)
  const unknown = log.filter(r => !r.known)
  const avg = rows => rows.length ? rows.reduce((a, r) => a + r.entropy, 0) / rows.length : null

  return (
    <div className="grid gap-8 lg:grid-cols-[420px_1fr]">
      <div className="space-y-3">
        <DrawCanvas ref={canvas} onChange={run} />
        <div className="space-y-2">
          <p className="text-xs uppercase tracking-wider text-slate-500">It was taught these</p>
          <div className="flex flex-wrap gap-1.5">
            {KNOWN.map(w => (
              <button key={w} onClick={() => { canvas.current.clear(); reset(); setPrompt({ word: w, known: true }) }}
                className={`rounded-full border px-3 py-1 text-xs ${prompt?.word === w
                  ? 'border-emerald-400/60 bg-emerald-500/15 text-emerald-200'
                  : 'border-slate-700 text-slate-400 hover:text-slate-200'}`}>{w}</button>
            ))}
          </div>
          <p className="text-xs uppercase tracking-wider text-slate-500">It was never taught these</p>
          <div className="flex flex-wrap gap-1.5">
            {UNKNOWN.map(w => (
              <button key={w} onClick={() => { canvas.current.clear(); reset(); setPrompt({ word: w, known: false }) }}
                className={`rounded-full border px-3 py-1 text-xs ${prompt?.word === w
                  ? 'border-rose-400/60 bg-rose-500/15 text-rose-200'
                  : 'border-slate-700 text-slate-400 hover:text-slate-200'}`}>{w}</button>
            ))}
          </div>
        </div>
        <button onClick={record} disabled={!result || !prompt}
          className="w-full rounded-lg bg-sky-500/90 px-4 py-2 text-sm font-medium text-slate-950 hover:bg-sky-400 disabled:opacity-40">
          {prompt ? `Log this attempt at “${prompt.word}”` : 'Pick a prompt above, then draw it'}
        </button>
      </div>

      <div className="space-y-4">
        {result ? <DoubtMeter result={result} /> : (
          <p className="text-sm text-slate-500">
            Pick a word above and draw it. Try one it knows, then one it doesn't — and watch
            whether its confidence actually tells you which was which.
          </p>
        )}

        {result && (
          <div className="rounded-xl border border-slate-800 p-4 text-sm">
            <p className="text-slate-300">
              It says <span className="font-semibold text-slate-100">{result.top5[0].label}</span>{' '}
              at {(result.top5[0].p * 100).toFixed(1)}%.
            </p>
            {prompt && !prompt.known && (
              <p className="mt-2 text-rose-300">
                “{prompt.word}” is not one of the 50 classes, so this answer is wrong by
                construction. Notice it is often <em>confident</em> anyway: the network divides
                100% among the words it has, and a high percentage means “this is the closest
                thing I know”, not “I am right”.
              </p>
            )}
          </div>
        )}

        {log.length > 0 && (
          <>
            {known.length > 0 && unknown.length > 0 && (
              <div className="rounded-xl border border-slate-700 bg-slate-900/60 p-4">
                <p className="text-sm text-slate-300">Your results so far</p>
                <p className="mt-1 text-xs text-slate-500">
                  If these two numbers are close, that is the finding: a single model's confidence
                  is a poor detector of what it has never seen. Disagreement between independently
                  trained members is the more honest signal.
                </p>
                <div className="mt-2 grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p className="text-emerald-300">Taught ({known.length})</p>
                    <p className="tabular-nums text-2xl text-slate-100">{avg(known).toFixed(3)}</p>
                    <p className="text-xs text-slate-500">mean entropy</p>
                  </div>
                  <div>
                    <p className="text-rose-300">Never taught ({unknown.length})</p>
                    <p className="tabular-nums text-2xl text-slate-100">{avg(unknown).toFixed(3)}</p>
                    <p className="text-xs text-slate-500">mean entropy</p>
                  </div>
                </div>
              </div>
            )}
            <ul className="space-y-1 text-xs">
              {log.map((r, i) => (
                <li key={i} className="flex items-center justify-between rounded border border-slate-800 px-3 py-2">
                  <span className={r.known ? 'text-emerald-300' : 'text-rose-300'}>{r.asked}</span>
                  <span className="text-slate-400">
                    → {r.said} · {(r.p * 100).toFixed(0)}% · H={r.entropy.toFixed(3)} · {r.level}
                  </span>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  )
}
