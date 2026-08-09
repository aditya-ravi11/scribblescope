import { useEffect, useState } from 'react'
import { loadEnsemble, isEnsemble } from './lib/ensemble'
import LabPredict from './labs/LabPredict'
import LabAblation from './labs/LabAblation'
import LabOOD from './labs/LabOOD'
import LabBias from './labs/LabBias'
import ParityCheck from './dev/ParityCheck'

const LABS = [
  { id: 'predict', n: 1, title: 'What does a network see?',
    blurb: 'Draw, and watch a real CNN ensemble think out loud.' },
  { id: 'ablation', n: 2, title: 'Erase a stroke',
    blurb: 'Find out which pen strokes the network was actually using.' },
  { id: 'ood', n: 3, title: "When AI doesn't know",
    blurb: "Draw something it was never taught. It answers anyway — often confidently." },
  { id: 'bias', n: 4, title: 'Does AI understand everyone?',
    blurb: 'The same model, measured across 44 countries.' },
]

export default function App() {
  const [tab, setTab] = useState('predict')
  const [status, setStatus] = useState('loading')
  const [classes, setClasses] = useState(null)

  useEffect(() => {
    if (location.hash === '#parity') { setTab('parity'); return }
    loadEnsemble(setStatus)
      .then(s => { setClasses(s.classes); setStatus('ready') })
      .catch(e => setStatus(`error: ${e.message}`))
  }, [])

  if (tab === 'parity') return <ParityCheck />

  return (
    <div className="mx-auto max-w-6xl px-6 py-10">
      <header className="mb-10">
        <h1 className="text-3xl font-bold tracking-tight text-slate-50">ScribbleScope</h1>
        <p className="mt-2 max-w-2xl text-slate-400">
          Google's game guesses your doodle. ScribbleScope shows you{' '}
          <em className="text-slate-200">why</em>, <em className="text-slate-200">how sure</em>, and{' '}
          <em className="text-slate-200">who it fails</em>.
        </p>
        <p className="mt-3 text-xs text-slate-500">
          {status === 'ready'
            ? `${isEnsemble() ? '3-model ensemble' : 'single model'} · ${classes?.length ?? 0} classes · running in your browser`
            : String(status)}
        </p>
      </header>

      <nav className="mb-8 flex flex-wrap gap-2">
        {LABS.map(l => (
          <button
            key={l.id}
            onClick={() => setTab(l.id)}
            className={`rounded-xl border px-4 py-2 text-left transition ${
              tab === l.id
                ? 'border-sky-500/60 bg-sky-500/10 text-slate-100'
                : 'border-slate-800 text-slate-400 hover:border-slate-700 hover:text-slate-200'
            }`}
          >
            <span className="block text-[11px] uppercase tracking-wider opacity-60">Lab {l.n}</span>
            <span className="block text-sm font-medium">{l.title}</span>
          </button>
        ))}
      </nav>

      <p className="mb-6 text-sm text-slate-400">{LABS.find(l => l.id === tab)?.blurb}</p>

      {status !== 'ready' ? (
        <p className="text-slate-500">{String(status)}…</p>
      ) : tab === 'predict' ? (
        <LabPredict classes={classes} />
      ) : tab === 'ablation' ? (
        <LabAblation />
      ) : tab === 'ood' ? (
        <LabOOD />
      ) : tab === 'bias' ? (
        <LabBias />
      ) : (
        <p className="rounded-xl border border-dashed border-slate-800 p-8 text-center text-slate-500">
          Lab under construction.
        </p>
      )}
    </div>
  )
}
