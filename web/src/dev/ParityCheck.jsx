// Dev-only harness: does the browser's rasterization match the training set's?
//
// Any drift here is silent — the model would just get quietly worse on live
// input than the reported test metrics claim. This renders 100 Python-generated
// fixtures through the JS path and reports pixel agreement + prediction match.

import { useEffect, useState } from 'react'
import { strokesToTensor, unpackBase64 } from '../lib/preprocess'
import { loadEnsemble, predict } from '../lib/ensemble'

export default function ParityCheck() {
  const [rows, setRows] = useState(null)
  const [summary, setSummary] = useState('running…')

  useEffect(() => {
    (async () => {
      try {
        await loadEnsemble()
        const { fixtures } = await fetch('/models/parity_fixtures.json').then(r => r.json())
        const out = []
        for (const f of fixtures) {
          const js = strokesToTensor(f.strokes)
          const py = unpackBase64(f.px)
          let diff = 0, inkPy = 0, inkJs = 0
          for (let i = 0; i < py.length; i++) {
            if (js[i] !== py[i]) diff++
            inkPy += py[i]; inkJs += js[i]
          }
          const r = await predict(js)
          out.push({
            cls: f.cls,
            pxAgree: 1 - diff / py.length,
            inkPy, inkJs,
            jsTop: r.top5[0].label,
            pyTop: f.cls,
            predMatch: r.top5[0].index === f.top1,
          })
        }
        setRows(out)
        const agree = out.reduce((a, r) => a + r.pxAgree, 0) / out.length
        const match = out.filter(r => r.predMatch).length
        const inkRatio = out.reduce((a, r) => a + r.inkJs / Math.max(r.inkPy, 1), 0) / out.length
        setSummary(
          `pixel agreement ${(agree * 100).toFixed(2)}% · prediction match ${match}/${out.length} ` +
          `· ink ratio js/py ${inkRatio.toFixed(3)}`
        )
      } catch (e) {
        setSummary(`FAILED: ${e.message}`)
      }
    })()
  }, [])

  const worst = rows ? [...rows].sort((a, b) => a.pxAgree - b.pxAgree).slice(0, 8) : []

  return (
    <div className="p-6 font-mono text-sm text-slate-200">
      <h2 className="mb-2 text-lg">Preprocessing parity: browser vs training</h2>
      <p data-testid="parity-summary" className="mb-4 text-emerald-300">{summary}</p>
      {rows && (
        <table className="text-xs">
          <thead><tr className="text-slate-500">
            <th className="pr-4 text-left">worst 8 by pixel agreement</th>
            <th className="pr-4">px agree</th><th className="pr-4">ink py/js</th>
            <th className="pr-4">py top1</th><th>js top1</th>
          </tr></thead>
          <tbody>
            {worst.map((r, i) => (
              <tr key={i} className={r.predMatch ? '' : 'text-amber-300'}>
                <td className="pr-4">{r.cls}</td>
                <td className="pr-4 text-right tabular-nums">{(r.pxAgree * 100).toFixed(2)}%</td>
                <td className="pr-4 text-right tabular-nums">{r.inkPy}/{r.inkJs}</td>
                <td className="pr-4">{r.pyTop}</td><td>{r.jsTop}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
