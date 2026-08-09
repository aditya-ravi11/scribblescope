import { useCallback, useRef, useState } from 'react'
import { strokesToTensor } from './preprocess'
import { predict } from './ensemble'

/**
 * Live inference against a drawing that is still changing.
 *
 * Inference takes ~20ms while pointer events arrive far faster, so requests must
 * be coalesced. Dropping the newest request is the tempting shortcut and it is
 * wrong: the displayed prediction then describes an earlier version of the
 * drawing. (A half-drawn house is a convincing television.) Always keep the
 * latest strokes and re-run once the in-flight call returns.
 */
export function useLivePredict() {
  const [result, setResult] = useState(null)
  const [ms, setMs] = useState(null)
  const busy = useRef(false)
  const latest = useRef(null)

  const run = useCallback(async function go(strokes) {
    if (!strokes?.length) { setResult(null); latest.current = null; return }
    if (busy.current) { latest.current = strokes; return }
    busy.current = true
    const t0 = performance.now()
    const r = await predict(strokesToTensor(strokes))
    setMs(performance.now() - t0)
    setResult(r)
    busy.current = false
    if (latest.current) {
      const next = latest.current
      latest.current = null
      go(next)
    }
  }, [])

  const reset = useCallback(() => { setResult(null); latest.current = null }, [])

  /** Resolves once no further work is queued — lets callers log a settled result. */
  const settled = useCallback(async () => {
    while (busy.current || latest.current) await new Promise(r => setTimeout(r, 15))
  }, [])

  return { result, ms, run, reset, settled }
}
