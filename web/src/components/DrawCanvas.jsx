import { useEffect, useImperativeHandle, useRef, forwardRef, useState } from 'react'
import { normalizeStrokes } from '../lib/preprocess'

const SIZE = 420

/**
 * Pointer-driven drawing surface that emits Quick, Draw!-style stroke arrays.
 * Strokes are kept in canvas pixels and normalized on read, so undo is exact.
 */
const DrawCanvas = forwardRef(function DrawCanvas({ onChange, disabled = false }, ref) {
  const canvasRef = useRef(null)
  const strokes = useRef([])
  const current = useRef(null)
  const [isEmpty, setEmpty] = useState(true)

  const redraw = () => {
    const ctx = canvasRef.current.getContext('2d')
    ctx.fillStyle = '#0b0f19'
    ctx.fillRect(0, 0, SIZE, SIZE)
    ctx.strokeStyle = '#e9edf7'
    ctx.lineWidth = 13
    ctx.lineJoin = 'round'
    ctx.lineCap = 'round'
    for (const [xs, ys] of strokes.current) {
      if (xs.length < 2) {
        ctx.beginPath()
        ctx.arc(xs[0], ys[0], 6.5, 0, Math.PI * 2)
        ctx.fillStyle = '#e9edf7'
        ctx.fill()
        continue
      }
      ctx.beginPath()
      ctx.moveTo(xs[0], ys[0])
      for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i], ys[i])
      ctx.stroke()
    }
  }

  const emit = () => {
    setEmpty(strokes.current.length === 0)
    onChange?.(normalizeStrokes(strokes.current), strokes.current)
  }

  useEffect(redraw, [])

  useImperativeHandle(ref, () => ({
    clear() { strokes.current = []; current.current = null; redraw(); emit() },
    undo() { strokes.current.pop(); redraw(); emit() },
    load(raw) { strokes.current = raw.map(([xs, ys]) => [[...xs], [...ys]]); redraw(); emit() },
    rawStrokes: () => strokes.current,
  }))

  const pos = e => {
    const r = canvasRef.current.getBoundingClientRect()
    return [(e.clientX - r.left) * (SIZE / r.width), (e.clientY - r.top) * (SIZE / r.height)]
  }

  const down = e => {
    if (disabled) return
    // some browsers throw if the pointer id isn't currently active
    try { e.currentTarget.setPointerCapture(e.pointerId) } catch { /* non-fatal */ }
    const [x, y] = pos(e)
    current.current = [[x], [y]]
    strokes.current.push(current.current)
    redraw()
  }

  const move = e => {
    if (!current.current || disabled) return
    const [x, y] = pos(e)
    const [xs, ys] = current.current
    // drop sub-pixel jitter: fewer points means faster ablation re-runs
    if (Math.hypot(x - xs.at(-1), y - ys.at(-1)) < 2.5) return
    xs.push(x); ys.push(y)
    redraw()
  }

  const up = () => {
    if (!current.current) return
    current.current = null
    emit()
  }

  return (
    <div className="flex flex-col gap-3">
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        onPointerDown={down}
        onPointerMove={move}
        onPointerUp={up}
        onPointerLeave={up}
        className="w-full max-w-[420px] aspect-square touch-none rounded-2xl border border-slate-700/70 bg-[#0b0f19] shadow-inner cursor-crosshair"
        aria-label="Drawing canvas"
      />
      <div className="flex gap-2">
        <button
          onClick={() => ref.current.undo()}
          disabled={isEmpty}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-40"
        >Undo stroke</button>
        <button
          onClick={() => ref.current.clear()}
          disabled={isEmpty}
          className="rounded-lg border border-slate-700 px-3 py-1.5 text-sm text-slate-300 hover:bg-slate-800 disabled:opacity-40"
        >Clear</button>
      </div>
    </div>
  )
})

export default DrawCanvas
