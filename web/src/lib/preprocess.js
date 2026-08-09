// Mirror of ml/kaggle/scribblescope_pipeline.py::rasterize.
//
// Training rasterized every drawing by stroking at 256x256 with width 8,
// downsampling to 64x64, then thresholding at >32. The browser must reproduce
// that exactly or the model sees a different distribution at serve time than
// it saw at train time. Parity is asserted against Python-generated fixtures
// in ParityCheck.jsx — change nothing here without re-running that.

export const PRE = { img: 64, supersample: 256, strokeWidth: 8, threshold: 32 }

/** Stroke-format drawing ([[xs],[ys]] per stroke, 0..255) -> 256x256 canvas. */
export function strokesToCanvas(strokes, size = PRE.supersample) {
  const c = document.createElement('canvas')
  c.width = c.height = size
  const ctx = c.getContext('2d', { willReadFrequently: true })
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, size, size)
  ctx.strokeStyle = '#fff'
  ctx.fillStyle = '#fff'
  ctx.lineWidth = PRE.strokeWidth * (size / PRE.supersample)
  ctx.lineJoin = 'round'   // PIL's joint="curve"
  ctx.lineCap = 'butt'     // PIL's ImageDraw.line does NOT round polyline ends
  const s = size / PRE.supersample
  for (const [xs, ys] of strokes) {
    if (xs.length === 1) {
      ctx.beginPath()
      ctx.arc(xs[0] * s, ys[0] * s, ctx.lineWidth / 2, 0, Math.PI * 2)
      ctx.fill()
      continue
    }
    ctx.beginPath()
    ctx.moveTo(xs[0] * s, ys[0] * s)
    for (let i = 1; i < xs.length; i++) ctx.lineTo(xs[i] * s, ys[i] * s)
    ctx.stroke()
  }
  return c
}

/**
 * Coefficients for one axis of PIL's `Image.resize(..., BILINEAR)`.
 * Ported from PIL's precompute_coeffs so the browser does not depend on
 * `drawImage` downsampling, which differs across browsers and does not match
 * what the training set was rasterized with.
 */
function coeffs(inSize, outSize) {
  const scale = inSize / outSize
  const support = 1.0 * scale // bilinear support 1.0, scaled for downsampling
  const out = []
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale
    const xmin = Math.max(Math.floor(center - support + 0.5), 0)
    const xmax = Math.min(Math.ceil(center + support + 0.5), inSize)
    const k = []
    let sum = 0
    for (let x = xmin; x < xmax; x++) {
      const t = Math.abs((x - center + 0.5) / scale)
      const w = t < 1 ? 1 - t : 0
      k.push(w)
      sum += w
    }
    if (sum !== 0) for (let i = 0; i < k.length; i++) k[i] /= sum
    out.push({ xmin, k })
  }
  return out
}

let COEFF = null

/** Any canvas -> Float32Array(64*64) of {0,1}, bit-identical to the training path. */
export function canvasToTensor(src) {
  const n = PRE.img
  const s = PRE.supersample
  const big = src.width === s
    ? src
    : (() => {
        const c = document.createElement('canvas')
        c.width = c.height = s
        c.getContext('2d').drawImage(src, 0, 0, s, s)
        return c
      })()
  const { data } = big.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, s, s)
  COEFF ??= coeffs(s, n)

  // PIL's ImageDraw renders hard-edged strokes; canvas anti-aliases them, which
  // leaves a grey fringe that survives the low ink threshold and inflates ink by
  // ~5%. Collapse the supersampled image to binary first, as PIL's already is.
  const hard = new Uint8Array(s * s)
  for (let i = 0; i < s * s; i++) hard[i] = data[i * 4] > 127 ? 255 : 0

  // horizontal pass, rounded to 8-bit like PIL does between passes
  const mid = new Uint8ClampedArray(s * n)
  for (let y = 0; y < s; y++) {
    for (let x = 0; x < n; x++) {
      const { xmin, k } = COEFF[x]
      let acc = 0
      for (let i = 0; i < k.length; i++) acc += hard[y * s + xmin + i] * k[i]
      mid[y * n + x] = Math.round(acc)
    }
  }
  // vertical pass, then threshold
  const out = new Float32Array(n * n)
  for (let y = 0; y < n; y++) {
    const { xmin, k } = COEFF[y]
    for (let x = 0; x < n; x++) {
      let acc = 0
      for (let i = 0; i < k.length; i++) acc += mid[(xmin + i) * n + x] * k[i]
      out[y * n + x] = Math.round(acc) > PRE.threshold ? 1 : 0
    }
  }
  return out
}

export function strokesToTensor(strokes) {
  return canvasToTensor(strokesToCanvas(strokes))
}

/** Bounding box of raw strokes, or null when there is no ink. */
export function bboxOf(raw) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity
  for (const [xs, ys] of raw) {
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < minX) minX = xs[i]
      if (xs[i] > maxX) maxX = xs[i]
      if (ys[i] < minY) minY = ys[i]
      if (ys[i] > maxY) maxY = ys[i]
    }
  }
  return minX === Infinity ? null : { minX, minY, maxX, maxY }
}

/** Normalize raw pointer strokes (canvas px) into the 0..255 box the model expects.
 *  Quick, Draw! art is scaled to fill its bounding box, so live input must be too.
 *
 *  `box` pins the framing to another drawing's extent. Stroke ablation needs this:
 *  re-normalizing each leave-one-out variant would rescale the survivors, so the
 *  measured probability drop would confound "this stroke mattered" with "the
 *  drawing got resized". */
export function normalizeStrokes(raw, pad = 8, box = null) {
  const b = box ?? bboxOf(raw)
  if (!b) return []
  const { minX, minY, maxX, maxY } = b
  const span = Math.max(maxX - minX, maxY - minY, 1)
  const scale = (255 - 2 * pad) / span
  // centre the shorter axis so aspect ratio is preserved
  const offX = pad + ((255 - 2 * pad) - (maxX - minX) * scale) / 2
  const offY = pad + ((255 - 2 * pad) - (maxY - minY) * scale) / 2
  return raw.map(([xs, ys]) => [
    xs.map(x => (x - minX) * scale + offX),
    ys.map(y => (y - minY) * scale + offY),
  ])
}

/** Unpack a base64 bit-packed 64x64 fixture (Python np.packbits) for comparison. */
export function unpackBase64(b64) {
  const bin = atob(b64)
  const out = new Float32Array(PRE.img * PRE.img)
  for (let i = 0; i < bin.length; i++) {
    const byte = bin.charCodeAt(i)
    for (let b = 0; b < 8; b++) out[i * 8 + b] = (byte >> (7 - b)) & 1
  }
  return out
}
