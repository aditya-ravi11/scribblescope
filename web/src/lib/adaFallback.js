// Rule-based explanations derived from the model's own numbers.
//
// This is not a stub. It is what runs whenever the LLM is unavailable — no key
// configured, quota exhausted, rate limited, offline — which during judging is
// exactly when a broken tutor would be most costly. Everything below is
// computed from real outputs, so the lab still teaches with the network off.

const pct = x => `${(x * 100).toFixed(1)}%`

export function explainPrediction(result) {
  if (!result) return null
  const [a, b] = result.top5
  const margin = a.p - b.p
  const out = []

  out.push(`The ensemble settled on **${a.label}** at ${pct(a.p)}.`)

  if (margin > 0.6) {
    out.push(`Its runner-up, ${b.label}, is only ${pct(b.p)} — it is not remotely torn.`)
  } else if (margin > 0.15) {
    out.push(`${b.label} trails at ${pct(b.p)}; there is a real second opinion here, just not a close one.`)
  } else {
    out.push(`But ${b.label} is right behind at ${pct(b.p)} — a ${pct(margin)} margin. Two more pen strokes could flip this.`)
  }

  if (result.entropy > 0.5) {
    out.push(`Entropy is ${result.entropy.toFixed(2)} out of 1, meaning the probability is smeared across many classes rather than concentrated. That is what "confused" looks like numerically.`)
  } else if (result.entropy < 0.1) {
    out.push(`Entropy is ${result.entropy.toFixed(3)} — almost all the probability sits on one class.`)
  }

  if (result.nModels > 1 && result.disagreement > 0) {
    out.push(`Worth noticing: the ${result.nModels} independently-trained members did **not** all agree (${result.perModelTop.join(', ')}). Disagreement between models is a more honest warning sign than any single model's confidence.`)
  } else if (result.nModels === 1) {
    out.push(`Only one model is loaded, so there is no disagreement signal to read — confidence is all you have, and confidence alone is not a reliable guide.`)
  }

  return out.join(' ')
}

export function explainOOD(result, prompt) {
  if (!result || !prompt) return null
  if (prompt.known) {
    return `${prompt.word} is one of the 50 classes it was trained on, so a confident answer here is at least *possible* to be right. It said ${result.top5[0].label} at ${pct(result.top5[0].p)}.`
  }
  return [
    `**${prompt.word} was never in the training set**, so whatever it says is wrong by construction — it said ${result.top5[0].label} at ${pct(result.top5[0].p)}.`,
    result.top5[0].p > 0.6
      ? `And notice it is *confident*. This is the uncomfortable part: a softmax always sums to 100% across the classes it knows, so "92%" means "closest thing I have", not "I am right". Confidence cannot tell you the answer was never available.`
      : `Here its probability did spread out, which is the behaviour people expect. It does not always happen — that is why uncertainty has to be measured rather than assumed.`,
  ].join(' ')
}

export function explainAblation(base, contribs) {
  if (!contribs?.length) return null
  const sorted = [...contribs].sort((x, y) => y.drop - x.drop)
  const top = sorted[0]
  const bottom = sorted[sorted.length - 1]
  const out = [
    `The whole drawing reads as **${base.label}** at ${pct(base.p)}.`,
    `Stroke ${top.i + 1} is carrying it: removing that one stroke costs ${(top.drop * 100).toFixed(1)} points${top.flipped ? ` and the answer becomes "${top.becomes}"` : ''}.`,
  ]
  if (Math.abs(bottom.drop) < 0.02) {
    out.push(`Stroke ${bottom.i + 1}, by contrast, is worth almost nothing (${(bottom.drop * 100).toFixed(1)} points) — the network was barely using it.`)
  }
  if (bottom.drop < -0.01) {
    out.push(`Stroke ${bottom.i + 1} actually *hurt*: the network got more confident once it was gone.`)
  }
  out.push(`This is measured, not estimated — the network really was re-run once per stroke.`)
  return out.join(' ')
}

export function explainBias(summary, worst, best) {
  if (!summary) return null
  return [
    `Across ${summary.n_countries} countries with at least 2,000 held-out drawings each, the same model scores ${(summary.max_min_gap * 100).toFixed(1)} points higher on its best country than its worst.`,
    `Nothing about the model changes between those rows — only whose drawings it is being asked about.`,
    `Retraining on a geographically flatter sample of the same size moves that gap to ${(summary.gap_balanced * 100).toFixed(1)} points.`,
    `Be careful what you conclude: country is not culture, and device, translation, or time spent drawing could each produce a gap like this.`,
  ].join(' ')
}

/** Deterministic answers to the questions students actually ask. */
const FAQ = [
  { k: ['entropy', 'what is entropy'], a: 'Entropy measures how spread out the probabilities are. Zero means all 100% sits on one class. One means it is split evenly across all 50 — total confusion. It is calculated from the predictions themselves, so it costs nothing extra to compute.' },
  { k: ['ensemble', 'three models', 'disagreement'], a: 'Three copies of the same architecture were trained separately with different random starts. They learn slightly different things, so when they disagree about a drawing, that drawing is genuinely ambiguous to this kind of network — a signal one model alone cannot give you.' },
  { k: ['calibration', 'calibrated', 'temperature'], a: 'A neural network can be 90% confident and right only 70% of the time. Temperature scaling divides the outputs by one constant, fitted on held-out data, to bring stated confidence in line with actual accuracy.' },
  { k: ['bias', 'fair', 'country'], a: 'The training data comes overwhelmingly from a handful of countries. A model fits what it sees most, so it can end up understanding some people\'s drawings better than others. Lab 4 measures exactly that, per country, with confidence intervals.' },
  { k: ['how big', 'parameters', 'size'], a: 'About 850,000 parameters — small enough to quantize to under a megabyte and run in your browser, which is why nothing you draw is ever uploaded.' },
  { k: ['wrong', 'why did it get', 'mistake'], a: 'Usually one of three things: the drawing is genuinely ambiguous, it belongs to a class the network was never taught, or your drawing style differs from the training data. Lab 2 tells you which strokes drove the answer; Lab 3 covers the second case.' },
]

export function answerQuestion(text) {
  const q = text.toLowerCase()
  const hit = FAQ.find(f => f.k.some(k => q.includes(k)))
  return hit
    ? hit.a
    : "I'm running without the language model right now, so I can only answer from a fixed set of topics — try asking about entropy, the ensemble, calibration, bias, or why it got something wrong. Everything the labs show you is still live and measured."
}
