# ScribbleScope — instructor guide

A one-period lab kit for teaching model behaviour with a real network instead of slides.
Free, no accounts, no installs, no data leaves the room: everything runs in the browser.

**Why it exists.** Most AI-literacy material explains uncertainty, calibration, explainability and
bias as definitions. ScribbleScope lets a class *measure* all four on a network they can poke at,
in about forty minutes. The model is a 3-member CNN ensemble trained from scratch on ~600,000
Quick, Draw! doodles across 50 classes.

## Before class

Open the link. That's the whole setup. It works offline after first load, on a Chromebook, and on
phones. Nothing a student draws is uploaded — inference runs on their own device — so it is safe
for classrooms with student-privacy restrictions.

## Mapping to a standard AI curriculum

| Lesson topic | Lab | What students do | The idea that lands |
|---|---|---|---|
| What a neural network is | **1 — What does a network see?** | Draw; watch 50 probabilities update live | A classifier outputs a *distribution*, not an answer |
| How models represent inputs | **1** | Compare a clean drawing to a messy one | Small input changes move probabilities continuously |
| Model confidence | **1** | Read the Doubt Meter while drawing badly | Confidence is computed, not felt |
| Explainability | **2 — Erase a stroke** | Remove each stroke; see the cost | Attribution can be *measured* by intervention |
| Evaluating models | **2** | Find a stroke worth <1 point | Not all input matters; models use shortcuts |
| Limits of AI | **3 — When AI doesn't know** | Draw a giraffe (never taught) | Softmax always sums to 100%: it cannot say "unseen" |
| Uncertainty | **3** | Compare taught vs untaught entropy | Confidence ≠ correctness; ensembles disagree |
| Data and bias | **4 — Does AI understand everyone?** | Sort 44 countries by accuracy | Training-set skew becomes measurable harm |
| AI ethics | **4** | Toggle the rebalanced model | Mitigation is possible, partial, and costs something |
| Research methods | **4** | Read the confidence intervals | Overlapping intervals mean "we can't tell" |

## Suggested 40-minute plan

1. **(5 min) Draw anything.** Lab 1. Ask: *what does 98% mean here?* Collect answers before
   explaining. Most students say "it's right" — that's the misconception the rest of the lesson
   dismantles.
2. **(10 min) Break it.** Lab 2. Have students draw a house, then erase the roof. The prediction
   flips to "chair". Ask: *did the network understand houses, or roof-shapes?*
3. **(10 min) Ask the impossible.** Lab 3. Everyone draws something from the "never taught" list.
   Collect how many got a *confident* wrong answer. This is the lesson's turning point.
4. **(10 min) Whose drawings?** Lab 4. Sort worst-first. Ask *why might this model do worse on
   these countries?* before revealing the training mix.
5. **(5 min) Close.** *Where else does a system get graded on data that doesn't represent everyone?*

## Discussion questions

- If the network is 92% confident and wrong, whose fault is that — the model, the data, or whoever
  deployed it?
- Lab 4 shows a gap between countries. What would you need to know before calling it "cultural
  bias"? (Device, translation, time spent drawing — country is not culture.)
- Rebalancing the training data changes the gap. Who decides what "balanced" means?
- The model runs on your device and uploads nothing. What does that change about who can use it?

## Honest notes for teachers

- The 50 classes are a closed world. Anything outside them will be answered wrongly, by design —
  that's Lab 3's point, not a defect.
- Per-country differences have confounds; the app says so, and students should be held to the same
  standard of care in what they conclude.
- Ada, the built-in tutor, works without any API key by reading the model's numbers directly. If a
  language model is configured it adds depth, but no lesson depends on it.

Everything here — the training code, the evaluation, and the numbers behind Lab 4 — is open at
the project repository, including the parts that didn't work.
