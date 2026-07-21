# GreenGate Frontend — Improvement Backlog

Review of the Streamlit UI (localhost:8501, FastAPI backend on :8043),
SCB-branded "AI-powered green & transition finance eligibility screener".

Overall: clean, on-brand, functional — reads as a real product, not a toy.
Numbered linear flow, good left-rail context, staged progress modal that
makes latency feel intentional. Below is the fix list, ranked.

---

## P0 — Judges will catch these

### 1. Score scale inconsistency
- Result shows **Overall score 5.0**, but category chips read
  `environmental: 0.0, social: 0.0, governance: 16.7`, and the console
  pipeline scores Metro Concrete Works at ~20 on a 0–100 scale.
- "5.0" next to "16.7" next to a 0–100 band table is confusing and
  undercuts credibility.
- **Fix:** pick ONE scale (0–100 recommended — matches the questionnaire
  recommendation bands) and make overall score, category sub-scores, and
  verdict banding all consistent.

### 2. No company / deal identity on result
- The result modal shows the verdict but not *which* company or deal it
  belongs to.
- In a demo running 4 companies back-to-back, results are anonymous.
- **Fix:** header on the result view — e.g.
  "Metro Concrete Works Ltd. — Not Eligible (5 / 100)".

---

## P1 — Core UX / differentiator

### 3. Evidence traceability (the actual differentiator)
- Gaps correctly cite question IDs `(E1, E2, E4, E5)` — good — but there
  is no click-through to the source sentence / document / page.
- v0.md's "win the hackathon" feature was clickable evidence → highlighted
  source, like a citation. Not in the UI yet. Biggest gap vs our own design.
- **Fix:** make each cited fact expandable → show the exact evidence text,
  source document, and (once real PDFs land) page number.

### 4. Verdict band visual distinction
- Four bands exist (Eligible / Conditionally Eligible / Further Review /
  Not Eligible) but the banner reads mostly as red/green binary.
- **Fix:** distinct color per band — green (Eligible), amber
  (Conditionally Eligible), orange (Further Review), red (Not Eligible).

### 5. Category scores lack visual weight
- `environmental: 0.0` shown as flat text chips. These are the analyst's
  first read.
- **Fix:** mini bars or an E/S/G donut/gauge so gaps are visible at a
  glance, not parsed from numbers.

### 6. Results in a modal → no comparison
- Results pop in a modal over the input form; can't compare companies.
- For the closing demo beat (ABC 90 vs Metro 5 side by side), this blocks
  the strongest visual.
- **Fix:** a results / history view so multiple assessments sit next to
  each other.

---

## P2 — Polish

### 7. Domain dropdown
- "Select deal domain → Loans" appears to have only one value, while the
  header promises "loans, bonds and facilities".
- **Fix:** either populate Bonds / Facilities, or remove the dropdown until
  they're supported (empty selector reads unfinished).

### 8. Dense narrative blocks
- Executive Summary and Reasoning are wall-of-text.
- **Fix:** apply the same bullet treatment already used for Strengths /
  Gaps / Recommended Actions; keep Executive Summary to 2–3 sentences.

### 9. Download summary report
- Confirm the "Download summary report" button actually produces a file —
  it's a key demo moment and a likely judge click.

### 10. General
- Confirm score/coverage tiles, category chips, and verdict all update
  correctly across all 4 test companies (not just the Not Eligible case).

---

## Notes

- This Streamlit + FastAPI frontend is a separate codebase from the
  `esg_langgraph/` console pipeline in this repo. Score-scale fix (#1)
  should reconcile the two so UI and pipeline report the same numbers.
- Screenshots reviewed: input form, upload state, progress modal, and the
  Metro Concrete Works "Not Eligible" result (score 5.0, coverage 100%).
