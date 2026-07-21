# GreenGate Backend v1 — Response Contract & Prompt Changes

Scope of this version: **prompt refinements only**. No Python changes.
Addresses P1 items #3, #4, #5 from `Backend_Implementation.md`.

All changes live in `esg_langgraph/configs/prompts.json`.

---

## 1. Finalized response contract

### `document_analyzer` → `extracted_context`

```json
{
  "facts": [
    {
      "id": "...",
      "category": "Environmental | Social | Governance",
      "topic": "...",
      "evidence": "exact statement from the document",
      "document": "source filename",
      "confidence": 0.0,
      "keywords": ["..."]
    }
  ]
}
```
*Unchanged from v0 except the ASCII-hyphen rule.*

### `esg_question_answering` → `answers`

```json
[
  {
    "question_id": "E1",
    "status": "Fully Addressed | Partially Addressed | Not Disclosed | Not Applicable",
    "confidence": 0.0,
    "reason": "...",
    "evidence": ["fact ids"]
  }
]
```
*Unchanged from v0 except the ASCII-hyphen rule.*

### `score_calculation` → `scores`

Computed deterministically in Python, not by the LLM.

```json
{
  "categories": { "environmental": 0.0, "social": 0.0, "governance": 0.0 },
  "overall": 0.0,
  "coverage": 0.0
}
```

### `recommendation` → `recommendation`  **(CHANGED)**

```json
{
  "executive_summary": "2-3 sentences. What the ESG profile IS.",
  "strengths": ["..."],
  "gaps": ["..."],
  "actions": ["..."],
  "green_finance_recommendation": {
    "instrument": "short phrase, e.g. Sustainability-linked loan",
    "rationale": "1-2 sentences on why this structure fits",
    "conditions": ["conditions precedent / covenants"],
    "monitoring_kpis": ["measurable indicators over facility life"]
  }
}
```

**Breaking change:** `green_finance_recommendation` was a long prose
string in v0; it is now a structured object.

### `verdict` → `verdict` + `reasoning`

- `verdict` — deterministic band lookup in Python (unchanged).
- `reasoning` — plain text, 2-4 sentences, **why this band specifically**.

---

## 2. Section responsibilities (the anti-repetition contract)

The v0 output restated the same thesis three times. Each field now has one
job and must not trespass on the others:

| Field | Job | Must NOT do |
|---|---|---|
| `executive_summary` | What the profile *is* — where mature, where thin | State the verdict, recommend structures, list actions |
| `strengths` | What is demonstrably done well (Fully Addressed) | Speculate beyond evidence |
| `gaps` | What is missing or partial; distinguish "not disclosed" from "disclosed but weak" | Prescribe fixes (that's `actions`) |
| `actions` | Concrete prioritised steps; each maps to a gap | Generic ESG advice |
| `green_finance_recommendation` | The **deal structure only** | Re-summarise the company profile |
| `reasoning` | Why **this band** vs the one above/below | Repeat the profile or the deal structure |

---

## 3. Changes made

### #3 — Eliminated the 3x repeated thesis
- Added an explicit "each field has a DISTINCT job" rule to the
  `recommendation` prompt, with per-field "do NOT" clauses.
- Rewrote the `verdict` prompt to focus solely on the **band boundary** —
  which strengths lifted it, which gaps held it back — and forbade
  re-summarising the profile or restating the financing structure.

### #4 — Removed inline numeric scores from prose
- Added rule to both `recommendation` and `verdict` prompts: never repeat
  numeric scores, because the interface already displays overall and
  category scores next to the text.
- Instructed qualitative phrasing instead ("strong governance", "lagging
  environmental disclosure") rather than "governance (100.0)".

### #5 — Structured `green_finance_recommendation`
- Changed from a ~200-word prose wall with inline (1)(2)(3) enumeration to
  a four-key object: `instrument`, `rationale`, `conditions`,
  `monitoring_kpis`.
- Gives the frontend bulletable content, matching the treatment
  `strengths` / `gaps` / `actions` already receive.

### Bonus — ASCII hyphen enforcement (P0 #2, solved via prompt)
- Added "Use plain ASCII hyphens (-) only. Never use non-breaking hyphens,
  en dashes, or em dashes." to `document_analyzer`,
  `esg_question_answering`, `recommendation`, and `verdict`.
- Removes the U+2011 pollution without touching `tools.py`.

### Also added
- Question-ID citation rule — every point cites supporting IDs, e.g.
  `(E3)`, `(S1, S2)` — feeding the frontend's evidence-traceability work.
- "Never recalculate or dispute the score" made explicit in
  `recommendation`.

---

## 4. Verification

Ran all four mock companies end-to-end against `openai/gpt-5.1` after the
change:

| Company | Overall | Verdict | GFR shape | Non-breaking hyphens | Exec summary |
|---|---|---|---|---|---|
| ABC Green Infrastructure Ltd. | 82.5 | Eligible for Green Finance | ok | 0 | 3 sentences |
| GreenBuild India Pvt. Ltd. | 65.0 | Conditionally Eligible | ok | 0 | 2 sentences |
| XYZ Urban Developers Ltd. | 50.0 | Further Review Required | ok | 0 | 2 sentences |
| Metro Concrete Works Ltd. | 5.0 | Not Eligible | ok | 0 | 3 sentences |

- All four bands remain distinct.
- `green_finance_recommendation` returned the correct four-key object on
  every company.
- Zero non-breaking hyphens across all generated fields.
- Executive summaries within the 2-3 sentence limit.

Score drift vs. earlier runs (e.g. ABC 77.5 → 82.5, Metro 22.5 → 5.0) is
LLM run-to-run variance in the question-answering stage, not caused by
these prompt changes. Bands are unaffected.

---

## 5. Frontend impact

- **Breaking:** `green_finance_recommendation` is now an object, not a
  string. The Streamlit UI must render `instrument` / `rationale` /
  `conditions[]` / `monitoring_kpis[]` instead of printing one paragraph.
- Prose no longer contains numeric scores, so the score tiles and category
  chips are now the single source of truth for numbers — no more
  contradiction between the "5.0" tile and scores quoted in the text.
- Question IDs appear in brackets throughout, ready to become
  click-through evidence links (Frontend_Improvements.md item #3).

---

## 6. Not done in this version

Still open from `Backend_Implementation.md`:

- **P0 #1** — the broken `intake` field. The `input_request` prompt still
  receives no company data, so the model replies by asking for it. Needs a
  code change in `nodes.py` (build the intake dict in Python, or feed
  company metadata into the prompt). Deliberately out of scope here since
  this version is prompt-only.
