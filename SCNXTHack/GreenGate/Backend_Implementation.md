# GreenGate Backend — Implementation & Wording Backlog

Review of the backend response (ABC Green Infrastructure Ltd., model
`openai/gpt-5.1`, overall 77.5 → Conditionally Eligible).

Overall content quality is high: every claim is evidence-tagged
(E1/S3/G2...), scores are band-consistent, recommendations are actionable
and non-fabricated. The GPT-5.1 upgrade clearly improved reasoning depth.
Issues below are wording / structure, not correctness.

---

## P0 — Must fix

### 1. `intake` field is broken
- The `input_request` prompt receives no company data, so GPT-5.1 replies
  by *asking the user to provide the details it already has*:
  > "Please provide the incoming ESG assessment request details so I can
  > validate them. Specifically, share: 1. Company metadata..."
- This is dead, embarrassing output if ever surfaced in the UI.
- **Fix (recommended):** drop the LLM call in `input_request_node` — build
  the intake dict directly in Python from `companies.json` metadata. It's
  pure metadata echo, no reasoning needed, and can't hallucinate.
- **Alt fix:** feed company_name / industry / country / year / document
  list INTO the prompt so it confirms rather than requests.

### 2. Non-breaking hyphen pollution (`‑`)
- GPT-5.1 emits `TCFD‑aligned`, `Net‑Zero`, `board‑level` as U+2011
  non-breaking hyphens throughout. Renders acceptably but bloats JSON,
  breaks copy/search, looks odd in some fonts.
- **Fix:** normalize inside `parse_json_response` in `tools.py`:
  ```python
  cleaned = cleaned.replace("‑", "-").replace("–", "-")
  ```

---

## P1 — Should fix (prompt rewrites in configs/prompts.json)

### 3. Same thesis repeated 3x
- `executive_summary`, `reasoning`, and `green_finance_recommendation` all
  restate "strong gov 100 / social 87.5, weak env 60, needs time-bound
  conditions." Redundant when shown together in the UI.
- **Fix:** give each section a distinct job in its prompt —
  - `executive_summary` → what the ESG profile *is* (2–3 sentences).
  - `reasoning` → why *this specific band*.
  - `green_finance_recommendation` → the *deal structure only*
    (SLL vs use-of-proceeds, covenants, KPIs).

### 4. Scores re-printed inline in prose
- "strong governance (score 100.0) and robust social performance (87.5)" —
  the numbers already appear in the category chips above.
- **Fix:** instruct prompts to reference strength qualitatively
  ("strong governance"), not re-print scores the UI already renders.

### 5. `green_finance_recommendation` is a wall of text
- One ~200-word paragraph with inline (1)(2)(3) enumeration.
- **Fix:** return it as a structured list, matching the bulleted treatment
  `strengths` / `gaps` / `actions` already receive. May require changing
  the field to an array or an object with `approach` + `conditions[]`.

---

## Fine as-is (no change)

- Scoring math: 77.5 overall, categories consistent, band correct.
- `strengths` / `gaps` / `actions`: well-scoped, evidence-tagged,
  no fabrication, correct question-ID references.
- Coverage reporting (100%) working.

---

## Suggested implementation order

1. #1 intake fix (code, `nodes.py`) — highest impact, removes broken output.
2. #2 hyphen normalize (code, `tools.py`) — one-line, affects all fields.
3. #3 / #4 / #5 prompt rewrites (`configs/prompts.json`) — polish pass,
   re-run all 4 companies afterward to confirm bands unchanged.

## Notes

- Model in use: `openai/gpt-5.1` via OpenRouter (set in `llm.py`,
  `OPENROUTER_MODEL` env override supported).
- Re-run the full 4-company sweep after any prompt change to confirm
  ABC / GreenBuild / XYZ / Metro still land in distinct bands
  (90 / 65 / 50 / 20 region).
