# GreenGate — Solution Submission (First Draft)

**Event:** Accelerate 3.0 Pre-Conference Hackathon
**Deadline:** 22 July 2026, 23:59 IST
**Project:** GreenGate — AI-powered green & transition finance eligibility screener

---

## Submission checklist

| Step | Field | Limit | Status |
|---|---|---|---|
| 1 | Access submission site (Solution-Upload link) | — | pending |
| 2 | Share how it is built / Tech stack | 50 words | drafted below |
| 3 | Challenges Faced | 100 words | drafted below |
| 4 | What's Next? | 100 words | drafted below |
| 5 | Repository URL | Azure DevOps | **blocked — see open items** |
| 6 | Upload supporting document / video | 4 min, 200 MB | script ready, not recorded |
| 7 | Final Submit | — | pending |

---

## Step 2 — Share how it is built / Tech stack

*(max 50 words — draft is 40)*

> Python backend orchestrated with LangGraph: a six-node pipeline covering intake,
> document analysis, questionnaire assessment, scoring, recommendation and verdict.
> LLM calls via OpenRouter (GPT-5.1) using structured JSON contracts between stages.
> Scoring is deterministic Python, not model-generated. Streamlit frontend, FastAPI
> service layer, python-docx for report ingestion.

---

## Step 3 — Challenges Faced

*(max 100 words — draft is 93; guide asks to focus on how the team addressed them)*

> Three main obstacles. First, the model graded disclosure completeness rather than
> adequacy — a weak but specific figure still scored "Fully Addressed". We tightened
> the status definitions and rebuilt the test corpus so every band was reachable.
> Second, free-text LLM output was unreliable to parse and repeated the same
> conclusion across sections; we introduced strict JSON contracts and gave every
> output field a single, non-overlapping job. Third, trusting an LLM with the score
> itself was indefensible for a bank, so we moved all scoring maths into
> deterministic Python, leaving the model to classify evidence only.

---

## Step 4 — What's Next?

*(max 100 words — draft is 89; guide suggests features / scalability / deployment / user testing)*

> Features: page-level evidence citations linking every score to the exact sentence
> in the source PDF, and an industry materiality engine so questionnaires adapt
> beyond construction. Scalability: parallel Environmental, Social and Governance
> branches in the graph, plus response caching to cut repeat assessment cost.
> Production: PDF and OCR ingestion, LangGraph checkpointing for resumable runs,
> model fallback, and per-run token and cost audit logging. Validation:
> analyst-in-the-loop user testing with the ESG and Green Finance teams,
> benchmarking AI assessments against manual scores on real filings to measure
> agreement before any lending use.

---

## Step 5 — Repository URL

Requirements from the guide:
- Repository must be accessible to judges.
- Must contain all source code and documentation.
- Must include a clear README.

Example format given: `https://dev.azure.com/sc-ado-academy/Accelerate3.0-Hackathon/...`

**Open item:** the guide asks for an **Azure DevOps** repository. Current work lives
in a GitHub repo, so the code needs to be pushed to Azure DevOps (or the correct
URL confirmed) before submitting. A README also needs writing — there isn't one yet.

---

## Step 6 — Supporting document / video

- Limit: **4 minutes, 200 MB**. Recording via Clipchamp, Teams Recording (Copilot),
  or PowerPoint.
- Organiser clarified in chat: format is up to the team, supporting documents can be
  anything besides the video, but **a video is recommended**.
- Our demo script (`second-draft-demo.md`) runs ~2:35 — comfortably inside the limit,
  with roughly 90 seconds of headroom if we want to slow the pace or extend the live
  demo section.

---

## Step 7 — Final Submit

Click **Final Submit** once steps 2-6 are complete.

---

## Open items before submitting

- [ ] Push code to an Azure DevOps repository and confirm judge access (Step 5).
- [ ] Write the README — currently missing, and explicitly required.
- [ ] Record the demo video from `second-draft-demo.md`.
- [ ] Fix the broken `intake` field (P0 #1 in `Backend_Implementation.md`) — it
      currently returns a template asking the user for data it already has, which
      would look bad if a judge inspects raw output.
- [ ] Reconcile the UI score scale with the pipeline (P0 #1 in
      `Frontend_Improvements.md`) — the UI showed 5.0 alongside category values on a
      different scale.
- [ ] Decide whether to include supporting documents beyond the video (the design
      mockup and architecture diagram are both presentable).

---

## Reference material already in the repo

- `second-draft-demo.md` — video script with timings and word-for-word narration.
- `Frontend_Improvements.md` — UI backlog, ranked.
- `Backend_Implementation.md` / `BACKEND_v1.md` — backend contract and prompt changes.
- `mockups/result_screen_mockup.html` — standalone UI mockup, opens in any browser.
- `esg_langgraph/` — the working pipeline.
- `data/` — four mock companies, eight documents, spanning all four verdict bands.
