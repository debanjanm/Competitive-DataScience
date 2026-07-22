# GreenGate — Solution Submission (Second Draft)

**Event:** Accelerate 3.0 Pre-Conference Hackathon
**Deadline:** 22 July 2026, 23:59 IST
**Project:** GreenGate — AI-powered green and transition finance eligibility screener

*Second draft: same content as the first, written in plainer English.*

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

*(limit 50 words — this draft is 46)*

> The backend is Python, built with LangGraph. Six steps run in order: intake,
> document analysis, question answering, scoring, recommendation and verdict.
> Each step calls GPT-5.1 through OpenRouter and returns JSON. The score is
> plain Python maths, not AI. Streamlit frontend, FastAPI service, python-docx
> reads the reports.

---

## Step 3 — Challenges Faced

*(limit 100 words — this draft is 99. The guide asks us to focus on how we solved them.)*

> We hit three problems. First, the AI rewarded any clear number, even a bad one.
> A company with only two independent directors still passed. We rewrote the rules
> for each status and rebuilt our test documents until all four verdict bands could
> be reached. Second, the AI wrote long text that was hard to read and said the same
> thing three times. We made it return JSON and gave each field one job. Third,
> letting the AI do the maths was too risky for a bank. We moved all scoring into
> plain Python, so the AI only reads evidence.

---

## Step 4 — What's Next?

*(limit 100 words — this draft is 92. The guide suggests covering features,
scale, deployment and user testing.)*

> Features: link every score to the exact sentence and page in the source PDF, and
> adapt the questions to each industry, not just construction. Scale: score
> Environmental, Social and Governance at the same time instead of one after
> another, and cache results to cut cost. Production: read PDFs and scans, save
> progress so a failed run can resume, add a backup model, and log tokens and cost
> per run. Testing: let ESG analysts use it, then compare its scores against their
> manual ones on real reports before any lending decision uses it.

---

## Step 5 — Repository URL

What the guide asks for:
- Judges must be able to open the repository.
- It must hold all the source code and documentation.
- It must include a clear README.

Example given: `https://dev.azure.com/sc-ado-academy/Accelerate3.0-Hackathon/...`

**Blocked.** The guide asks for an **Azure DevOps** repository. Our code is in
GitHub, so it needs to be pushed across (or the right URL confirmed) before we
submit. There is also no README yet, and the guide says one is required.

---

## Step 6 — Supporting document / video

- Limit: **4 minutes, 200 MB**. Record with Clipchamp, Teams Recording (Copilot),
  or PowerPoint.
- The organiser confirmed in chat that the format is up to us. Supporting documents
  can be anything besides the video, but a video is recommended.
- Our script in `second-draft-demo.md` runs about 2 minutes 35 seconds. That leaves
  roughly 90 seconds spare if we want to slow down or show more of the live demo.

---

## Step 7 — Final Submit

Click **Final Submit** once steps 2 to 6 are done.

---

## Open items before submitting

- [ ] Push the code to Azure DevOps and check the judges can open it (Step 5).
- [ ] Write the README. It does not exist yet and the guide requires it.
- [ ] Record the demo video using `second-draft-demo.md`.
- [ ] Fix the broken `intake` field (P0 #1 in `Backend_Implementation.md`). Right now
      it replies by asking the user for data it already has. That looks bad if a
      judge reads the raw output.
- [ ] Make the UI score scale match the pipeline (P0 #1 in
      `Frontend_Improvements.md`). The UI showed 5.0 next to category values on a
      different scale.
- [ ] Decide if we upload anything besides the video. The design mockup and the
      architecture diagram are both ready to show.

---

## What is already in the repo

- `second-draft-demo.md` — video script with timings and narration.
- `Frontend_Improvements.md` — UI fixes, ranked by priority.
- `Backend_Implementation.md` and `BACKEND_v1.md` — backend contract and prompt changes.
- `mockups/result_screen_mockup.html` — UI mockup, opens in any browser.
- `esg_langgraph/` — the working pipeline.
- `data/` — four mock companies, eight documents, covering all four verdict bands.
