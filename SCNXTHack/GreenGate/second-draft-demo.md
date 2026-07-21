# GreenGate — Demo Video Skeleton (2:30–2:40 target)

## 0:00–0:15 — Hook / Problem

A construction company applies to a bank for a green infrastructure loan.
Today: analyst manually reads 150+ pages (sustainability report + annual
report), spends days answering the bank's ESG questionnaire. Different
analysts score the same company differently.

## 0:15–0:35 — Solution one-liner

"GreenGate — an AI ESG assessment engine. Upload the company's reports,
it scores them against the bank's own ESG questionnaire, and explains
every score with evidence — in minutes, not days."

## 0:35–1:00 — Architecture flash + positioning disclaimer

Show pipeline diagram once, name each stage fast, don't linger:

```
Input Request → Document Analyzer → ESG Question Answering
→ Score Calculator → Recommendation → Verdict
```

Built on LangGraph. Each node makes one focused LLM call, state flows
through the graph.

**New:** explicit positioning line — this is *inspired by* the
methodology patterns of major ESG rating agencies and frameworks
(S&P CSA, CDP, ISSB, TNFD), not a claim to reproduce any proprietary
scoring model. Pre-empts the "S&P doesn't publish their algorithm"
challenge before a judge can raise it.

## 1:00–1:55 — Live demo (the meat)

Run two contrasting companies back-to-back through the real pipeline.

**ABC Green Infrastructure Ltd.**
- Score: 90 / 100 → **Eligible for Green Finance**
- Point at 1–2 extracted facts → matching question status
  (e.g. "Net Zero by 2045, TCFD/IFRS S2 aligned" → E1/E2 Fully Addressed)

**Metro Concrete Works Ltd.**
- Score: 20 / 100 → **Not Eligible**
- Point at the gaps (no biodiversity assessment, no emissions target,
  vague safety statement, no board independence data)

Show the category breakdown table on screen for one of them
(Environmental / Social / Governance sub-scores).

## 1:55–2:20 — Explainability angle

Show one recommendation output (strengths / gaps / recommended actions).
This is the differentiator over "just a score":

"Every number traces back to a sentence in the actual document —
not a black box."

## 2:20–2:35 — Close

- Business impact: days → minutes, consistent scoring across analysts.
- What's next: real PDF ingestion, page-level evidence citations,
  parallel Environmental/Social/Governance scoring at scale.
- End screen: all 4 companies' scores/verdicts side by side.

| Company | Score | Verdict |
|---|---|---|
| ABC Green Infrastructure Ltd. | 90 | Eligible for Green Finance |
| GreenBuild India Pvt. Ltd. | 65 | Conditionally Eligible |
| XYZ Urban Developers Ltd. | 50 | Further Review Required |
| Metro Concrete Works Ltd. | 20 | Not Eligible |

---

## Narration Script (second draft, word-for-word)

*Read at a natural pace, ~2.5 words/sec. Timings are targets, not hard cuts.*

### 0:00–0:15 — Hook / Problem

> "When a construction company applies to a bank for a green loan, an
> ESG analyst has to read hundreds of pages — sustainability reports,
> annual reports, risk questionnaires — and manually decide if it
> qualifies as green financing. That takes days. And two analysts
> reading the same report can land on two different scores."

### 0:15–0:35 — Solution one-liner

> "This is GreenGate. You upload a company's reports, and it scores
> them against the bank's own ESG questionnaire — automatically,
> consistently, and every score comes with the evidence behind it.
> What used to take days now takes minutes."

### 0:35–1:00 — Architecture flash + positioning disclaimer

> "Under the hood, it's a LangGraph pipeline. Documents go in, an AI
> analyzer extracts factual ESG evidence, a second stage answers the
> bank's questionnaire using only that evidence, a scoring engine
> calculates a weighted score, and a final stage generates a
> recommendation and a verdict. Six stages, one flow, fully traceable.
>
> To be clear — this isn't reverse-engineering S&P, MSCI, or
> Sustainalytics' proprietary scoring models. Those algorithms aren't
> public, and we're not claiming to replicate them. GreenGate is
> inspired by the methodology patterns those agencies and frameworks
> like CDP, ISSB, and TNFD use — evidence-based disclosure assessment,
> weighted materiality — applied through our own transparent,
> bank-specific questionnaire."

### 1:00–1:55 — Live demo

> "Let's run two real companies through it."

*[run ABC Green Infrastructure Ltd.]*

> "ABC Green Infrastructure — a company with a published Net Zero
> target, TCFD-aligned climate strategy, and a majority-independent
> board. GreenGate extracts these facts straight from the documents,
> maps them against the questionnaire, and comes back with a score of
> 90 out of 100 — Eligible for Green Finance."

*[run Metro Concrete Works Ltd.]*

> "Now compare that to Metro Concrete Works. No emissions target, no
> biodiversity assessment, safety mentioned only in passing, no board
> independence data. Score: 20 out of 100 — Not Eligible. Same
> pipeline, same questions, radically different — and honest — outcome."

*[show category breakdown table]*

> "And it's not just one number — you get the breakdown: Environmental,
> Social, Governance, scored separately, so an analyst can see exactly
> where the gaps are."

### 1:55–2:20 — Explainability angle

> "Here's what makes this different from a plain ESG score generator.
> Every recommendation — the strengths, the gaps, the suggested next
> steps — is generated directly from the extracted evidence. Nothing
> is fabricated. If the report doesn't say it, GreenGate doesn't claim
> it. It's a black-box score turned into an audit trail."

### 2:20–2:35 — Close

> "The result: what took an ESG analyst days now takes minutes, and
> every company is scored against the exact same criteria. Here's all
> four companies we tested, side by side — from Eligible for Green
> Finance down to Not Eligible. This is GreenGate."

---

## Open items for next draft

- [x] Word-for-word narration script per block
- [x] Position disclaimer re: not replicating S&P/MSCI/Sustainalytics
- [ ] Decide: screen recording of terminal logs, or a built UI?
- [ ] Confirm which 2 companies to demo live (ABC + Metro give the
      cleanest contrast; GreenBuild/XYZ better for the closing table only)
- [ ] Time the actual read-aloud against a stopwatch and trim to fit
      (now running ~2:35, may need a small cut in the disclaimer line)
