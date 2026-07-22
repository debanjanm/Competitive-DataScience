# GreenGate — Demo Video Script (Third Draft)

**Hard limit: 4 minutes, 200 MB** (per the submission guide).
**This script runs about 3 minutes 40 seconds**, leaving a small buffer.

Written in simple English. Short sentences. Read at a calm pace, roughly
two and a half words per second. Pause at each full stop.

Total narration: ~528 words.

---

## Running order

| Time | Block | What is on screen |
|---|---|---|
| 0:00–0:16 | The problem | Title card, or a thick stack of report pages |
| 0:16–0:32 | What GreenGate is | Product name / logo |
| 0:32–0:52 | How it works, and what we do not claim | Pipeline diagram |
| 0:52–1:11 | **The input** | Upload form |
| 1:11–1:38 | **The scores** | Result header |
| 1:38–1:50 | **Executive summary** | Summary paragraph |
| 1:50–2:16 | **Assessment tab** | Strengths / gaps / actions |
| 2:16–2:41 | **Financing view tab** | Rationale, structure, covenants, KPIs |
| 2:41–3:09 | **Evidence tab** | Question table with quoted source text |
| 3:09–3:24 | Who reads what | Highlight the two audiences |
| 3:24–3:40 | Closing | All four companies side by side |

---

## Full narration

### 0:00–0:16 — The problem

*On screen: title card, or a stack of documents.*

> A company asks the bank for a green loan. To answer, an ESG analyst has to
> read hundreds of pages. Annual reports. Sustainability reports. It takes days.
> And two analysts reading the same report often give two different scores.

---

### 0:16–0:32 — What GreenGate is

*On screen: product name.*

> This is GreenGate. It reads those documents, answers the bank's own ESG
> questionnaire, gives a score, and recommends how to structure the loan.
> Every answer is backed by a line from the document. What took days now
> takes minutes.

---

### 0:32–0:52 — How it works, and what we do not claim

*On screen: the six-step pipeline diagram.*

> Behind it is a six-step pipeline built with LangGraph. One important point:
> we are not copying S&P or MSCI. Their models are private. We follow the same
> ideas that public frameworks use - TCFD, ISSB, TNFD - inside the bank's own
> questionnaire. And the score itself is plain maths, not AI.

*Note: say this early. It stops a judge from asking it later as a challenge.*

---

### 0:52–1:11 — The input

*On screen: the upload form. Point at each field as you name it.*

> Let's screen a company. The input is simple. First, the company name -
> ABC Green Infrastructure. Second, the deal documents. We upload two: the
> annual report and the sustainability report. Third, an optional comment,
> where the relationship manager can add context about the deal. Then submit.

---

### 1:11–1:38 — The scores

*On screen: result header. Point at each number in turn.*

> Here is the result. At the top, the verdict: Conditionally Eligible. Then the
> overall score, out of one hundred. Next to it, the disclosure rate - how much
> of the questionnaire the company actually answered. That matters. A low score
> can mean poor performance, or it can mean they simply did not disclose.
> These are different problems. Below that, the three pillar scores:
> Environmental, Social and Governance.

*This is the strongest single idea in the demo. Do not rush the two sentences
about disclosure rate.*

---

### 1:38–1:50 — Executive summary

*On screen: the summary paragraph under the scores.*

> Under the scores is the executive summary. Two or three lines. It says what
> kind of company this is - where it is strong, where it is thin. Nothing more.

---

### 1:50–2:16 — Assessment tab

*On screen: click the Assessment tab. Scroll slowly through the three sections.*

> Then three tabs. The first is Assessment. Strengths - what the company clearly
> does well. Disclosure gaps - what is missing, or only half answered. Notice we
> call them disclosure gaps, not failures, because most of the time the company
> simply did not report it. And recommended actions - the exact steps that would
> close those gaps. Each point shows the question it came from.

---

### 2:16–2:41 — Financing view tab

*On screen: click Financing view. Pause on each of the four sections.*

> The second tab is the Financing view. This is the banker's tab. Eligibility
> rationale explains why this company landed in this band and not the one above.
> Financing recommendation gives the deal structure - here, a sustainability-linked
> loan. Conditions and covenants list what the bank should demand before lending.
> And monitoring KPIs list what to track for the life of the loan.

---

### 2:41–3:09 — Evidence tab

*On screen: click Evidence. Scroll the table. Hover or zoom on one quoted line.*

> The third tab is Evidence. This is the most important one. Every question in
> the bank's questionnaire, its status, its weight, and the exact sentence from
> the report that supports it - with the document it came from. Nothing is
> invented. If the report does not say it, the system marks it Not Disclosed.
> A bank cannot lend on a number it cannot explain. This tab is that explanation.

*Land the last two lines firmly. This is the differentiator.*

---

### 3:09–3:24 — Who reads what

*On screen: highlight the tabs as you name them.*

> Who uses what? The ESG analyst works in Assessment and Evidence - checking the
> gaps and the proof. The credit committee reads the top of the page and the
> Financing view - the verdict and the deal terms.

---

### 3:24–3:40 — Closing

*On screen: the comparison strip with all four companies.*

> Here are four companies, screened the same way, landing in all four bands.
> Same questions, same rules, every time. The analyst stays in control -
> GreenGate does the reading, and shows its work. Thank you.

---

## Audience map (for reference while recording)

| Screen area | Who reads it | Why |
|---|---|---|
| Verdict, overall score, disclosure rate, pillar scores | Credit committee | The decision, at a glance |
| Executive summary | Both | Two lines on what the company is |
| Assessment tab | ESG analyst | The gaps to chase and the actions to request |
| Financing view tab | Credit committee | Deal structure, covenants, what to monitor |
| Evidence tab | ESG analyst, and audit | Proof behind every status |

---

## Recording notes

- Record the screen with real output, not stills. The staged progress messages
  during the run make the wait look intentional.
- Use ABC Green Infrastructure as the walkthrough company. It has content in
  every section, so no part of the screen looks empty.
- Keep Metro Concrete Works for the closing strip only. Its near-empty result
  is a good contrast, but a poor tour.
- If you run short of time, the block to cut is 0:32–0:52. Say only the one line
  about not copying S&P, and drop the framework names.
- Do not read numbers aloud that are already on screen. Point at them instead.

---

## Open items

- [ ] Record and time a real read-through with a stopwatch.
- [ ] Confirm the UI score scale is fixed before recording, so the overall score
      and the pillar scores are on the same scale on camera.
- [ ] Decide whether to show the live app or the mockup at
      `mockups/result_screen_mockup.html`.
