Yes. I actually think we should **merge the recommendation and decision into one final step**. That makes the demo much cleaner.

So the complete flow becomes:

```text
1. Upload Company Documents
        ↓
2. AI Document Intelligence
        ↓
3. ESG Assessment Against Bank Framework
        ↓
4. ESG Score Calculation
        ↓
5. ESG Score Review (optional screen)
        ↓
6. AI Decision & Recommendations
```

Or, if you want only **5 screens**, skip Step 5 entirely:

```text
1. Upload Documents
        ↓
2. AI Document Intelligence
        ↓
3. ESG Framework Assessment
        ↓
4. ESG Score Calculation
        ↓
5. AI Recommendation & Decision
```

I actually prefer the **5-step flow**.

---

# Step 5 — AI Recommendation & Decision

This is the screen the Relationship Manager or Credit Analyst cares about.

It should answer **four questions**:

## 1. What is the score?

```text
Overall ESG Score

69 / 100
```

---

## 2. Why?

Instead of showing only a number:

```text
Strengths

✓ Climate strategy publicly disclosed
✓ Net Zero target established
✓ Strong worker safety practices
✓ Independent board oversight

Gaps

⚠ Biodiversity assessment incomplete
⚠ Physical climate risk exposure not disclosed
```

This makes the score explainable.

---

## 3. What should the bank do?

For the POC, keep it simple.

Example

```text
Recommendation

Conditionally Eligible for Green Finance
```

Not

```text
Approve
Reject
```

Banks rarely automate approval.

The AI assists the analyst.

Possible values:

| Result                  | Meaning                         |
| ----------------------- | ------------------------------- |
| Eligible                | Strong ESG evidence             |
| Conditionally Eligible  | Minor gaps need follow-up       |
| Further Review Required | Significant missing information |
| Not Eligible            | Major ESG deficiencies          |

This is much more realistic.

---

## 4. How can the company improve?

Generated automatically.

Example

```text
Recommended Actions

1.
Complete a TNFD-aligned biodiversity assessment.

2.
Quantify physical climate risks for project locations.

3.
Publish science-based emissions reduction targets.
```

Notice these recommendations come directly from the unanswered or partially answered questions.

---

# Final Screen

```text
--------------------------------------------------

Green Finance ESG Assessment

--------------------------------------------------

Company

ABC Infrastructure Ltd.

Overall ESG Score

69 / 100

Recommendation

🟡 Conditionally Eligible

--------------------------------------------------

Strengths

✓ Climate strategy disclosed

✓ Net Zero target

✓ Strong governance

✓ Worker safety programme

--------------------------------------------------

Key Gaps

⚠ Biodiversity assessment incomplete

⚠ Physical climate risk exposure unavailable

--------------------------------------------------

Recommended Actions

• Conduct TNFD LEAP assessment

• Complete physical climate risk analysis

• Publish location-specific climate risk disclosure

--------------------------------------------------

Supporting Evidence

Climate Strategy
Page 18

Net Zero Target
Page 19

Board Governance
Page 81

--------------------------------------------------
```

---

# One thing I'd add that most teams won't

Instead of ending with:

> **Conditionally Eligible**

add one more line:

```text
AI Recommendation

Conditionally Eligible

Human Review Required
```

This is **very important in banking**.

It tells the judges:

* AI is **assisting** the analyst.
* The final lending decision remains with a human.
* The system supports explainable, auditable decision-making.

That aligns with responsible AI practices and is likely to resonate well in a banking hackathon.

---

## Finalized End-to-End POC

```text
1. Upload Company Documents
        │
        ▼
2. AI Document Intelligence
        │
        ▼
3. ESG Assessment Against Bank Framework
        │
        ▼
4. ESG Score Calculation
        │
        ▼
5. AI Recommendation & Decision
             │
             ├── ESG Score
             ├── Strengths
             ├── Gaps
             ├── Recommended Actions
             └── Human Review Required
```

I think this is the right level of complexity for a hackathon: it tells a complete business story without getting bogged down in implementation details.
