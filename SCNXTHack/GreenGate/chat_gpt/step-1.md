Excellent catch. **I actually wouldn't upload the CRQ.**

If this is **SCB's internal process**, then the CRQ belongs to the **bank**, not the customer.

Think of the workflow.

## Real World

```
Customer
    │
    ├── Sustainability Report
    ├── Annual Report
    └── (Optional ESG Report)
            │
            ▼
        Bank System
            │
            ├── Loads Bank's Climate Risk Questionnaire (internal)
            ├── Loads Scoring Rules (internal)
            ├── Loads Industry Materiality (internal)
            ▼
        AI Assessment
```

The customer never sees the bank's internal questionnaire or scoring logic.

---

# For the POC

I would have the user upload **only two documents**.

```
📄 Sustainability Report

📄 Annual Report
```

Everything else comes from the backend.

---

## Backend contains

### 1. ESG Question Bank

Example

```
Construction_ESG_Questionnaire.json
```

```json
{
  "industry": "Construction",
  "questions": [
    {
      "id": "Q1",
      "question": "Does the company disclose climate strategy?",
      "weight": 20
    },
    {
      "id": "Q2",
      "question": "Has biodiversity impact been assessed?",
      "weight": 15
    }
  ]
}
```

---

### 2. Mock Scoring Rules

```
Construction_Scoring.json
```

```text
Full = 100

Partial = 50

No = 0
```

---

### 3. Industry Materiality

```
Construction_Materiality.json
```

```text
Climate            20%

Physical Risk      20%

Biodiversity       15%

Safety             15%

Governance         15%

Water              10%

Community          5%
```

---

### 4. Recommendation Library

```
recommendations.json
```

Example

```
IF Biodiversity = Missing

↓

Recommend

Conduct TNFD LEAP Assessment
```

---

# The AI flow becomes much cleaner

```
Relationship Manager

        │

Upload Company Reports
        │
        ▼

AI Document Extraction
        │
        ▼

Backend loads

✓ Construction ESG Questions
✓ Scoring Rules
✓ Materiality Weights
✓ Recommendation Rules

        │
        ▼

AI answers every question
        │
        ▼

Calculate Score
        │
        ▼

Generate Recommendation
```

---

# This is also easier to explain to judges

You can say:

> "The uploaded documents come from the customer. The ESG questionnaire, scoring methodology, and materiality weights are maintained internally by the bank and automatically selected based on the customer's industry."

That sounds exactly like an enterprise banking solution.

---

## I would even simplify the architecture further.

Instead of thinking about a **Climate Risk Questionnaire**, think about a **Bank ESG Framework**.

```
Customer Uploads
----------------
✓ Sustainability Report
✓ Annual Report

↓

Bank ESG Framework
------------------
✓ ESG Question Bank
✓ Industry Materiality
✓ Scoring Rules
✓ Recommendation Rules

↓

AI ESG Assessment
```

This design is **more realistic**, **cleaner**, and **more scalable**. If tomorrow the bank wants to assess a manufacturing company instead of a construction company, you simply switch the backend framework to the manufacturing question bank and weights—without asking the customer to upload a different questionnaire. I would recommend finalizing this architecture for your POC.
