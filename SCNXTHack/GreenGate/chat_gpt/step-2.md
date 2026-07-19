Yes. This is the next logical step. However, I'd avoid calling it **"Document Parsing"** in the demo because it sounds too technical.

Think from a bank user's perspective. They don't care that you're parsing PDFs—they care that the AI **understands the documents**.

## I would name this step:

> **Document Intelligence** *(my favorite)*

or

> **ESG Information Extraction**

or

> **AI Document Analysis**

---

# Flow So Far

```text
1. Upload Company Documents
        │
        ▼
2. AI Document Intelligence
        │
        ▼
3. ESG Question Evaluation
        │
        ▼
4. ESG Score Calculation
        │
        ▼
5. Recommendations & Decision
```

This tells a much better story.

---

# What happens in "Document Intelligence"?

The AI doesn't immediately calculate scores.

It first **creates a structured knowledge base** from the uploaded documents.

## Input

```text
ABC_Sustainability_Report.pdf

ABC_Annual_Report.pdf
```

↓

## AI extracts ESG evidence

Example:

| ESG Topic          | Extracted Evidence                         | Page | Confidence |
| ------------------ | ------------------------------------------ | ---- | ---------- |
| Climate Strategy   | Net Zero by 2045                           | 18   | 98%        |
| Carbon Emissions   | Scope 1 & 2 disclosed                      | 21   | 99%        |
| Biodiversity       | Commitment mentioned                       | 34   | 93%        |
| Flood Adaptation   | New projects follow flood-resilient design | 27   | 95%        |
| Board Independence | 6 of 9 directors independent               | 82   | 99%        |

Notice something important:

**This is not yet answering questions.**

It is simply saying:

> "Here is everything I found."

---

# Internally

The AI creates something like:

```json
{
  "facts": [
    {
      "topic": "Climate Strategy",
      "value": "Net Zero by 2045",
      "page": 18
    },
    {
      "topic": "Flood Adaptation",
      "value": "Critical assets relocated",
      "page": 27
    },
    {
      "topic": "Biodiversity",
      "value": "Commitment only",
      "page": 34
    }
  ]
}
```

The user never sees this JSON.

---

# Why separate this step?

Because one extracted fact can answer multiple ESG questions.

Example:

From one sentence:

> "The company has adopted flood-resilient construction standards."

The AI can infer:

* ✅ Physical adaptation exists
* ✅ Climate resilience planning exists
* ✅ Infrastructure resilience evidence exists

Extract once, reuse many times.

This is much more efficient than asking the LLM the same question repeatedly.

---

# UI for Demo

After clicking **Analyze**, the user could briefly see:

```
Analyzing Sustainability Report...
✓ Climate Strategy
✓ Carbon Emissions
✓ Biodiversity
✓ Water Management
✓ Worker Safety

Analyzing Annual Report...
✓ Board Structure
✓ Risk Management
✓ Governance
```

Then show:

```
Information Extracted

✓ 27 ESG facts extracted
✓ 18 Climate-related facts
✓ 6 Governance facts
✓ 3 Social facts

Proceeding to ESG Assessment...
```

This gives judges confidence that the AI is doing something meaningful before scoring.

---

# Finalized Step 2

## **Step 2 — AI Document Intelligence**

**Input**

* Sustainability Report
* Annual Report

**Output**

* Structured ESG evidence (facts with source pages and confidence)

This output becomes the input for the next step:

**Step 3 — ESG Question Evaluation**, where the bank's internal ESG framework checks each question against the extracted evidence. That separation makes the architecture clean, modular, and very similar to how enterprise document intelligence systems are designed.
