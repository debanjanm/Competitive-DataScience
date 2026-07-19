Yes. This is where your POC becomes interesting.

I would avoid saying **"LLM answers ESG questions."** Instead, present it as:

> **The AI evaluates the company's disclosures against the bank's ESG assessment framework.**

That sounds much more professional and closer to how banks operate.

---

# Step 3 — ESG Framework Evaluation

Input:

```text
Extracted ESG Evidence
        +
Bank ESG Question Bank
```

↓

Output:

```text
Question-by-Question Assessment
```

---

## Example

The bank has an internal question:

### Q1

> **Does the company disclose a climate strategy?**

AI searches the extracted evidence.

Found:

> "ABC Infrastructure is committed to achieving Net Zero emissions by 2045."

Page 18

Decision

```text
Status

✓ Fully Addressed
```

---

### Q2

> **Has the company conducted a biodiversity impact assessment?**

Evidence found

> "The company is committed to protecting biodiversity."

No assessment.

Decision

```text
Status

⚠ Partially Addressed
```

---

### Q3

> **Has the company quantified physical climate risk?**

Evidence

Nothing found.

Decision

```text
Status

❌ Missing
```

---

# Don't use Yes / No

This is one improvement I'd strongly recommend.

Instead use four states.

| Status                 | Meaning                                         |
| ---------------------- | ----------------------------------------------- |
| ✅ Fully Addressed      | Strong evidence found                           |
| 🟡 Partially Addressed | Some evidence but incomplete                    |
| ⚪ Not Disclosed        | No evidence found                               |
| 🔵 Not Applicable      | Question doesn't apply to this industry/company |

This is much more realistic.

---

# Output

Instead of JSON,

show a nice table.

| Question                 | Status            | Evidence                            |
| ------------------------ | ----------------- | ----------------------------------- |
| Climate Strategy         | ✅ Fully Addressed | Net Zero by 2045 (Page 18)          |
| Biodiversity Assessment  | 🟡 Partial        | Commitment only (Page 34)           |
| Physical Risk Assessment | ⚪ Not Disclosed   | No evidence found                   |
| Worker Safety Policy     | ✅ Fully Addressed | Safety KPIs disclosed (Page 42)     |
| Board Independence       | ✅ Fully Addressed | 6/9 Independent Directors (Page 81) |

This is probably the screen judges will spend the most time looking at.

---

# Why this step is powerful

This step creates **traceability**.

Later, if someone asks

> "Why did you say Biodiversity was Partial?"

The AI can immediately show

```text
Evidence

Page 34

"The company is committed to protecting biodiversity."

Reason

No biodiversity impact/dependency assessment was found.
```

That's explainable AI, which is extremely important in banking.

---

# Internally

The output of this step could look like:

```json
{
  "Q1": {
    "status": "Fully Addressed",
    "confidence": 0.98,
    "page": 18
  },
  "Q2": {
    "status": "Partially Addressed",
    "confidence": 0.91,
    "page": 34
  },
  "Q3": {
    "status": "Not Disclosed",
    "confidence": 0.99,
    "page": null
  }
}
```

The user never sees this JSON—it simply feeds the scoring engine.

---

## I would make one small improvement

Rather than calling it **"ESG Question Evaluation"**, I'd name it:

> **Step 3 – ESG Assessment Against Bank Framework**

That wording makes it clear that:

* the **bank owns the assessment framework**,
* the **AI evaluates the uploaded documents against that framework**,
* and the output is an **auditable assessment with supporting evidence**, not just a chatbot answering questions.

It also sets up the next step naturally:

> **Step 4 – ESG Score Calculation**, where these assessment results are converted into a score using the bank's internal weights and rules.
