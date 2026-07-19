This is where we should be careful.

For a **hackathon**, you do **not** want to invent a complex ESG formula. You want something that is:

* ✅ Easy to explain in 30 seconds
* ✅ Transparent
* ✅ Auditable
* ✅ Looks realistic

---

# My recommendation

Don't score directly from the documents.

Score from the **assessment results**.

```
Documents
      ↓
Evidence
      ↓
Question Assessment
      ↓
Weighted Score
```

---

# Step 1 — Every Question has a Weight

Example (Construction Industry)

| Question                 | Weight |
| ------------------------ | -----: |
| Climate Strategy         |     20 |
| Net Zero Target          |     15 |
| Biodiversity Assessment  |     15 |
| Physical Risk Assessment |     20 |
| Worker Safety            |     10 |
| Community Engagement     |      5 |
| Board Independence       |     10 |
| ESG Governance           |      5 |

Total = **100**

---

# Step 2 — Convert Status to Points

This is the simplest part.

| Assessment          |   Points |
| ------------------- | -------: |
| Fully Addressed     |      100 |
| Partially Addressed |       50 |
| Not Disclosed       |        0 |
| Not Applicable      | Excluded |

Notice **Not Applicable** is excluded rather than treated as zero. That avoids unfairly penalizing companies for irrelevant questions.

---

# Step 3 — Weighted Score

For each question:

```
Question Score = Weight × (Points / 100)
```

Example:

| Question           | Weight | Status          | Points | Contribution |
| ------------------ | -----: | --------------- | -----: | -----------: |
| Climate Strategy   |     20 | Fully Addressed |    100 |           20 |
| Net Zero Target    |     15 | Fully Addressed |    100 |           15 |
| Biodiversity       |     15 | Partial         |     50 |          7.5 |
| Physical Risk      |     20 | Not Disclosed   |      0 |            0 |
| Worker Safety      |     10 | Fully Addressed |    100 |           10 |
| Community          |      5 | Partial         |     50 |          2.5 |
| Board Independence |     10 | Fully Addressed |    100 |           10 |
| ESG Governance     |      5 | Fully Addressed |    100 |            5 |

Final Score

```
20
+15
+7.5
+0
+10
+2.5
+10
+5
-------
69 / 100
```

Round to **69** or **70**.

---

# Even Better: Add Confidence

This is something many hackathon teams won't think about.

Instead of

```
ESG Score

69
```

show

```
Overall ESG Score

69 / 100

Confidence

92%
```

Confidence comes from:

* number of documents analyzed
* extraction confidence
* amount of missing evidence

---

# Add Coverage

Suppose one document is missing.

Then:

```
Overall ESG Score

69

Coverage

82%

Confidence

91%
```

This immediately tells the analyst:

> "The score is based on 82% of the expected information."

---

# Final Output

Instead of a single number:

```
Overall ESG Score

69 / 100

Environmental
63

Social
75

Governance
90

Coverage
88%

Confidence
94%

Recommendation

Conditionally Eligible for Green Finance
```

---

# One thing I'd change from our earlier discussion

Initially, we considered assigning **0** to "Not Disclosed."

After thinking about it from a banking perspective, I'd make a small refinement.

| Status              |   Points |
| ------------------- | -------: |
| Fully Addressed     |      100 |
| Partially Addressed |       50 |
| Not Disclosed       |       0* |
| Not Applicable      | Excluded |

The `0*` means:

* If the bank **expects the company to disclose** that information (for example, physical risk for a construction company), then not disclosing it should score **0** because it represents a material gap.
* If the question truly doesn't apply to that company or industry, it should be marked **Not Applicable** and excluded from the denominator.

That distinction is simple, defensible, and aligns well with how risk assessments are typically handled.

---

## This is the formula I'd finalize for the POC

```
Overall ESG Score
=
Σ (Question Weight × Status Points)
÷
Σ (Applicable Question Weights)
```

where:

* **Status Points** = 100, 50, or 0
* **Applicable Question Weights** exclude any "Not Applicable" questions.

It's transparent, explainable to judges in under a minute, and easy to implement while leaving room for a more sophisticated model in the future.
