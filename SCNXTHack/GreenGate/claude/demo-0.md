Now write a generator script producing 4 companies' realistic report pairs, engineered to hit each ESG band.

Extraction works cleanly, richer/longer content now. Update `configs/companies.json` to register the two new companies.

Now run all 4 companies through the real pipeline to confirm the four bands actually land where designed.

ABC and XYZ landed weaker/stronger than intended — need to see per-question answers to retune wording precisely.

GreenBuild (70→Conditionally Eligible) and Metro (20→Not Eligible) landed correctly. ABC and XYZ need rewording — the classifier reads "disclosure completeness" not "adequacy," so a concrete number always reads as Full even if weak (e.g. 2/8 independent directors still = Full). Retuning both.

Now XYZ — make E2 vague (no firm target) and G2 vague (no specific numbers).

Regenerate all (idempotent overwrite) and re-run all 4 to verify bands.

All four distinct bands hit cleanly:

| Company | Overall | Verdict |
|---|---|---|
| ABC Green Infrastructure Ltd. | 90.0 | **Eligible for Green Finance** |
| GreenBuild India Pvt. Ltd. | 65.0 | **Conditionally Eligible** |
| XYZ Urban Developers Ltd. | 50.0 | **Further Review Required** |
| Metro Concrete Works Ltd. | 20.0 | **Not Eligible** |

Full spread across all 4 recommendation bands from the questionnaire spec, clean separation, no boundary overlaps.

**What changed in [data/](SCNXTHack/GreenGate/data/):**
- Regenerated all 8 docx files as proper multi-section Word documents (via `python-docx`) — headings, CEO messages, named frameworks (TCFD/IFRS S2), quantified KPIs (LTIFR, %, INR figures), more realistic than the old single-paragraph mocks.
- Added 2 new companies: **GreenBuild India Pvt. Ltd.** (mid performer) and **Metro Concrete Works Ltd.** (poor performer), each with sustainability + annual report pair — bringing the total to 4 companies × 2 docs = 8 files.
- Retuned ABC (now clearly strong, one deliberate gap in community disclosure) and XYZ (now clearly weak-to-mixed) after seeing the classifier's actual behavior: it grades disclosure completeness, not adequacy — a concrete number (even a bad one, like 2/8 independent directors) reads as "Fully Addressed," while vague/aspirational language reads as "Partially" or "Not Disclosed."
- `configs/companies.json` — registered all 4 companies with their doc mappings.

To demo a different company, just change `company_name` in [main.py](SCNXTHack/GreenGate/esg_langgraph/main.py) to any of the 4 names above.