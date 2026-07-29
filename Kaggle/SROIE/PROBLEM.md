# Problem: Key Information Extraction from Scanned Receipts (SROIE2019)

## Task

SROIE2019 (Scanned Receipts OCR and Information Extraction) provides scanned
receipt images and their raw OCR output, and asks for four structured fields
per receipt:

- **company** — merchant/vendor name
- **date** — transaction date
- **address** — merchant address
- **total** — total amount paid

Input per document:
- `img/*.jpg` — the receipt photo
- `box/*.txt` — OCR output: one line per detected text box, format
  `x1,y1,x2,y2,x3,y3,x4,y4,text` (4-corner quad + recognized string)
- `entities/*.txt` — ground-truth JSON: `{"company": ..., "date": ...,
  "address": ..., "total": ...}`

Output needed: for every OCR text box, a label — which of the four entities
(if any) that line of text belongs to — so a downstream sequence model
(BiLSTM-CRF, LayoutLM, etc.) can be trained.

## Why this is hard

1. **OCR gives flat, unordered lines, not structure.** A receipt is a 2-D
   layout — totals sit in a column, addresses wrap across multiple lines,
   company names sit at the top — but OCR just returns boxes with no
   relationship between them. Concatenating text top-to-bottom loses the
   visual cues (relative position, alignment, whitespace) that make it
   possible to tell "$9.00" is the *total* and not just a subtotal or tax
   line sitting two lines above it. This is exactly the challenge motivating
   the [Graph Convolution for Multimodal IE from VRDs](../1903.11279v1.pdf)
   paper (Alibaba, 2019) sitting alongside this dataset — text alone is
   ambiguous; visual/positional context resolves it.

2. **No ground-truth bounding boxes for entities, only strings.** The
   `entities/*.txt` file gives you the *value* of `total` (e.g. `"9.00"`),
   not *which OCR box* it came from. Ground truth was hand-typed separately
   from OCR, so:
   - OCR text and entity text don't match exactly (OCR noise, spacing,
     punctuation differences).
   - A single entity string can span multiple OCR boxes (e.g. a
     multi-line address), or an OCR box can contain more than the entity
     (e.g. `"TOTAL: 9.00"` when the entity is just `"9.00"`).
   - You must reconstruct the box↔entity mapping yourself via fuzzy text
     matching before any model can be trained.

3. **Several entities can look identical.** A receipt often has multiple
   money-shaped strings (subtotal, tax, discount, total) and sometimes
   multiple dates. Text-only heuristics can't disambiguate; you need
   positional signals (e.g. "the total is usually the largest/last money
   box", "the address is usually the block right under the company name").

4. **No relational structure between OCR boxes.** To eventually feed a
   layout-aware model (graph conv, GAT, LayoutLM), each text box needs to
   know its geometric neighbors — what's directly below it, what's directly
   beside it — since that's the signal that lets a model learn "this box is
   below the COMPANY box, so it's probably an ADDRESS line."

## What needs to be produced before modeling can start

1. Clean, structured per-document CSVs of OCR boxes (drop redundant corner
   coordinates OCR gives you; keep just `x_min, y_min, x_max, y_max, text`).
2. A **graph of geometric relationships** between OCR boxes per document —
   nearest neighbor below, nearest neighbor to the right — so a downstream
   model has access to layout, not just raw text sequence.
3. **Entity labels per OCR line**, derived by fuzzily matching OCR text
   against the ground-truth entity JSON, with disambiguation rules for
   which line "wins" when multiple candidates match (e.g. picking the
   largest bounding box among TOTAL/DATE candidates).

This data-preparation stage (see [SOLUTION.md](SOLUTION.md) and
[data_preparation.py](data_preparation.py)) is the piece that turns raw
SROIE2019 OCR + entity files into a labeled, graph-annotated dataset ready
for a sequence/graph model to train on.
