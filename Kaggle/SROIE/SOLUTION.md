# Solution: Data Preparation Pipeline

Implementation: [data_preparation.py](data_preparation.py). Run via
`uv run data_preparation.py` (see `pyproject.toml` / `uv.lock`).

The pipeline runs in three stages over both `train/` and `test/` splits of
`SROIE2019/`, each stage writing a new folder of per-document CSVs so
intermediate results are cached and inspectable.

```
box/*.txt  ──▶  df/*.csv  ──▶  graph/*.csv  ──▶  annotation/*.csv
(raw OCR)     (parsed)       (+ layout graph)   (+ entity labels)
```

## Stage 1 — `parse_box_files`: raw OCR → clean per-doc CSV

Each line in `box/*.txt` is a comma-separated record:
`x1,y1,x2,y2,x3,y3,x4,y4,text...` — 4 corners of the OCR quad, then the
recognized text (which may itself contain commas, hence rejoining
`split_lines[8:]` instead of assuming a fixed column count).

Since OCR quads in this dataset are effectively axis-aligned rectangles,
only two opposite corners are needed: `(x1,y1)` = top-left =
`(x_min, y_min)`, `(x3,y3)` = bottom-right = `(x_max, y_max)`. The other two
corners are dropped. Output: one CSV per document with columns
`x_min, y_min, x_max, y_max, text`.

## Stage 2 — `form_graph_connection`: geometric layout graph

For every OCR box (the "source"), find:
- its **nearest neighbor directly below** (vertical edge)
- its **nearest neighbor directly to the right** (horizontal edge)

This gives every node in the document a link to what's structurally
beneath and beside it — the layout signal that plain text-sequence models
(BiLSTM-CRF on concatenated text) don't have, and which the graph
convolution paper accompanying this dataset (`1903.11279v1.pdf`) uses as
edge features between nodes.

**Vertical pass** — for each ordered pair (src, dest) where dest's
vertical center is below src's:
- Compute the horizontal overlap between the two boxes' x-ranges. Four
  cases are handled depending on how the ranges overlap (dest fully
  contains src's x-range, src fully contains dest's, partial overlap on
  either side) — each produces a shared `x_common` column to draw a
  vertical connector through, plus `height` = vertical center-to-center
  distance.
- If a src/dest pair has **no** horizontal overlap at all, it's not a
  vertical neighbor candidate — falls through to the horizontal pass
  instead (`is_beneath` flag guards this).

**Horizontal pass** — mirror logic on the y-ranges for boxes to the right
of src, producing `length` = horizontal center-to-center distance.

**Nearest-neighbor selection** — among all vertical candidates for a
source box, keep only the one with smallest `height`; same for horizontal
with smallest `length`. (`sorted(..., key=lambda x: x[3])`, take index 0.)

**Single-parent dedup** — a box can be "nearest below" for several sources
simultaneously (e.g. one address line is directly below both the company
name and a logo box). To keep the graph a clean forest rather than a dense
many-to-one tangle, group by destination index and keep only the
edge with the **globally smallest distance**; every other edge pointing
at that same destination gets its distance reset to `-1` (no edge). This
is done separately for vertical and horizontal directions
(`groups_dict_vert` / `groups_dict_hori`).

Output: `graph/*.csv` — original box data plus `below_object`,
`below_dist`, `below_obj_index`, `side_object`, `side_length`,
`side_obj_index`, and plotting coordinates for each connector line.

## Stage 3 — `assign_labels`: fuzzy entity tagging

**`read_entities`** loads the ground-truth JSON (`company`, `date`,
`address`, `total`) into a single-row DataFrame.

**`assign_line_label(line, entities)`** decides which entity (if any) an
OCR line belongs to:
1. Tokenize both the OCR line and each candidate entity value into words
   (stripping commas).
2. For every OCR token, check if it fuzzy-matches (`SequenceMatcher.ratio()
   > 0.8`) any token in the entity's value — this absorbs OCR noise
   (misread characters, spacing) without requiring exact string equality.
3. Decide a match using different thresholds per entity type:
   - **ADDRESS**: accept if **≥50%** of the OCR line's tokens matched —
     addresses are long and often split/merged differently between OCR
     and ground truth, so a strict 100% match is too brittle.
   - **Other entities** (company/date/total): accept if **all** OCR
     tokens matched, OR all entity tokens were matched — covers both
     "OCR line is a subset of the entity" and "entity is a subset of the
     OCR line" cases.
4. First entity column that satisfies its threshold wins; otherwise the
   line is labeled `"O"` (outside/no entity).

**`assign_labels(words, entities)`** applies this per line across the
whole document, then resolves conflicts:
- **Suppression rules** — an `ADDRESS` label is dropped if a `TOTAL` was
  already assigned earlier in the doc (money lines below the address
  region on some layouts can spuriously fuzzy-match address tokens);
  similarly `COMPANY` is dropped if `DATE` or `TOTAL` was already seen.
  These are cheap layout priors: company/address normally appear before
  date/total on a receipt, so a late "match" is more likely noise.
- **Largest-bbox-wins for TOTAL/DATE** — receipts often have multiple
  lines that fuzzy-match "total" (subtotal, tax, grand total) or "date"
  (invoice date vs. due date). Rather than trust the first/last match,
  every candidate line's bounding-box area (`(x_max-x_min) + (y_max-y_min)`)
  is tracked, and only the single largest-area candidate is kept as the
  final label; all other candidates for that entity are reset to `"O"`.
  This is a simple, cheap proxy for "the total is normally the boldest/
  most prominent line" without needing font-size features.

Output: `annotation/*.csv` — the graph CSV plus a final `label` column
(`COMPANY` / `DATE` / `ADDRESS` / `TOTAL` / `O`) per OCR box, ready to
feed a BiLSTM-CRF or graph-conv model as IOB-style training targets.

## Design notes / known trade-offs

- **O(n²) per document** for the graph step (every box compared against
  every other box) — fine at SROIE's scale (~100–300 boxes/doc per the
  accompanying paper), would need spatial indexing (k-d tree / grid) to
  scale further.
- **Fuzzy matching threshold (0.8) and the ADDRESS 50% rule are hand-tuned
  heuristics**, not learned — reasonable starting point given no bbox-level
  ground truth exists, but worth revisiting if label quality is checked
  against a hand-annotated sample.
- **Single nearest-neighbor edges only** (not fully-connected graph
  attention as in the accompanying paper) — a deliberate simplification;
  the paper's model computes attention over *all* pairs, this pipeline
  only pre-computes the strongest single below/right edge per node as a
  cheaper structural feature.
