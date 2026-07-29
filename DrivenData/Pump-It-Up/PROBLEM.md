# Pump it Up: Data Mining the Water Table

DrivenData competition (intermediate practice) — [link](https://www.drivendata.org/competitions/7/pump-it-up-data-mining-the-water-table/)

## Problem

Predict operating condition of water pumps (waterpoints) across Tanzania, using data aggregated by Taarifa from the Tanzania Ministry of Water. Goal: help target maintenance so clean water access improves.

## Task type

Multiclass classification, 3 classes:

- `functional` — operational, no repairs needed
- `functional needs repair` — operational but needs repair
- `non functional` — not operational

## Features (~40 cols)

Grouped by theme:

- **Water/technical**: amount_tsh, extraction_type(_group/_class), water_quality, quality_group, quantity, quantity_group, source, source_type, source_class, waterpoint_type(_group)
- **Location**: gps_height, longitude, latitude, basin, subvillage, region, region_code, district_code, lga, ward
- **Management/ownership**: funder, installer, scheme_management, scheme_name, management, management_group, payment, payment_type, permit, public_meeting
- **Metadata**: date_recorded, wpt_name, num_private, recorded_by, population, construction_year

Note: several near-duplicate cols (extraction_type / extraction_type_group / extraction_type_class; quality / quality_group; quantity / quantity_group; source / source_type / source_class; waterpoint_type / waterpoint_type_group) — likely coarse-to-fine encodings of same info, check correlation/redundancy before using all.

## Target

`status_group` — one of the 3 classes above, keyed by `id`.

## Submission format

```
id,status_group
50785,functional
51630,functional
...
```

Row per `id` in test set values, predicted label as string (not encoded).

## Evaluation metric

Classification rate (accuracy) — % of rows predicted correctly.

## Data files

- Training set values (features)
- Training set labels (status_group per id)
- Test set values (features, no labels — predict these)
- SubmissionFormat.csv (template)

## Approach notes

- Baseline: majority class (`functional`) — check class balance first.
- Handle high-cardinality categoricals (funder, installer, scheme_name, wpt_name, subvillage, ward) — target encoding or drop if too sparse.
- construction_year, population often have 0/missing placeholder values — needs cleaning.
- date_recorded → derive age of pump at recording (date_recorded - construction_year) as feature.
- Geo cols (region, district, lga, ward, lat/long) — potential spatial signal, consider clustering or aggregated stats.
- redundant grouped cols — pick one granularity per theme, or let tree model handle multicollinearity.
