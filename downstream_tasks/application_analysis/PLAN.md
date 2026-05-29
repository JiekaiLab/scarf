# Implementation Plan for Additional Downstream Analyses

## Goal

Add downstream analyses that demonstrate biological and practical utility:

- reference-based annotation
- cross-dataset generalization
- few-shot annotation
- ATAC-to-RNA biological validation

## Code Layout

- `common.py`
  - embedding loading
  - label alignment
  - weighted kNN label transfer
  - standard metrics and output writers
- `reference_mapping.py`
  - generic labeled-reference to query-cell annotation
  - supports scATAC-to-scRNA, scRNA-to-scATAC, or same-modality mapping depending on input files
- `cross_dataset_generalization.py`
  - wrapper around reference mapping using two embedding directories
  - intended for dataset A reference and dataset B query
- `few_shot_annotation.py`
  - samples N labeled cells per type
  - evaluates label transfer under limited annotation
- `rna_imputation_validation.py`
  - compares predicted expression to measured expression
  - reports per-cell, per-gene, and marker-set correlations

## Manuscript-Ready Outputs

Each script writes tabular outputs suitable for figures:

- `metrics.json`
- `classification_report.tsv`
- `confusion_matrix.tsv`
- `predictions.tsv`
- `few_shot_summary.tsv`
- `per_cell_correlation.tsv`
- `per_gene_correlation.tsv`
- `marker_gene_set_validation.tsv`

