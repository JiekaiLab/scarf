import argparse
import os
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def as_dense_row(matrix):
    if sparse.issparse(matrix):
        return np.asarray(matrix.toarray()).ravel()
    return np.asarray(matrix).ravel()


def pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size == 0 or y.size == 0:
        return np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def gene_id_index(adata) -> pd.Index:
    if "gene_id" in adata.var.columns:
        return pd.Index(adata.var["gene_id"].astype(str))
    return pd.Index(adata.var_names.astype(str))


def align_anndata(truth, prediction):
    common_cells = truth.obs_names.intersection(prediction.obs_names)
    truth_gene_ids = gene_id_index(truth)
    pred_gene_ids = gene_id_index(prediction)
    common_genes = truth_gene_ids.intersection(pred_gene_ids)
    truth_var = truth.var_names[truth_gene_ids.isin(common_genes)]
    pred_var = prediction.var_names[pred_gene_ids.isin(common_genes)]
    truth_aligned = truth[common_cells, truth_var].copy()
    prediction_aligned = prediction[common_cells, pred_var].copy()
    prediction_aligned = prediction_aligned[:, prediction_aligned.var_names].copy()
    prediction_aligned.var_names = gene_id_index(prediction_aligned).astype(str)
    truth_aligned.var_names = gene_id_index(truth_aligned).astype(str)
    prediction_aligned = prediction_aligned[:, truth_aligned.var_names].copy()
    return truth_aligned, prediction_aligned


def per_cell_correlations(truth, prediction) -> pd.DataFrame:
    rows = []
    for cell in truth.obs_names:
        rows.append(
            {
                "cell_name": cell,
                "pearson": pearson_safe(as_dense_row(truth[cell].X), as_dense_row(prediction[cell].X)),
            }
        )
    return pd.DataFrame(rows)


def per_gene_correlations(truth, prediction) -> pd.DataFrame:
    rows = []
    truth_X = truth.X.tocsc() if sparse.issparse(truth.X) else np.asarray(truth.X)
    pred_X = prediction.X.tocsc() if sparse.issparse(prediction.X) else np.asarray(prediction.X)
    for j, gene in enumerate(truth.var_names):
        tx = as_dense_row(truth_X[:, j].T)
        px = as_dense_row(pred_X[:, j].T)
        rows.append({"gene_id": gene, "pearson": pearson_safe(tx, px)})
    return pd.DataFrame(rows)


def marker_gene_validation(truth, prediction, marker_path: str) -> pd.DataFrame:
    markers = pd.read_csv(marker_path, sep=None, engine="python")
    if "gene_id" not in markers.columns:
        raise ValueError("marker file must contain a gene_id column")
    group_col = "cell_type" if "cell_type" in markers.columns else None
    rows = []
    marker_groups = markers.groupby(group_col) if group_col else [(None, markers)]
    for group, group_df in marker_groups:
        genes = [gene for gene in group_df["gene_id"].astype(str) if gene in truth.var_names]
        if len(genes) == 0:
            continue
        truth_score = np.asarray(truth[:, genes].X.mean(axis=1)).ravel()
        pred_score = np.asarray(prediction[:, genes].X.mean(axis=1)).ravel()
        rows.append(
            {
                "marker_set": group if group is not None else "all_markers",
                "n_genes": len(genes),
                "pearson": pearson_safe(truth_score, pred_score),
                "truth_mean": float(np.mean(truth_score)),
                "prediction_mean": float(np.mean(pred_score)),
            }
        )
    return pd.DataFrame(rows)


def parse_args():
    parser = argparse.ArgumentParser(description="Biological validation for ATAC-to-RNA imputation.")
    parser.add_argument("--truth-h5ad", required=True)
    parser.add_argument("--prediction-h5ad", required=True)
    parser.add_argument("--marker-genes", default=None)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    truth = sc.read_h5ad(args.truth_h5ad)
    prediction = sc.read_h5ad(args.prediction_h5ad)
    truth, prediction = align_anndata(truth, prediction)

    cell_df = per_cell_correlations(truth, prediction)
    gene_df = per_gene_correlations(truth, prediction)
    cell_df.to_csv(os.path.join(args.output_dir, "per_cell_correlation.tsv"), sep="\t", index=False)
    gene_df.to_csv(os.path.join(args.output_dir, "per_gene_correlation.tsv"), sep="\t", index=False)

    summary = {
        "n_cells": int(truth.n_obs),
        "n_genes": int(truth.n_vars),
        "mean_per_cell_pearson": float(np.nanmean(cell_df["pearson"])),
        "median_per_cell_pearson": float(np.nanmedian(cell_df["pearson"])),
        "mean_per_gene_pearson": float(np.nanmean(gene_df["pearson"])),
        "median_per_gene_pearson": float(np.nanmedian(gene_df["pearson"])),
    }
    pd.Series(summary).to_csv(os.path.join(args.output_dir, "summary.tsv"), sep="\t", header=False)

    if args.marker_genes:
        marker_df = marker_gene_validation(truth, prediction, args.marker_genes)
        marker_df.to_csv(os.path.join(args.output_dir, "marker_gene_set_validation.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
