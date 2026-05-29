import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from common import (
    align_labels,
    embedding_paths,
    ensure_dir,
    knn_predict,
    load_embeddings,
    load_names,
    write_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Few-shot SCARF label-transfer benchmark.")
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--modality", choices=["rna", "atac"], required=True)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 5, 10, 50])
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def sample_few_shot_indices(labels: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    selected = []
    for label in sorted(set(labels)):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            continue
        take = min(shots, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())
    return np.array(sorted(selected), dtype=np.int32)


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    emb_path, names_path, labels_path = embedding_paths(args.embedding_dir, args.modality)
    embeddings = load_embeddings(emb_path)
    names = load_names(names_path)
    keep, labels = align_labels(names, labels_path)
    embeddings = embeddings[keep]
    labels = labels.astype(str)

    all_rows = []
    for shots in args.shots:
        for repeat in range(args.repeats):
            rng = np.random.default_rng(args.seed + repeat + shots * 1000)
            reference_idx = sample_few_shot_indices(labels, shots, rng)
            query_mask = np.ones(labels.shape[0], dtype=bool)
            query_mask[reference_idx] = False
            query_idx = np.flatnonzero(query_mask)

            predictions, confidences, _, _ = knn_predict(
                reference_emb=embeddings[reference_idx],
                reference_labels=labels[reference_idx],
                query_emb=embeddings[query_idx],
                k=args.k,
            )
            run_dir = os.path.join(args.output_dir, f"shots_{shots}_repeat_{repeat}")
            metrics = write_metrics(run_dir, labels[query_idx], predictions)
            metrics.update({"shots": shots, "repeat": repeat, "n_reference": int(len(reference_idx))})
            all_rows.append(metrics)

    pd.DataFrame(all_rows).to_csv(os.path.join(args.output_dir, "few_shot_summary.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()

