import argparse
import os

import numpy as np

from common import (
    add_common_knn_args,
    align_labels,
    ensure_dir,
    knn_predict,
    load_embeddings,
    load_names,
    write_metrics,
    write_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Reference-based label transfer, e.g. scATAC query to labeled scRNA reference."
    )
    parser.add_argument("--reference-emb", required=True)
    parser.add_argument("--reference-names", required=True)
    parser.add_argument("--reference-labels", required=True)
    parser.add_argument("--query-emb", required=True)
    parser.add_argument("--query-names", required=True)
    parser.add_argument("--query-labels", default=None)
    parser.add_argument("--output-dir", required=True)
    add_common_knn_args(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)

    reference_emb = load_embeddings(args.reference_emb)
    reference_names = load_names(args.reference_names)
    reference_keep, reference_labels = align_labels(reference_names, args.reference_labels)
    reference_emb = reference_emb[reference_keep]
    reference_names = reference_names[reference_keep]

    query_emb = load_embeddings(args.query_emb)
    query_names = load_names(args.query_names)

    predictions, confidences, neighbor_indices, neighbor_distances = knn_predict(
        reference_emb=reference_emb,
        reference_labels=reference_labels,
        query_emb=query_emb,
        k=args.k,
        metric=args.metric,
    )

    true_labels = None
    if args.query_labels:
        query_keep, labels = align_labels(query_names, args.query_labels)
        true_labels = np.full(query_names.shape, "NA", dtype=object)
        true_labels[query_keep] = labels
        metric_mask = true_labels != "NA"
        if metric_mask.any():
            write_metrics(args.output_dir, true_labels[metric_mask], predictions[metric_mask])

    write_predictions(args.output_dir, query_names, predictions, confidences, true_labels=true_labels)
    np.save(os.path.join(args.output_dir, "neighbor_indices.npy"), neighbor_indices.astype(np.int32))
    np.save(os.path.join(args.output_dir, "neighbor_distances.npy"), neighbor_distances.astype(np.float32))


if __name__ == "__main__":
    main()

