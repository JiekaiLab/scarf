import argparse
import os

from common import embedding_paths
from reference_mapping import main as reference_mapping_main


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-dataset SCARF label transfer wrapper.")
    parser.add_argument("--reference-dir", required=True)
    parser.add_argument("--query-dir", required=True)
    parser.add_argument(
        "--direction",
        choices=["atac_to_rna", "rna_to_atac", "rna_to_rna", "atac_to_atac"],
        default="atac_to_rna",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
    return parser.parse_args()


def main():
    args = parse_args()
    reference_modality, query_modality = {
        "atac_to_rna": ("rna", "atac"),
        "rna_to_atac": ("atac", "rna"),
        "rna_to_rna": ("rna", "rna"),
        "atac_to_atac": ("atac", "atac"),
    }[args.direction]

    reference_emb, reference_names, reference_labels = embedding_paths(args.reference_dir, reference_modality)
    query_emb, query_names, query_labels = embedding_paths(args.query_dir, query_modality)

    cmd_args = [
        "reference_mapping.py",
        "--reference-emb",
        reference_emb,
        "--reference-names",
        reference_names,
        "--reference-labels",
        reference_labels,
        "--query-emb",
        query_emb,
        "--query-names",
        query_names,
        "--query-labels",
        query_labels,
        "--output-dir",
        args.output_dir,
        "--k",
        str(args.k),
        "--metric",
        args.metric,
    ]

    import sys

    old_argv = sys.argv
    try:
        sys.argv = cmd_args
        reference_mapping_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()

