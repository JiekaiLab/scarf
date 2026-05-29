import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_names(path: str) -> np.ndarray:
    names = np.load(path, allow_pickle=True)
    return names.astype(str)


def load_embeddings(path: str, l2_normalize: bool = True) -> np.ndarray:
    emb = np.load(path)
    emb = np.asarray(emb, dtype=np.float32)
    if l2_normalize:
        emb = normalize(emb, norm="l2", axis=1)
    return emb


def read_labels(path: str) -> pd.DataFrame:
    labels = pd.read_csv(path, sep=None, engine="python", header=None, names=["cell_name", "label"])
    labels["cell_name"] = labels["cell_name"].astype(str)
    labels["label"] = labels["label"].astype(str)
    return labels.drop_duplicates("cell_name")


def align_labels(names: Sequence[str], label_path: str, drop_unlabeled: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    label_df = read_labels(label_path)
    label_map = dict(zip(label_df["cell_name"], label_df["label"]))
    labels = np.array([label_map.get(str(name), None) for name in names], dtype=object)
    keep = np.array([label is not None for label in labels], dtype=bool)
    if drop_unlabeled:
        return keep, labels[keep].astype(str)
    return keep, labels


def weighted_vote(neighbor_labels: np.ndarray, neighbor_distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    predictions = []
    confidences = []
    similarities = 1.0 - neighbor_distances
    for labels, sims in zip(neighbor_labels, similarities):
        scores = defaultdict(float)
        for label, sim in zip(labels, sims):
            scores[str(label)] += max(float(sim), 0.0)
        if not scores:
            predictions.append("NA")
            confidences.append(0.0)
            continue
        pred, score = max(scores.items(), key=lambda item: item[1])
        total = sum(scores.values())
        predictions.append(pred)
        confidences.append(score / total if total > 0 else 0.0)
    return np.array(predictions), np.array(confidences, dtype=np.float32)


def knn_predict(
    reference_emb: np.ndarray,
    reference_labels: np.ndarray,
    query_emb: np.ndarray,
    k: int = 15,
    metric: str = "cosine",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = min(k, reference_emb.shape[0])
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(reference_emb)
    distances, indices = nn.kneighbors(query_emb)
    neighbor_labels = reference_labels[indices]
    predictions, confidences = weighted_vote(neighbor_labels, distances)
    return predictions, confidences, indices, distances


def write_predictions(
    output_dir: str,
    query_names: Sequence[str],
    predictions: Sequence[str],
    confidences: Sequence[float],
    true_labels: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    ensure_dir(output_dir)
    pred_df = pd.DataFrame(
        {
            "cell_name": np.asarray(query_names).astype(str),
            "prediction": np.asarray(predictions).astype(str),
            "confidence": np.asarray(confidences, dtype=np.float32),
        }
    )
    if true_labels is not None:
        pred_df["truth"] = np.asarray(true_labels).astype(str)
        pred_df["correct"] = pred_df["prediction"] == pred_df["truth"]
    pred_df.to_csv(os.path.join(output_dir, "predictions.tsv"), sep="\t", index=False)
    return pred_df


def write_metrics(output_dir: str, truth: Sequence[str], predictions: Sequence[str]) -> Dict[str, float]:
    ensure_dir(output_dir)
    truth = np.asarray(truth).astype(str)
    predictions = np.asarray(predictions).astype(str)
    metrics = {
        "accuracy": float(accuracy_score(truth, predictions)),
        "macro_f1": float(f1_score(truth, predictions, average="macro")),
        "weighted_f1": float(f1_score(truth, predictions, average="weighted")),
    }
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    report = classification_report(truth, predictions, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(os.path.join(output_dir, "classification_report.tsv"), sep="\t")

    labels = sorted(set(truth) | set(predictions))
    cm = confusion_matrix(truth, predictions, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(os.path.join(output_dir, "confusion_matrix.tsv"), sep="\t")
    return metrics


def embedding_paths(embedding_dir: str, modality: str) -> Tuple[str, str, str]:
    if modality not in {"rna", "atac"}:
        raise ValueError("modality must be 'rna' or 'atac'")
    emb_name = "rna_cell_embs.npy" if modality == "rna" else "atac_cell_embs.npy"
    return (
        os.path.join(embedding_dir, emb_name),
        os.path.join(embedding_dir, "cell_names.npy"),
        os.path.join(embedding_dir, "labels.tsv.gz"),
    )


def add_common_knn_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--k", type=int, default=15)
    parser.add_argument("--metric", default="cosine", choices=["cosine", "euclidean"])
