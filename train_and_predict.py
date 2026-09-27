"""Train a local business-entity-resolution baseline and create valid submission TSVs."""

from __future__ import annotations

import re
import subprocess
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "dataset" / "train"
TEST_DIR = ROOT / "dataset" / "test"
OUTPUT_DIR = ROOT / "output"
RANDOM_STATE = 42
CANDIDATES_PER_SOURCE = 50


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t", keep_default_na=False)


def normalize(value: object) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = value.encode("ascii", "ignore").decode("ascii").lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy().reset_index(drop=True)
    frame["name_norm"] = frame["business_name"].map(normalize)
    frame["address_norm"] = frame["business_address"].map(normalize)
    frame["country_norm"] = frame["country"].map(normalize)
    frame["combined_text"] = frame["name_norm"] + " " + frame["address_norm"]
    return frame


def build_candidates(left: pd.DataFrame, right_sources: list[pd.DataFrame]) -> pd.DataFrame:
    """Return the exact final candidate list to be scored by the classifier."""
    result: dict[str, dict[str, float]] = {entity_id: {} for entity_id in left.entity_id}

    for right in right_sources:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(2, 5), min_df=1, sublinear_tf=True
        )
        vectorizer.fit(pd.concat([left.combined_text, right.combined_text], ignore_index=True))
        x_left = vectorizer.transform(left.combined_text)
        x_right = vectorizer.transform(right.combined_text)

        for country, left_group in left.groupby("country_norm", dropna=False):
            right_group = right[right.country_norm == country]
            if right_group.empty:
                right_group = right

            neighbor_count = min(CANDIDATES_PER_SOURCE, len(right_group))
            if neighbor_count == 0:
                continue

            matcher = NearestNeighbors(
                n_neighbors=neighbor_count, metric="cosine", algorithm="brute"
            )
            matcher.fit(x_right[right_group.index])
            distances, positions = matcher.kneighbors(x_left[left_group.index])

            for left_index, match_positions, match_distances in zip(
                left_group.index, positions, distances
            ):
                source1_id = left.loc[left_index, "entity_id"]
                for position, distance in zip(match_positions, match_distances):
                    candidate_id = right_group.iloc[position].entity_id
                    similarity = 1.0 - float(distance)
                    result[source1_id][candidate_id] = max(
                        result[source1_id].get(candidate_id, 0.0), similarity
                    )

    return pd.DataFrame(
        [
            {
                "source1_entity_id": source1_id,
                "candidate_entity_id": candidate_id,
                "retrieval_similarity": similarity,
            }
            for source1_id, candidates in result.items()
            for candidate_id, similarity in candidates.items()
        ]
    )


def token_jaccard(left_value: str, right_value: str) -> float:
    left_tokens = set(left_value.split())
    right_tokens = set(right_value.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def first_number(value: str) -> str:
    match = re.search(r"\d+", value)
    return match.group(0) if match else ""


def make_features(
    left: pd.DataFrame, right: pd.DataFrame, candidate_pairs: pd.DataFrame
) -> pd.DataFrame:
    left_records = left.set_index("entity_id")
    right_records = right.set_index("entity_id")
    feature_rows: list[dict[str, float]] = []

    for pair in candidate_pairs.itertuples(index=False):
        source1 = left_records.loc[pair.source1_entity_id]
        candidate = right_records.loc[pair.candidate_entity_id]
        source1_name_tokens = set(source1.name_norm.split())
        candidate_name_tokens = set(candidate.name_norm.split())
        source1_address_tokens = set(source1.address_norm.split())
        candidate_address_tokens = set(candidate.address_norm.split())

        feature_rows.append(
            {
                "retrieval_similarity": pair.retrieval_similarity,
                "name_jaccard": token_jaccard(source1.name_norm, candidate.name_norm),
                "address_jaccard": token_jaccard(
                    source1.address_norm, candidate.address_norm
                ),
                "name_token_overlap": len(source1_name_tokens & candidate_name_tokens),
                "address_token_overlap": len(
                    source1_address_tokens & candidate_address_tokens
                ),
                "same_country": float(source1.country_norm == candidate.country_norm),
                "same_first_number": float(
                    bool(first_number(source1.address_norm))
                    and first_number(source1.address_norm)
                    == first_number(candidate.address_norm)
                ),
                "name_length_ratio": min(len(source1.name_norm), len(candidate.name_norm))
                / max(len(source1.name_norm), len(candidate.name_norm), 1),
                "address_length_ratio": min(
                    len(source1.address_norm), len(candidate.address_norm)
                )
                / max(len(source1.address_norm), len(candidate.address_norm), 1),
            }
        )

    return pd.DataFrame(feature_rows)


def macro_f05(
    source1_ids: np.ndarray,
    candidate_pairs: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    truth: dict[str, set[str]],
) -> float:
    predictions: dict[str, set[str]] = {entity_id: set() for entity_id in source1_ids}
    for pair, probability in zip(candidate_pairs.itertuples(index=False), probabilities):
        if probability >= threshold:
            predictions[pair.source1_entity_id].add(pair.candidate_entity_id)

    scores: list[float] = []
    for source1_id in source1_ids:
        actual = truth.get(source1_id, set())
        predicted = predictions[source1_id]
        if not actual:
            scores.append(1.0 if not predicted else 0.0)
            continue

        overlap = len(actual & predicted)
        precision = overlap / len(predicted) if predicted else 0.0
        recall = overlap / len(actual)
        denominator = 0.25 * precision + recall
        scores.append(1.25 * precision * recall / denominator if denominator else 0.0)

    return float(np.mean(scores))


def write_submission(
    source1: pd.DataFrame,
    pairs: pd.DataFrame,
    column_name: str,
    destination: Path,
) -> None:
    grouped: dict[str, str] = {}
    if not pairs.empty:
        grouped = (
            pairs.groupby("source1_entity_id")["candidate_entity_id"]
            .agg(lambda ids: ",".join(sorted(set(ids))))
            .to_dict()
        )
    output = pd.DataFrame(
        {
            "source1_entity_id": source1.entity_id,
            column_name: [grouped.get(entity_id, "") for entity_id in source1.entity_id],
        }
    )
    output.to_csv(destination, sep="\t", index=False)


def main() -> None:
    tr_s1 = prepare(read_tsv(TRAIN_DIR / "train_source1.tsv"))
    tr_s2 = prepare(read_tsv(TRAIN_DIR / "train_source2.tsv"))
    tr_s3 = prepare(read_tsv(TRAIN_DIR / "train_source3.tsv"))
    te_s1 = prepare(read_tsv(TEST_DIR / "test_source1.tsv"))
    te_s2 = prepare(read_tsv(TEST_DIR / "test_source2.tsv"))
    te_s3 = prepare(read_tsv(TEST_DIR / "test_source3.tsv"))
    truth_frame = read_tsv(TRAIN_DIR / "train_ground_truth.tsv")
    truth = {
        row.source1_entity_id: set(filter(None, row.matched_entity_ids.split(",")))
        for row in truth_frame.itertuples(index=False)
    }

    print(f"Training records: S1={len(tr_s1)}, S2={len(tr_s2)}, S3={len(tr_s3)}")
    print(f"Test records: S1={len(te_s1)}, S2={len(te_s2)}, S3={len(te_s3)}")

    train_candidates = build_candidates(tr_s1, [tr_s2, tr_s3])
    test_candidates = build_candidates(te_s1, [te_s2, te_s3])
    candidate_set = set(
        zip(train_candidates.source1_entity_id, train_candidates.candidate_entity_id)
    )
    all_positive_pairs = {
        (source1_id, candidate_id)
        for source1_id, candidate_ids in truth.items()
        for candidate_id in candidate_ids
    }
    print(
        "Training candidate recall: "
        f"{len(candidate_set & all_positive_pairs) / max(len(all_positive_pairs), 1):.4f}"
    )

    train_right = pd.concat([tr_s2, tr_s3], ignore_index=True)
    test_right = pd.concat([te_s2, te_s3], ignore_index=True)
    train_features = make_features(tr_s1, train_right, train_candidates)
    train_pairs = pd.concat([train_candidates.reset_index(drop=True), train_features], axis=1)
    train_pairs["label"] = [
        int(candidate_id in truth.get(source1_id, set()))
        for source1_id, candidate_id in zip(
            train_pairs.source1_entity_id, train_pairs.candidate_entity_id
        )
    ]

    feature_columns = [
        "retrieval_similarity",
        "name_jaccard",
        "address_jaccard",
        "name_token_overlap",
        "address_token_overlap",
        "same_country",
        "same_first_number",
        "name_length_ratio",
        "address_length_ratio",
    ]
    fit_ids, validation_ids = train_test_split(
        tr_s1.entity_id.to_numpy(), test_size=0.20, random_state=RANDOM_STATE
    )
    fit_pairs = train_pairs[train_pairs.source1_entity_id.isin(fit_ids)]
    validation_pairs = train_pairs[train_pairs.source1_entity_id.isin(validation_ids)]
    positive_weight = (fit_pairs.label == 0).sum() / max((fit_pairs.label == 1).sum(), 1)

    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(
        fit_pairs[feature_columns],
        fit_pairs.label,
        sample_weight=np.where(fit_pairs.label == 1, positive_weight, 1.0),
    )
    validation_probabilities = model.predict_proba(validation_pairs[feature_columns])[:, 1]
    threshold_scores = [
        (
            threshold,
            macro_f05(
                validation_ids, validation_pairs, validation_probabilities, threshold, truth
            ),
        )
        for threshold in np.arange(0.35, 0.96, 0.02)
    ]
    threshold, validation_score = max(threshold_scores, key=lambda item: item[1])
    print(f"Validation macro F0.5: {validation_score:.4f} at threshold {threshold:.2f}")

    final_positive_weight = (train_pairs.label == 0).sum() / max(
        (train_pairs.label == 1).sum(), 1
    )
    final_model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.06,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    final_model.fit(
        train_pairs[feature_columns],
        train_pairs.label,
        sample_weight=np.where(train_pairs.label == 1, final_positive_weight, 1.0),
    )

    test_features = make_features(te_s1, test_right, test_candidates)
    test_pairs = pd.concat([test_candidates.reset_index(drop=True), test_features], axis=1)
    test_pairs["probability"] = final_model.predict_proba(test_pairs[feature_columns])[:, 1]

    OUTPUT_DIR.mkdir(exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"
    write_submission(te_s1, test_pairs, "candidate_entity_ids", candidate_path)
    write_submission(
        te_s1,
        test_pairs[test_pairs.probability >= threshold],
        "matched_entity_ids",
        matching_path,
    )

    validation = subprocess.run(
        [
            "python",
            str(ROOT / "utils" / "validate_submission.py"),
            "--matching",
            str(matching_path),
            "--candidate",
            str(candidate_path),
            "--test-dir",
            str(TEST_DIR),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    print(validation.stdout.strip())
    if validation.stderr.strip():
        print(validation.stderr.strip())
    if validation.returncode:
        raise SystemExit(validation.returncode)


if __name__ == "__main__":
    main()
