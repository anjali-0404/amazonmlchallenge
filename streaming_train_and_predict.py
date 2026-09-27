"""Disk-backed local training and inference for the Amazon ML Challenge dataset."""

from __future__ import annotations

import csv
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "dataset" / "train"
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"
DATABASE = ROOT / "tmp" / "entity_blocks.sqlite"
SAMPLE_SIZE = 60000
QUERY_LIMIT = 30
PREDICTION_BATCH_SIZE = 3000
STOP_TOKENS = {"ltd", "limited", "llc", "inc", "corp", "corporation", "company", "co", "pvt", "private", "llp"}


def normalize(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).casefold().replace("&", " and ")
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def name_key(name: str) -> str:
    tokens = [token for token in normalize(name).split() if token not in STOP_TOKENS]
    if not tokens:
        return ""
    return max(tokens, key=len)[:16]


def address_number(address: str) -> str:
    match = re.search(r"\d+", normalize(address))
    return match.group(0) if match else ""


def rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as file:
        yield from csv.DictReader(file, delimiter="\t")


def build_index(force_rebuild: bool = False) -> sqlite3.Connection:
    DATABASE.parent.mkdir(exist_ok=True)
    if DATABASE.exists() and not force_rebuild:
        print("Using existing SQLite index database:", DATABASE, flush=True)
        connection = sqlite3.connect(DATABASE)
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        connection.execute("PRAGMA temp_store = MEMORY")
        return connection
    if DATABASE.exists():
        DATABASE.unlink()
    connection = sqlite3.connect(DATABASE)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute(
        "CREATE TABLE right_entities ("
        "entity_id TEXT PRIMARY KEY, dataset TEXT, country TEXT, name_key TEXT, address_number TEXT, "
        "name_norm TEXT, address_norm TEXT)"
    )
    connection.execute(
        "CREATE TABLE ground_truth (source1_entity_id TEXT PRIMARY KEY, matched_entity_ids TEXT)"
    )

    for filename in ("train_source2.tsv", "train_source3.tsv", "test_source2.tsv", "test_source3.tsv"):
        dataset = "train" if filename.startswith("train") else "test"
        path = TRAIN / filename if dataset == "train" else TEST / filename
        batch = []
        count = 0
        for row in rows(path):
            name_norm = normalize(row["business_name"])
            address_norm = normalize(row["business_address"])
            batch.append(
                (
                    row["entity_id"],
                    dataset,
                    normalize(row["country"]),
                    name_key(row["business_name"]),
                    address_number(row["business_address"]),
                    name_norm,
                    address_norm,
                )
            )
            if len(batch) == 50000:
                connection.executemany("INSERT INTO right_entities VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                connection.commit()
                count += len(batch)
                batch.clear()
        if batch:
            connection.executemany("INSERT INTO right_entities VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
            connection.commit()
            count += len(batch)
        print(f"Indexed {count:,} rows from {filename}", flush=True)

    batch = []
    for row in rows(TRAIN / "train_ground_truth.tsv"):
        batch.append((row["source1_entity_id"], row["matched_entity_ids"]))
        if len(batch) == 50000:
            connection.executemany("INSERT INTO ground_truth VALUES (?, ?)", batch)
            connection.commit()
            batch.clear()
    if batch:
        connection.executemany("INSERT INTO ground_truth VALUES (?, ?)", batch)
        connection.commit()

    connection.execute("CREATE INDEX index_right_name ON right_entities(dataset, country, name_key)")
    connection.execute("CREATE INDEX index_right_address ON right_entities(dataset, country, address_number)")
    connection.commit()
    return connection


def candidates(connection: sqlite3.Connection, row: dict[str, str], dataset: str):
    country = normalize(row["country"])
    n_key = name_key(row["business_name"])
    a_number = address_number(row["business_address"])
    found: dict[str, tuple[str, str, str]] = {}
    queries = []
    if len(n_key) >= 3:
        queries.append(("name_key", n_key))
    if a_number:
        queries.append(("address_number", a_number))

    for field, value in queries:
        sql = (
            f"SELECT entity_id, country, name_norm, address_norm FROM right_entities "
            f"WHERE dataset = ? AND country = ? AND {field} = ? LIMIT ?"
        )
        for entity_id, candidate_country, candidate_name, candidate_address in connection.execute(
            sql, (dataset, country, value, QUERY_LIMIT)
        ):
            found[entity_id] = (candidate_country, candidate_name, candidate_address)
    return found


def jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(left.split()), set(right.split())
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def feature_row(source: dict[str, str], candidate: tuple[str, str, str]) -> list[float]:
    candidate_country, candidate_name, candidate_address = candidate
    source_name = normalize(source["business_name"])
    source_address = normalize(source["business_address"])
    source_number = address_number(source["business_address"])
    candidate_number = address_number(candidate_address)
    return [
        jaccard(source_name, candidate_name),
        jaccard(source_address, candidate_address),
        float(name_key(source["business_name"]) == name_key(candidate_name)),
        float(bool(source_number) and source_number == candidate_number),
        float(normalize(source["country"]) == candidate_country),
        min(len(source_name), len(candidate_name)) / max(len(source_name), len(candidate_name), 1),
        min(len(source_address), len(candidate_address)) / max(len(source_address), len(candidate_address), 1),
    ]


def fit_model(connection: sqlite3.Connection):
    features, labels, source_ids = [], [], []
    positive_pairs = 0
    candidate_pairs = 0

    for index, source in enumerate(rows(TRAIN / "train_source1.tsv")):
        if index >= SAMPLE_SIZE:
            break
        truth_row = connection.execute(
            "SELECT matched_entity_ids FROM ground_truth WHERE source1_entity_id = ?",
            (source["entity_id"],),
        ).fetchone()
        actual = set(filter(None, truth_row[0].split(","))) if truth_row else set()
        for candidate_id, candidate in candidates(connection, source, "train").items():
            features.append(feature_row(source, candidate))
            labels.append(int(candidate_id in actual))
            source_ids.append(source["entity_id"])
            candidate_pairs += 1
            positive_pairs += int(candidate_id in actual)

    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int8)
    source_ids = np.asarray(source_ids)
    if positive_pairs == 0:
        raise RuntimeError("No positive pairs were generated. Broaden the blocking keys before training.")

    unique_sources = np.unique(source_ids)
    fit_sources, validation_sources = train_test_split(
        unique_sources, test_size=0.20, random_state=42
    )
    fit_mask = np.isin(source_ids, fit_sources)
    validation_mask = ~fit_mask
    positive_weight = (y[fit_mask] == 0).sum() / max((y[fit_mask] == 1).sum(), 1)
    model = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=42
    )
    model.fit(x[fit_mask], y[fit_mask], sample_weight=np.where(y[fit_mask] == 1, positive_weight, 1.0))
    validation_probabilities = model.predict_proba(x[validation_mask])[:, 1]

    print(
        f"Training sample: {len(unique_sources):,} Source 1 rows, {candidate_pairs:,} candidate pairs, "
        f"{positive_pairs:,} positive pairs",
        flush=True,
    )
    return model, x, y, source_ids, validation_mask, validation_probabilities


def choose_threshold(
    source_ids: np.ndarray, labels: np.ndarray, validation_mask: np.ndarray, probabilities: np.ndarray
) -> float:
    val_labels = labels[validation_mask]
    val_pairs = source_ids[validation_mask]

    grouped: dict[str, tuple[list[int], list[float]]] = defaultdict(lambda: ([], []))
    for s_id, lab, prob in zip(val_pairs, val_labels, probabilities):
        grouped[s_id][0].append(int(lab))
        grouped[s_id][1].append(float(prob))

    sources_data = []
    for s_id, (y_list, p_list) in grouped.items():
        y_arr = np.asarray(y_list, dtype=bool)
        p_arr = np.asarray(p_list, dtype=np.float32)
        sources_data.append((y_arr, p_arr, int(y_arr.sum())))

    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.35, 0.96, 0.02):
        scores = []
        for y_arr, p_arr, n_actual in sources_data:
            predicted = p_arr >= threshold
            n_pred = int(predicted.sum())
            if n_actual == 0:
                scores.append(1.0 if n_pred == 0 else 0.0)
                continue
            if n_pred == 0:
                scores.append(0.0)
                continue
            overlap = int(np.logical_and(y_arr, predicted).sum())
            precision = overlap / n_pred
            recall = overlap / n_actual
            denominator = 0.25 * precision + recall
            scores.append(1.25 * precision * recall / denominator if denominator else 0.0)
        score = float(np.mean(scores))
        if score > best_score:
            best_threshold, best_score = float(threshold), score
    print(f"Validation macro F0.5: {best_score:.4f} at threshold {best_threshold:.2f}", flush=True)
    return best_threshold


def write_predictions(connection: sqlite3.Connection, model, threshold: float) -> None:
    OUTPUT.mkdir(exist_ok=True)
    matching_path = OUTPUT / "matching_results.tsv"
    candidate_path = OUTPUT / "candidate_pairs.tsv"
    with matching_path.open("w", encoding="utf-8", newline="") as matching_file, candidate_path.open(
        "w", encoding="utf-8", newline=""
    ) as candidate_file:
        matching_writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        candidate_writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        matching_writer.writerow(["source1_entity_id", "matched_entity_ids"])
        candidate_writer.writerow(["source1_entity_id", "candidate_entity_ids"])

        batch = []
        processed = 0
        for source in rows(TEST / "test_source1.tsv"):
            batch.append(source)
            if len(batch) >= PREDICTION_BATCH_SIZE:
                _write_batch(batch, connection, model, threshold, matching_writer, candidate_writer)
                processed += len(batch)
                print(f"Scored {processed:,} test Source 1 rows", flush=True)
                batch.clear()
        if batch:
            _write_batch(batch, connection, model, threshold, matching_writer, candidate_writer)
            processed += len(batch)
            print(f"Scored {processed:,} test Source 1 rows", flush=True)


def _write_batch(batch, connection, model, threshold, matching_writer, candidate_writer) -> None:
    feature_matrix, locations, candidate_ids = [], [], []
    candidate_lists: dict[str, list[str]] = {}
    for source in batch:
        found = candidates(connection, source, "test")
        ids = sorted(found)
        candidate_lists[source["entity_id"]] = ids
        for candidate_id in ids:
            feature_matrix.append(feature_row(source, found[candidate_id]))
            locations.append(source["entity_id"])
            candidate_ids.append(candidate_id)

    matches = defaultdict(list)
    if feature_matrix:
        probabilities = model.predict_proba(np.asarray(feature_matrix, dtype=np.float32))[:, 1]
        for source_id, candidate_id, probability in zip(locations, candidate_ids, probabilities):
            if probability >= threshold:
                matches[source_id].append(candidate_id)

    for source in batch:
        source_id = source["entity_id"]
        matching_writer.writerow([source_id, ",".join(sorted(matches[source_id]))])
        candidate_writer.writerow([source_id, ",".join(candidate_lists[source_id])])


def main() -> None:
    import subprocess
    connection = build_index()
    try:
        model, _, labels, source_ids, validation_mask, validation_probabilities = fit_model(connection)
        threshold = choose_threshold(source_ids, labels, validation_mask, validation_probabilities)
        write_predictions(connection, model, threshold)
    finally:
        connection.close()

    print("Validating generated submission files...", flush=True)
    validation = subprocess.run(
        [
            "python",
            str(ROOT / "utils" / "validate_submission.py"),
            "--matching",
            str(OUTPUT / "matching_results.tsv"),
            "--candidate",
            str(OUTPUT / "candidate_pairs.tsv"),
            "--test-dir",
            str(TEST),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    print(validation.stdout.strip())
    if validation.stderr.strip():
        print(validation.stderr.strip())


if __name__ == "__main__":
    main()
