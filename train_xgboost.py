import argparse
import csv
import gzip
import json
import random
from pathlib import Path

import numpy as np
import xgboost as xgb

from honeypot import is_honeypot
from rank import build_memo, xgb_features_from_memo
from scorer import jd_fit_score, recruitability_score


def open_text(path):
    return gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, "rt", encoding="utf-8")


def iter_candidates(path):
    path = Path(path)
    with open_text(path) as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
            return

        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_labels(path):
    labels = {}
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if "candidate_id" not in reader.fieldnames or "relevance" not in reader.fieldnames:
            raise ValueError("labels CSV must contain candidate_id,relevance columns")
        for row in reader:
            cid = row["candidate_id"].strip()
            if not cid:
                continue
            labels[cid] = float(row["relevance"])
    return labels


def pseudo_label(candidate):
    score = jd_fit_score(candidate) * recruitability_score(candidate)
    if score >= 6.0:
        return 4.0
    if score >= 4.0:
        return 3.0
    if score >= 2.0:
        return 2.0
    if score >= 0.75:
        return 1.0
    return 0.0


def collect_training_rows(candidates_path, labels, sample_size, seed, use_pseudo_labels):
    rng = random.Random(seed)
    rows = []
    seen = 0

    for candidate in iter_candidates(candidates_path):
        if is_honeypot(candidate):
            continue

        cid = candidate.get("candidate_id")
        if labels is not None:
            if cid not in labels:
                continue
            relevance = labels[cid]
        elif use_pseudo_labels:
            relevance = pseudo_label(candidate)
        else:
            raise ValueError("provide --labels or pass --use-pseudo-labels for a demo model")

        jd_fit = jd_fit_score(candidate)
        rec = recruitability_score(candidate)
        memo = build_memo(candidate, jd_fit, rec, jd_fit * rec)
        row = (xgb_features_from_memo(memo), relevance, cid)
        seen += 1

        if not sample_size or len(rows) < sample_size:
            rows.append(row)
        else:
            replace_at = rng.randrange(seen)
            if replace_at < sample_size:
                rows[replace_at] = row

    return rows


def collect_bootstrap_rows(seed):
    rng = random.Random(seed)
    rows = []
    for _ in range(1500):
        jd_fit = rng.uniform(0.0, 14.0)
        recruitability = rng.uniform(0.05, 1.0)
        yoe = rng.uniform(1.0, 16.0)
        is_research_only = 1.0 if rng.random() < 0.12 else 0.0
        is_job_hopper = 1.0 if rng.random() < 0.18 else 0.0
        is_big_tech_lifer = 1.0 if rng.random() < 0.08 else 0.0
        has_production_keywords = 1.0 if rng.random() < 0.45 else 0.0
        notice_days = rng.choice([0, 15, 30, 45, 60, 90, 120, 180])
        response_rate = rng.uniform(0.0, 1.0)
        matched_skills = rng.randint(0, 3)
        preferred_location = 1.0 if rng.random() < 0.25 else 0.0

        relevance = (
            0.22 * jd_fit
            + 1.20 * recruitability
            + 0.45 * has_production_keywords
            + 0.25 * matched_skills
            + 0.35 * response_rate
            + 0.20 * preferred_location
        )
        if 5.0 <= yoe <= 9.0:
            relevance += 0.55
        elif 4.0 <= yoe <= 12.0:
            relevance += 0.25

        relevance -= 0.80 * is_research_only
        relevance -= 0.90 * is_job_hopper
        relevance -= 0.45 * is_big_tech_lifer
        if notice_days > 60:
            relevance -= 0.35
        if notice_days > 120:
            relevance -= 0.25

        label = max(0.0, min(4.0, round(relevance)))
        rows.append((
            [
                jd_fit,
                recruitability,
                yoe,
                is_research_only,
                is_job_hopper,
                is_big_tech_lifer,
                has_production_keywords,
                float(notice_days),
                response_rate,
                float(matched_skills),
                preferred_location,
            ],
            label,
            f"BOOT_{len(rows):04d}",
        ))
    return rows


def train_model(candidates_path, labels_path=None, out_path="xgboost_model.json", sample_size=2000, seed=42, use_pseudo_labels=False):
    if candidates_path:
        labels = load_labels(labels_path) if labels_path else None
        rows = collect_training_rows(candidates_path, labels, sample_size, seed, use_pseudo_labels)
    else:
        rows = collect_bootstrap_rows(seed)
    if len(rows) < 20:
        raise ValueError(f"need at least 20 labelled rows after filtering; found {len(rows)}")

    X = np.array([row[0] for row in rows], dtype=np.float32)
    y = np.array([row[1] for row in rows], dtype=np.float32)
    qid = np.zeros(len(rows), dtype=np.uint32)

    unique_labels = sorted(set(float(v) for v in y))
    if len(unique_labels) < 2:
        raise ValueError("training labels must contain at least two distinct relevance values")

    model = xgb.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        tree_method="hist",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=seed,
    )
    model.fit(X, y, qid=qid)
    model.save_model(out_path)

    print(f"trained on {len(rows)} rows")
    print(f"label values: {unique_labels}")
    print(f"saved model: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train the offline XGBoost re-ranker.")
    parser.add_argument("--candidates", help="Path to candidates.jsonl, candidates.jsonl.gz, or a JSON array sample.")
    parser.add_argument("--labels", help="CSV with candidate_id,relevance columns. Relevance should be 0-4.")
    parser.add_argument("--out", default="xgboost_model.json", help="Model artifact path.")
    parser.add_argument("--sample-size", type=int, default=2000, help="Reservoir sample size. Use 0 for all labelled rows.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--use-pseudo-labels",
        action="store_true",
        help="Train from heuristic-derived labels only for demos. Do not describe this as real LLM/hand-labelled training.",
    )
    parser.add_argument(
        "--bootstrap-baseline",
        action="store_true",
        help="Create a bundled baseline model from synthetic feature rows when the real candidate pool is unavailable.",
    )
    args = parser.parse_args()

    if not args.candidates and not args.bootstrap_baseline:
        parser.error("provide --candidates, or use --bootstrap-baseline to create the bundled baseline model")

    train_model(
        candidates_path=None if args.bootstrap_baseline else args.candidates,
        labels_path=args.labels,
        out_path=args.out,
        sample_size=args.sample_size,
        seed=args.seed,
        use_pseudo_labels=args.use_pseudo_labels,
    )


if __name__ == "__main__":
    main()
