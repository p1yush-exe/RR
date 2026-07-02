# XGBoost Training and Operating Guide

This project uses XGBoost only as an offline-trained re-ranker. The timed submission command must not call APIs, use network access, or train models. It should only load the bundled `xgboost_model.json` and rank candidates.

## What the Model Artifact Is

`xgboost_model.json` is the trained model weights file. Bundling the artifact means keeping this file in the repository so reviewers can clone the repo, install dependencies, place the candidate file, and run:

```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```

They should not need your private notebook, hidden labels, or a retraining step to reproduce the ranking command.

## Feature Vector

Training and inference both use the same 11 features from `rank.py`:

1. `jd_fit_score`
2. `recruitability_score`
3. years of experience
4. research-only flag
5. job-hopper flag
6. big-tech-lifer flag
7. production-keyword flag
8. notice period days
9. recruiter response rate
10. number of matched relevant skills
11. preferred-location flag

Do not edit these in only one place. `train_xgboost.py` imports the runtime feature extractor from `rank.py` specifically to avoid training/inference drift.

## Best Real Training Workflow

1. Put the real pool in the repo root as `candidates.jsonl.gz`.

2. Create `labels.csv` from offline review:

   ```csv
   candidate_id,relevance
   CAND_0001234,4
   CAND_0005678,2
   CAND_0009999,0
   ```

   Use 0-4 labels:

   - `4`: excellent founding Senior AI Engineer fit
   - `3`: strong fit with minor concern
   - `2`: plausible but incomplete fit
   - `1`: weak adjacent fit
   - `0`: not relevant, honeypot, keyword stuffer, or wrong domain

3. Train offline:

   ```bash
   python train_xgboost.py --candidates ./candidates.jsonl.gz --labels ./labels.csv --out ./xgboost_model.json --sample-size 2000
   ```

4. Run the ranker:

   ```bash
   python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
   ```

5. Validate:

   ```bash
   python examples/validate_submission.py submission.csv
   ```

## How to Get Labels Fast

If you have 4 hours, label a small high-signal set:

1. Run the heuristic ranker once without changing anything.
2. Manually review the top 300-500 candidates plus a random 200 lower-ranked candidates.
3. Label them 0-4 using the JD, not just skill keywords.
4. Make sure obvious bad profiles are labelled `0`: non-AI titles stuffed with AI skills, impossible timelines, pure research with no production deployment, services-only career if no product-company evidence.
5. Retrain and rerun `rank.py`.

More labels are better, but clean labels beat noisy volume.

## Offline LLM Labelling

If using an LLM before submission, do it outside the timed run:

1. Sample candidate JSON records.
2. Send each candidate plus the JD to the LLM.
3. Ask for a strict 0-4 relevance label and a short rationale.
4. Save only `candidate_id,relevance` into `labels.csv`.
5. Train `xgboost_model.json`.

Do not call the LLM from `rank.py`.

## Baseline Artifact

This repo can create a baseline model when the real dataset is not present:

```bash
python train_xgboost.py --bootstrap-baseline --out ./xgboost_model.json
```

That model is useful for fulfilling the bundled-artifact requirement and exercising the Stage 2 code path. It is not as strong as a model trained from real manual or LLM labels.

## When You Replace the Model

After retraining, keep these files together:

- `xgboost_model.json`
- `train_xgboost.py`
- `rank.py`
- `requirements.txt`

Then rerun validation. If `rank.py` still completes and the CSV validates, the bundled artifact is operational.
