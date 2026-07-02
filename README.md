# Redrob Hackathon: Team VictoryLap

Two-stage candidate ranking pipeline for the Intelligent Candidate Discovery & Ranking Challenge.

## Reproduce the submission

```bash
pip install -r requirements.txt
python reproduction_command.py --candidates ./candidates.jsonl --out ./team_VictoryLap.csv
```

`reproduction_command.py` is the **single unified script** that executes the entire pipeline end-to-end. It prints the total execution time to the terminal upon completion. The ranking step completes in **~25 seconds** on a 16 GB CPU-only machine. No network calls, no GPU, no hosted LLM APIs during ranking.

To validate format compliance:

```bash
python validate_submission.py team_VictoryLap.csv
```

## Architecture

**Stage 1 — O(N) heuristic filter (100K → top 2000)**

Each candidate is streamed from `candidates.jsonl` and filtered in a single pass:

1. **Honeypot detection** — Candidates with temporal paradoxes, impossible career durations, or fabricated skill claims are dropped immediately.
2. **Research-only filter** — Candidates whose career history mentions only research with no production experience are dropped (JD explicitly flags this).
3. **JD Fit Score** — 9 weighted sub-scores covering title match, YOE band, career description keywords, product-vs-services company history, career trajectory (2–4 roles, 2–4 year tenures), pre-2022 ML experience, location, skill depth, and education tier.
4. **Recruitability Score** — 23 behavioral signals grouped into 4 categories (availability, responsiveness, credibility, market signal), aggregated via geometric means with a 0.05 floor. Multiplied with salary match and assessment scores.
5. **Final Score** = JD Fit × Recruitability. Top 2000 maintained in a min-heap.

**Stage 2 — XGBRanker re-ranking (top 2000 → top 100)**

An `XGBRanker` (LambdaMART, objective `rank:ndcg`) trained offline on 2000 labelled candidates re-ranks the top 2000 using 7 features:

| Feature | Description |
|---|---|
| `jd_fit_score` | Composite JD alignment score |
| `recruitability_score` | Behavioral signal composite |
| `yoe` | Years of experience |
| `has_production_keywords` | Production deployment mentions in career history |
| `response_rate` | Recruiter response rate signal |
| `matched_skills` | Count of relevant skills with advanced/expert proficiency |
| `location_note` | Location match to JD preference |

The model weights are bundled as `xgboost_model.json`. If the model file is missing, Stage 1 heuristic scores are used directly as a fallback.

## Evaluation metric alignment

| Metric | Weight | How we target it |
|---|---|---|
| NDCG@10 | 50% | XGBRanker optimized with `ndcg@10` eval metric |
| NDCG@50 | 30% | Heuristic pre-filter ensures strong candidates enter the top 2000 |
| MAP | 15% | Multi-signal scoring separates tiers cleanly |
| P@10 | 5% | Honeypot + research-only detection keeps false positives out of the top 10 |

## Honeypot detection

Candidates are dropped if any of these fire:
- Career duration sum vs stated YOE differs by more than 3 years
- 5+ "expert" skills with 0 months duration
- Any career entry duration exceeds its date range by more than 3 months
- Any skill duration exceeds YOE × 1.5
- `signup_date` is after `last_active_date` (temporal paradox)

## Reasoning generation

Each reasoning string is assembled from facts extracted during the O(N) pass (current title, career history, matched skills, behavioral signals). No LLM is called. No facts are invented. Every claim maps to a field in the candidate's JSON record.

## File structure

| File | Purpose |
|---|---|
| `reproduction_command.py` | **Unified entry point.** Runs Stage 1 + Stage 2, writes CSV, prints execution time |
| `scorer.py` | JD fit (9 sub-scores) and recruitability (23 signals) |
| `honeypot.py` | 5-rule honeypot detection |
| `reasoning.py` | Fact-based reasoning assembly |
| `constants.py` | All keyword sets, company lists, thresholds |
| `train_xgboost.py` | Offline XGBRanker training script (pre-computation) |
| `xgboost_model.json` | Pre-trained model artifact |
| `validate_submission.py` | Official format validator |
| `submission_metadata.yaml` | Portal metadata |
| `requirements.txt` | Python dependencies |

## AI tools declaration

This submission was built with assistance from **Claude**, **Gemini**, and **OpenAI Codex**. All architecture decisions, scoring formulas, feature selection, and code review were performed by the team. The AI tools were used for code generation, bug detection, and documentation. The evaluation pipeline is designed so this level of AI-assisted engineering succeeds at Stages 3–5.

## XGBoost training (offline, before submission)

To retrain the model with your own labels:

```bash
python train_xgboost.py --candidates ./candidates.jsonl --labels ./labels.csv --out ./xgboost_model.json
```

`labels.csv` must have columns `candidate_id,relevance` with relevance values 0–4. Training uses an 80/20 train-test split and prints NDCG@10 validation metrics to the terminal.
