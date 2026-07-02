# Redrob Hackathon: Data & AI Challenge - VictoryLap Strategy


## 2. Solution Overview
### What is your proposed solution?
Our solution is a lightning-fast, two-stage pipeline. Stage 1 is an O(N) mathematical heuristic pass that filters 100,000 candidates down to the top 1,000 in seconds. Stage 2 applies an offline-trained XGBoost model (LambdaMART) to re-rank the top 1,000 candidates to produce the final top 100 list.

### What differentiates your approach from traditional candidate matching systems?
Unlike traditional systems that rely heavily on simplistic keyword matching or expensive runtime LLM calls, our system leverages the concept of "what the JD means, not what it says." We use complex heuristics and behavioral signals (e.g., temporal paradoxes, recruiter response rates) for candidate qualification. Additionally, our Stage 2 model uses offline LLM ground-truth generation to train a classic ML model, giving us GPT-4 level intelligence at XGBoost speeds (under 10 seconds on a CPU).

## 3. JD Understanding & Candidate Evaluation
### What are the key requirements extracted from the JD?
- **Core ML depth**: Production experience with embeddings, retrieval, ranking, and vector databases.
- **Product-engineering attitude**: Emphasis on "shipped", "deployed", and "production" over pure academic research.
- **Career trajectory (Goldilocks)**: 2-4 roles with 2-4 year tenures. Penalizing extreme job hoppers (<1.5 yrs) and big-tech lifers who haven't shipped code recently.

### Which candidate signals are most important? / Beyond keyword matching
Keyword matching is easily gamed (Keyword Stuffers). Instead, we heavily weight **career description quality** and **uniqueness**. Furthermore, we treat behavioral signals (the 23 `redrob_signals`) as a *multiplicative* recruitability score. A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate gets aggressively down-ranked, mimicking real-world recruiter behavior.

## 4. Ranking Methodology
### Retrieve, Score, Rank
- **Stage 1 (Filter)**: Evaluates a candidate using `jd_fit_score` and `recruitability_score`. Only candidates passing honeypot checks are added to a size-1,000 min-heap.
- **Stage 2 (Re-rank)**: Transforms the top 1,000 into an 11-dimension feature matrix and applies a pre-trained Learning-to-Rank ML model to predict optimal NDCG rankings.

### Models & Algorithms Used
- **Offline**: GPT-4o / Claude (for generating 0-4 ground truth scores on a 2,000 candidate sample).
- **Online (Runtime)**: pure Python heuristics, `heapq` (Min-Heap for O(N) memory-efficient filtering), and `XGBRanker` (LambdaMART) for ML prediction.

### Combining Signals
We use grouped geometric means for `recruitability_score` (categorized into availability, responsiveness, credibility, and market_signal) multiplied by an additive `jd_fit_score`.

## 5. Explainability & Data Validation
### How are ranking decisions explained?
During the O(N) pass, we memoize verified profile facts (e.g., specific skills, highest-value career entries, years of experience) into a dictionary. The reasoning is generated purely via string assembly based on these pre-validated fragments.

### Preventing Hallucinations
Because the reasoning generator only concatenates strings extracted directly from the candidate's JSON profile during the first pass, it is mathematically impossible for the system to invent employers, skills, or metrics.

### Handling Suspicious Profiles
We implemented a strict `honeypot.py` filter that explicitly drops candidates with inverted timelines, temporal paradoxes (active before signing up), or impossible durations (e.g., 5 expert skills with 0 duration, or skill durations exceeding overall YOE by 50%).

## 6. End-to-End Workflow
1. Parse `candidates.jsonl.gz` lazily (line-by-line).
2. Discard candidate if `is_honeypot()` returns True.
3. Compute `jd_fit_score` and `recruitability_score`. Multiply them for `final_score`.
4. Push `(final_score, candidate_id, memo_dict)` into a 1,000-size min-heap.
5. After the pass, load `xgboost_model.json`.
6. Extract 11 features from the 1,000 `memo_dict`s.
7. Run `model.predict(X)`.
8. Sort candidates by the new ML scores.
9. Take the top 100, assemble their reasoning strings from the memo, and write to `submission.csv`.

## 7. System Architecture
- **Data Ingestion**: Streaming JSON parser via Python `gzip`.
- **Heuristic Engine**: `scorer.py` and `honeypot.py`.
- **State Management**: Memory-efficient Min-Heap (`heapq`).
- **ML Re-Ranker**: `xgboost` model evaluating an 11-dimensional feature matrix.
- **Output Generator**: `reasoning.py` for dynamic template injection and `csv` for writing.

## 8. Results & Performance
- **Quality**: The use of an offline-trained XGBRanker directly targets the `NDCG@10` metric (50% of the challenge score), ensuring the top 10 are highly relevant and behaviorally active.
- **Constraints**: By abandoning heavy runtime LLMs and dense embedding models, the entire script parses 100,000 candidates and ranks them in a fraction of the 5-minute limit, well within the 16 GB RAM and CPU-only bounds.

## 9. Technologies Used
- **Python 3.11** & **Standard Library**: Used for 95% of the codebase to guarantee maximum speed and zero dependency bloat.
- **XGBoost (`XGBRanker`)**: Selected for Stage 2 because tree-based Learning-to-Rank models offer the absolute best performance-to-latency ratio for tabular candidate data.
- **LLM APIs (Offline only)**: Selected to generate ground-truth labels for XGBoost, allowing us to bake deep semantic understanding into a fast runtime model.

## 10. Submission Assets
- `rank.py` (Main entry point)
- `scorer.py`, `honeypot.py`, `reasoning.py`, `constants.py`
- `train_xgboost.py` (Offline training script)
- `xgboost_model.json` (Model weights)
- `README.md` & `strat.md`
- Code Repository & Sandbox Link (TBD)
