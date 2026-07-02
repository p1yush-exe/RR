# Redrob Hackathon: Data & AI Challenge - Team VictoryLap

Welcome to our submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge. Our solution focuses on an ultra-fast, two-stage ranking pipeline designed to surface the best 100 candidates out of 100,000 without relying on expensive real-time LLM calls or dense embedding models that fail under strict compute constraints.

---

## How to Use Instructions

To execute our ranking pipeline locally within the 5-minute CPU constraint, follow these instructions:

<details>
<summary><strong>Click to expand setup and execution steps</strong></summary>

### 1. Requirements
Ensure you are running **Python 3.11+**. Install the external dependencies:
```bash
pip install -r requirements.txt
```

### 2. Execution
Place the `candidates.jsonl` (or `candidates.jsonl.gz`) file in the root directory. Run the main orchestrator:
```bash
python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
```
The bundled `xgboost_model.json` is loaded automatically for Stage 2 re-ranking. The code has a defensive heuristic fallback if the artifact is missing or invalid, but the submitted repo includes the model file.

### 3. Validation
Validate the output format against the official checker:
```bash
python examples/validate_submission.py submission.csv
```

### 4. Offline XGBoost Training
The bundled model can be regenerated before the timed submission run using labelled candidates:
```bash
python train_xgboost.py --candidates ./candidates.jsonl.gz --labels ./labels.csv --out ./xgboost_model.json --sample-size 2000
```
See `train.md` for the full operating guide.

</details>

---

## Architectural Overview

Our core philosophy is **"Read what the JD means, not just what it says"**. Simplistic keyword matching is easily gamed by 'Keyword Stuffer' profiles. Instead, we score candidates on deep heuristics, career trajectories, and multi-dimensional behavioral signals.

We divided the challenge into three distinct phases:

### Phase 1: Offline Pre-Computation (LLM + XGBoost)
Before the submission run, we sampled 2,000 candidates and used an LLM API to score them against the Job Description. We then extracted 11 mathematical features and trained an `XGBRanker` (LambdaMART). This bakes deep semantic understanding into lightning-fast mathematical weights, bypassing the need for runtime API calls.

### Phase 2: O(N) Single-Pass Heuristic Filter
We stream the 100,000 `candidates.jsonl.gz` file. Each candidate is parsed, checked for honeypot traits (such as temporal paradoxes), and scored mathematically. The top 1,000 are maintained in a memory-efficient Min-Heap (`heapq`), meaning memory consumption remains flat regardless of dataset size.

### Phase 3: XGBoost Re-Ranker & Reasoning Output
We pass the 11-dimension feature matrix of the top 1,000 candidates into the `XGBRanker`. The model predicts the optimal NDCG order in under a second. The top 100 are sliced off, and we use a zero-hallucination memoization bag (assembled during Phase 2) to build dynamic, fact-based reasoning strings.

---

## Metrics and Formulas

To measure candidate quality, we do not rely on simple addition. We engineered a **Recruitability Score** using grouped geometric means across 23 behavioral signals. Geometric means punish severe deficiencies—if a candidate has a perfect profile but a 0% recruiter response rate, their final score drops to zero.

### Final Score Formula
The Stage 1 Final Score is the product of Job Fit and Recruitability:

$$ \text{Final Score} = \text{JD Fit Score} \times \text{Recruitability Score} $$

### Recruitability Grouping

<details>
<summary><strong>View Recruitability Sub-Metrics</strong></summary>

| Group | Signals Included | Aggregation Method |
|:---|:---|:---|
| **Availability** | Recency, Open to Work, Notice Period, Work Mode, Relocation | Geometric Mean (power of 1/5) |
| **Responsiveness** | Response Rate, Response Time, Interview Rate, Offer Accept Rate | Geometric Mean (power of 1/4) |
| **Credibility** | Profile Completeness, Verified Email, Verified Phone, LinkedIn, GitHub | Geometric Mean (power of 1/5) |
| **Market Signal** | Views, Search Appearances, Saved, Connections, Endorsements, Applications | Geometric Mean (power of 1/6) |

</details>

### Recruitability Formula

$$ \text{Recruitability} = \text{Availability} \times \text{Responsiveness} \times \text{Credibility} \times \text{Market Signal} \times \text{Salary Match} \times \text{Avg Assessment} $$

### Challenge Evaluation Metrics
Our offline XGBoost model is specifically optimized for **NDCG** (Normalized Discounted Cumulative Gain), which aligns with the Hackathon's scoring weights:

| Metric | Hackathon Weight | Our Optimization Strategy |
|:---|:---|:---|
| **NDCG@10** | 50% | Stage 2 XGBRanker focuses heavily on top-10 precision |
| **NDCG@50** | 30% | Enforced via Min-Heap thresholding |
| **MAP** | 15% | Global sorting mechanisms |
| **P@10** | 5% | Strict Honeypot evasion heuristics |

---

## Compliance with Hackathon Rules (README.docx)

To respect the strict guidelines outlined in the hackathon's `README.docx` and `submission_spec.docx`:

- **AI Tools Declaration:** Yes, this codebase was co-authored with AI coding agents (Claude, Gemini, etc.). However, as required, we have done "real engineering"—this is not an API-wrapper script. The entire pipeline runs locally on CPU with zero network calls during the 5-minute evaluation window.
- **Honeypot & Trap Evasion:** The dataset contains ~80 honeypots (impossible profiles) and 'Keyword Stuffers'. Our `honeypot.py` actively searches for temporal paradoxes and mathematically impossible career durations to ensure our top-100 honeypot rate stays strictly at 0% (well below the 10% disqualification threshold).
- **Compute Constraints:** The `rank.py` script leverages a highly optimized O(N) streaming parser and an in-place `heapq`, executing well within the 16 GB memory and 5-minute CPU constraints.

---

## Repository Structure

<details>
<summary><strong>Expand to see file descriptions</strong></summary>

- `rank.py`: Main entry point and Phase 2/Phase 3 orchestrator.
- `scorer.py`: Pure math heuristics evaluating JD Fit and Recruitability (all 23 signals).
- `honeypot.py`: Strict rules for detecting trap candidates.
- `reasoning.py`: Fact-based string assembly to guarantee 0% hallucination in reasoning.
- `constants.py`: Keyword sets, lists, and thresholds.
- `train_xgboost.py`: Offline training script for real labelled data and baseline artifact generation.
- `xgboost_model.json`: Bundled model weights for Phase 3 re-ranking.
- `requirements.txt`: Python dependencies for reproducible setup.
- `train.md`: Practical XGBoost training and operating guide.
- `strat.md`: Complete strategic breakdown mapping to the presentation template.
- `agent.md`: The original machine-readable architectural specification.

</details>

---

## Changelog & Recent Verification

- **Code Refactoring:** `rank.py` has been refactored to ensure training and inference use the exact same 11-feature extractor.
- **Model Artifact:** The `xgboost_model.json` artifact is now generated, bundled, and explicitly tracked in `.gitignore`, rather than being an optional runtime fallback.
- **Documentation:** The `learning_lab.py` script was removed to streamline the repo, and `train.md` was completely rewritten as a practical XGBoost operating guide.
- **Dependency Management:** A `requirements.txt` file was added for precise environment reproduction.
- **Validation:** All Python compile checks pass, the XGBoost model loads correctly, and the pipeline has successfully generated a `submission.csv` that passes `validate_submission.py`.
