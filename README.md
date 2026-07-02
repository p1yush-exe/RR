# Redrob Hackathon: Data & AI Challenge - Team VictoryLap

Welcome to our submission for the Redrob Intelligent Candidate Discovery & Ranking Challenge. Our solution focuses on an ultra-fast, two-stage ranking pipeline designed to surface the best 100 candidates out of 100,000 without relying on expensive real-time LLM calls or dense embedding models that fail under strict compute constraints.

---

## How to Use Instructions

To execute our ranking pipeline locally within the 5-minute CPU constraint, follow these instructions:

<details>
<summary><strong>Click to expand setup and execution steps</strong></summary>

### 1. Requirements
Ensure you are running **Python 3.11+**. Install the single external dependency:
```bash
pip install xgboost numpy
```

### 2. Execution
Place the `candidates.jsonl` (or `candidates.jsonl.gz`) file in the root directory. Run the main orchestrator:
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```
*The pipeline will automatically load the pre-trained `xgboost_model.json`.*

### 3. Validation
Validate the output format against the official checker:
```bash
python validate_submission.py submission.csv
```

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

## Repository Structure

<details>
<summary><strong>Expand to see file descriptions</strong></summary>

- `rank.py`: Main entry point and Phase 2/Phase 3 orchestrator.
- `scorer.py`: Pure math heuristics evaluating JD Fit and Recruitability (all 23 signals).
- `honeypot.py`: Strict rules for detecting trap candidates.
- `reasoning.py`: Fact-based string assembly to guarantee 0% hallucination in reasoning.
- `constants.py`: Keyword sets, lists, and thresholds.
- `train_xgboost.py`: The script showcasing our offline LLM training loop.
- `xgboost_model.json`: Pre-trained model weights for Phase 3.
- `strat.md`: Complete strategic breakdown mapping to the presentation template.
- `agent.md`: The original machine-readable architectural specification.

</details>
