# Redrob Hackathon — Implementation Spec

> **FOR AI CODING AGENTS. DO NOT DEVIATE. DO NOT IMPROVISE. FOLLOW EXACTLY.**

## CONSTRAINTS
- Runtime: ≤5 min wall-clock, CPU only, 16GB RAM, no GPU, no network
- Output: CSV, 100 rows + header, columns: `candidate_id,rank,score,reasoning`
- Input: `candidates.jsonl` (100,000 lines, one JSON object per line)
- Validate with: `python validate_submission.py submission.csv`
- No AI/LLM models. No embeddings models. Pure Python + math only.
- **Dependencies allowed:** `numpy`, `xgboost`. Everything else is stdlib.

## FILE STRUCTURE
```
rank.py              # Main entry point. Single command: python rank.py --candidates ./candidates.jsonl --out ./submission.csv
scorer.py            # All scoring functions
honeypot.py          # Honeypot detection
reasoning.py         # Reasoning generation from memo
constants.py         # All keyword lists, weights, thresholds
```

## ARCHITECTURE: 2-STAGE PIPELINE

### OFFLINE PRE-COMPUTATION: LLM + XGBOOST
(Run offline before submission, no time limits, LLM API allowed)
1. Sample ~2,000 candidates from `candidates.jsonl`.
2. Send each profile to GPT-4o / Claude 3.5 API with the JD to get a 0-4 relevance score (Ground Truth).
3. Train an `XGBRanker` on the 11 numeric/boolean features generated during the O(N) pass.
4. Save weights to `xgboost_model.json`.

### STAGE 1: O(N) SINGLE PASS (100K → top 1000)
(Runs in 5-minute window, CPU only, no network)
For each candidate JSON line:
1. Parse JSON
2. Run honeypot check → if honeypot, skip entirely
3. Compute `jd_fit_score` (float, 0+)
4. Compute `recruitability_score` (float, 0.0-1.0)
5. Compute `final_score = jd_fit_score * recruitability_score`
6. Collect `memo` dict (features + reasoning fragments)
7. Insert into a min-heap of size 1000 (keep top 1000 by final_score)

### STAGE 2: XGBOOST RE-RANKER (top 1000 → top 100)
1. Sort top 1000 by Stage 1 `final_score` descending.
2. Load pre-trained `xgboost_model.json`.
3. Construct the 11-dimension feature matrix from `memo`.
4. Run `model.predict(X)` to get ML relevance scores.
5. Re-sort the top 1000 by the new XGBoost scores.
6. Take the top 100, assign ranks 1-100.
7. Generate reasoning string from each memo, write CSV.

---

## HONEYPOT DETECTION (honeypot.py)
Return `True` (drop candidate) if ANY of these are true:
- `abs(sum(career_history[*].duration_months)/12 - profile.years_of_experience) > 3.0`
- Count of skills where `proficiency == "expert" AND duration_months == 0` is `>= 5`
- Any `career_history` entry: `duration_months > (end_date - start_date in months) + 3` (allow 3mo buffer; skip if `end_date` is null)
- Any skill: `duration_months > profile.years_of_experience * 12 * 1.5` (50% over = flag)
- `signup_date > last_active_date` (temporal paradox)

Do NOT hard-drop for inverted salary (`min > max`). Use as penalty signal instead.

---

## JD FIT SCORE (scorer.py)
Sum of weighted sub-scores. Each sub-score is 0.0 or 1.0 unless noted.

### Sub-score 1: Title Match (weight 3.0)
```python
POSITIVE_TITLES = {"ai", "ml", "machine learning", "data scientist", "nlp",
    "search", "retrieval", "ranking", "recommendation", "deep learning",
    "computer vision", "research engineer", "applied scientist"}
NEGATIVE_TITLES = {"marketing", "sales", "hr", "human resource", "support",
    "admin", "finance", "accountant", "civil", "mechanical", "graphic",
    "content writer", "project manager", "operations", "business analyst"}

titles = [profile.current_title] + [c.title for c in career_history]
all_titles_lower = " ".join(titles).lower()

if any(neg in all_titles_lower for neg in NEGATIVE_TITLES):
    if any skill.name.lower() matches AI keywords:
        score = -1.0  # keyword stuffer penalty
    else:
        score = 0.0
elif any(pos in all_titles_lower for pos in POSITIVE_TITLES):
    score = 1.0
else:
    score = 0.0
```

### Sub-score 2: YOE Match (weight 2.0)
```python
yoe = profile.years_of_experience
if 5 <= yoe <= 9:    score = 1.0
elif 4 <= yoe < 5:   score = 0.6
elif 9 < yoe <= 12:  score = 0.6
elif 3 <= yoe < 4:   score = 0.3
elif 12 < yoe <= 15: score = 0.3
else:                 score = 0.0
```

### Sub-score 3: Career Description Quality (weight 3.0)
```python
STRONG_KEYWORDS = {"shipped", "deployed", "production", "ranking", "retrieval",
    "embeddings", "recommendation", "search", "vector", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "xgboost", "lightgbm", "ndcg", "mrr",
    "a/b test", "fine-tun", "transformer", "bert", "sentence-transformer",
    "pytorch", "tensorflow", "inference", "latency", "throughput", "scale"}

all_descriptions = " ".join(c.description for c in career_history).lower()
hit_count = sum(1 for kw in STRONG_KEYWORDS if kw in all_descriptions)

score = min(1.0, hit_count / 5.0)  # 5+ hits = full score
```

### Sub-score 4: Product Company Experience (weight 2.0)
```python
SERVICES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini",
    "hcl", "tech mahindra", "mphasis", "mindtree", "ltimindtree", "persistent",
    "hexaware", "cyient", "l&t infotech", "zensar"}

companies = [c.company.lower() for c in career_history]
all_services = all(any(svc in comp for svc in SERVICES) for comp in companies)

if all_services:
    score = -1.0  # penalty: services-only career
elif any(any(svc in comp for svc in SERVICES) for comp in companies):
    score = 0.3   # mixed
else:
    score = 1.0   # all product companies
```

### Sub-score 5: Career Trajectory — Goldilocks (weight 2.0)
```python
BIG_TECH = {"microsoft", "google", "meta", "facebook", "amazon", "apple",
    "netflix", "uber", "oracle", "ibm", "intel", "nvidia", "salesforce", "adobe"}

num_roles = len(career_history)
avg_tenure = mean(c.duration_months for c in career_history)
has_big_tech_only = all(any(bt in c.company.lower() for bt in BIG_TECH) for c in career_history)
is_single_long = num_roles == 1 and career_history[0].duration_months >= 72

if num_roles >= 2 and num_roles <= 4 and 24 <= avg_tenure <= 48:
    score = 1.0   # ideal
elif num_roles >= 5 and avg_tenure < 18:
    score = -0.5  # job hopper
elif is_single_long and has_big_tech_only:
    score = -0.25 # big tech lifer
elif is_single_long:
    score = 0.0   # single company but not big tech — neutral
else:
    score = 0.5   # everything else
```

### Sub-score 6: Pre-2022 ML (weight 1.5)
```python
# Check if any career entry with ML-related title started before 2022-01-01
has_pre_2022 = any(
    c.start_date < "2022-01-01" and any(kw in c.title.lower() for kw in POSITIVE_TITLES)
    for c in career_history
)
score = 1.0 if has_pre_2022 else 0.0
```

### Sub-score 7: Location Match (weight 1.5)
```python
loc = profile.location.lower()
country = profile.country.lower()
willing = redrob_signals.willing_to_relocate

if country != "india" and not willing:
    score = -0.5
elif any(city in loc for city in ["pune", "noida"]):
    score = 1.0
elif any(city in loc for city in ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurgaon", "gurugram", "chennai"]):
    score = 0.6
elif country == "india":
    score = 0.3
else:
    score = 0.0
```

### Sub-score 8: Relevant Skill Depth (weight 1.5)
```python
RELEVANT_SKILLS = {"python", "pytorch", "tensorflow", "embeddings", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "nlp", "bert", "transformers",
    "sentence-transformers", "information retrieval", "ranking", "recommendation",
    "xgboost", "lightgbm", "deep learning", "machine learning", "mlops",
    "elasticsearch", "opensearch", "vector database", "langchain", "llm",
    "fine-tuning", "rag", "huggingface"}

relevant = [s for s in skills if s.name.lower() in RELEVANT_SKILLS
            and s.proficiency in ("advanced", "expert")
            and s.duration_months >= 12]
score = min(1.0, len(relevant) / 3.0)  # 3+ deep relevant skills = full
```

### Sub-score 9: Education Tier (weight 0.5)
```python
best_tier = min((e.tier for e in education if e.tier), default="unknown")
# tier_1 > tier_2 > tier_3 > tier_4 > unknown
if best_tier == "tier_1":   score = 1.0
elif best_tier == "tier_2": score = 0.6
else:                       score = 0.0
```

---

## RECRUITABILITY SCORE (scorer.py)
Uses ALL 23 redrob_signals. Multiplicative formula. Result: 0.0-1.0.

```python
def recruitability(s):
    # s = redrob_signals dict

    # --- Signal 1: profile_completeness_score (0-100) ---
    completeness = s["profile_completeness_score"] / 100.0  # normalize to 0-1

    # --- Signal 2: signup_date --- (used in honeypot check only, not scored here)

    # --- Signal 3: last_active_date ---
    days_inactive = (REFERENCE_DATE - parse(s["last_active_date"])).days
    # REFERENCE_DATE = date(2026, 7, 1)
    if days_inactive <= 30:    recency = 1.0
    elif days_inactive <= 90:  recency = 0.8
    elif days_inactive <= 180: recency = 0.5
    elif days_inactive <= 365: recency = 0.2
    else:                      recency = 0.1

    # --- Signal 4: open_to_work_flag ---
    open_to_work = 1.0 if s["open_to_work_flag"] else 0.6

    # --- Signal 5: profile_views_received_30d ---
    views = min(1.0, s["profile_views_received_30d"] / 100.0)  # 100+ views = full

    # --- Signal 6: applications_submitted_30d ---
    # High = actively looking (good). 0 = passive. Very high (>20) = desperation.
    apps = s["applications_submitted_30d"]
    if 1 <= apps <= 15:   apps_score = 1.0
    elif apps == 0:       apps_score = 0.7  # passive but not bad
    else:                 apps_score = 0.5  # >15 = spray-and-pray

    # --- Signal 7: recruiter_response_rate (0-1) ---
    response_rate = s["recruiter_response_rate"]  # use directly

    # --- Signal 8: avg_response_time_hours ---
    rt = s["avg_response_time_hours"]
    if rt <= 6:      response_time = 1.0
    elif rt <= 24:   response_time = 0.9
    elif rt <= 48:   response_time = 0.7
    elif rt <= 72:   response_time = 0.5
    else:            response_time = 0.3

    # --- Signal 9: skill_assessment_scores ---
    assessments = s["skill_assessment_scores"]  # dict or empty
    if assessments:
        avg_assessment = mean(assessments.values()) / 100.0
    else:
        avg_assessment = 0.5  # neutral if no assessments taken

    # --- Signal 10: connection_count ---
    connections = min(1.0, s["connection_count"] / 500.0)  # 500+ = full

    # --- Signal 11: endorsements_received ---
    endorsements = min(1.0, s["endorsements_received"] / 100.0)  # 100+ = full

    # --- Signal 12: notice_period_days ---
    np_days = s["notice_period_days"]
    if np_days <= 30:    notice = 1.0
    elif np_days <= 60:  notice = 0.8
    elif np_days <= 90:  notice = 0.6
    elif np_days <= 120: notice = 0.4
    else:                notice = 0.3

    # --- Signal 13: expected_salary_range_inr_lpa ---
    sal_min = s["expected_salary_range_inr_lpa"]["min"]
    sal_max = s["expected_salary_range_inr_lpa"]["max"]
    sal_mid = (sal_min + sal_max) / 2.0
    inverted_salary = sal_min > sal_max
    if inverted_salary:
        salary = 0.7  # penalty but not hard drop (25% of candidates have this)
    elif 20 <= sal_mid <= 65:
        salary = 1.0  # market rate for senior AI in India
    elif sal_mid < 10:
        salary = 0.4  # too low, likely junior/non-tech
    else:
        salary = 0.7

    # --- Signal 14: preferred_work_mode ---
    mode = s["preferred_work_mode"]
    if mode in ("hybrid", "flexible"): work_mode = 1.0  # JD is hybrid
    elif mode == "onsite":             work_mode = 0.8
    else:                              work_mode = 0.6  # remote-only is mild mismatch

    # --- Signal 15: willing_to_relocate ---
    relocate = 1.0 if s["willing_to_relocate"] else 0.7

    # --- Signal 16: github_activity_score ---
    gh = s["github_activity_score"]
    if gh == -1:       github = 0.5  # no github linked — neutral
    elif gh >= 50:     github = 1.0
    elif gh >= 30:     github = 0.8
    else:              github = 0.6

    # --- Signal 17: search_appearance_30d ---
    search_app = min(1.0, s["search_appearance_30d"] / 200.0)

    # --- Signal 18: saved_by_recruiters_30d ---
    saved = min(1.0, s["saved_by_recruiters_30d"] / 20.0)

    # --- Signal 19: interview_completion_rate (0-1) ---
    interview = s["interview_completion_rate"]  # use directly

    # --- Signal 20: offer_acceptance_rate ---
    oar = s["offer_acceptance_rate"]
    if oar == -1:      offer_accept = 0.5  # no data — neutral
    elif oar >= 0.5:   offer_accept = 1.0
    elif oar >= 0.3:   offer_accept = 0.7
    else:              offer_accept = 0.4  # low acceptance = flight risk

    # --- Signal 21: verified_email ---
    email_v = 1.0 if s["verified_email"] else 0.8

    # --- Signal 22: verified_phone ---
    phone_v = 1.0 if s["verified_phone"] else 0.8

    # --- Signal 23: linkedin_connected ---
    linkedin = 1.0 if s["linkedin_connected"] else 0.7

    # === FINAL RECRUITABILITY ===
    # Group into 4 categories, take geometric mean within each, then multiply
    availability = (recency * open_to_work * notice * work_mode * relocate) ** (1/5)
    responsiveness = (response_rate * response_time * interview * offer_accept) ** (1/4)
    credibility = (completeness * email_v * phone_v * linkedin * github) ** (1/5)
    market_signal = (views * search_app * saved * connections * endorsements * apps_score) ** (1/6)

    return availability * responsiveness * credibility * market_signal * salary * avg_assessment
```

---

## MEMO DICT (collected per candidate during O(N) pass)
```python
memo = {
    "candidate_id": str,         # CAND_XXXXXXX
    "final_score": float,        # jd_fit * recruitability
    "jd_fit_score": float,
    "recruitability_score": float,

    # Reasoning fragments (all from actual profile data, never invented)
    "current_title": str,        # profile.current_title
    "current_company": str,      # profile.current_company
    "yoe": float,                # profile.years_of_experience
    "location": str,             # profile.location
    "country": str,              # profile.country
    "top_matched_skills": list,  # up to 3 skill names that match RELEVANT_SKILLS
    "strongest_career_entry": str,  # f"{title} at {company} ({duration_months}mo)"
    "career_summary": str,       # f"{num_roles} roles including {company1}, {company2}"
    "concerns": list,            # e.g. ["notice period 120 days", "no vector DB experience"]
    "positive_signals": list,    # e.g. ["active 5d ago", "response rate 91%", "saved by 15 recruiters"]
    "yoe_note": str or None,     # "6.2 years (sweet spot)" or None if trajectory is bad
    "location_note": str or None,# "Pune (preferred)" or None
    "is_research_only": bool,
    "is_job_hopper": bool,
    "is_big_tech_lifer": bool,
    "has_production_keywords": bool,
    "notice_days": int,
    "response_rate": float,
}
```

---

## REASONING GENERATION (reasoning.py)
```python
def generate_reasoning(memo, rank):
    parts = []

    # 1. Lead with career
    parts.append(memo["strongest_career_entry"])

    # 2. Matched skills (max 3, from actual profile)
    if memo["top_matched_skills"]:
        parts.append(f"production experience in {', '.join(memo['top_matched_skills'][:3])}")

    # 3. YOE (only if meaningful)
    if memo["yoe_note"]:
        parts.append(memo["yoe_note"])

    # 4. Location (only if positive)
    if memo["location_note"]:
        parts.append(memo["location_note"])

    # 5. Behavioral positives (top 30 only)
    if rank <= 30 and memo["positive_signals"]:
        parts.append(memo["positive_signals"][0])

    # 6. Concerns (rank 50+ only)
    if rank >= 50 and memo["concerns"]:
        parts.append(f"concern: {memo['concerns'][0]}")

    # 7. Research flag (rank 40+ only)
    if memo["is_research_only"] and rank >= 40:
        parts.append("career leans research-focused")

    # 8. Job hopper flag
    if memo["is_job_hopper"] and rank >= 40:
        parts.append("frequent role changes")

    return "; ".join(parts) + "."
```

---

## CSV OUTPUT RULES
- Header: `candidate_id,rank,score,reasoning`
- 100 data rows, ranks 1-100
- `score` must be monotonically non-increasing (rank 1 has highest score)
- If two candidates have equal `final_score`, break ties by `candidate_id` ascending
- Wrap `reasoning` in double quotes, escape any internal quotes

---

## CONSTANTS (constants.py)
```python
REFERENCE_DATE = date(2026, 7, 1)

POSITIVE_TITLES = {"ai", "ml", "machine learning", "data scientist", "nlp",
    "search", "retrieval", "ranking", "recommendation", "deep learning",
    "computer vision", "research engineer", "applied scientist",
    "backend engineer", "software engineer", "platform engineer"}

NEGATIVE_TITLES = {"marketing", "sales", "hr", "human resource", "support",
    "admin", "finance", "accountant", "civil", "mechanical", "graphic",
    "content writer", "project manager", "operations", "business analyst"}

RELEVANT_SKILLS = {"python", "pytorch", "tensorflow", "embeddings", "faiss",
    "pinecone", "weaviate", "qdrant", "milvus", "nlp", "bert", "transformers",
    "sentence-transformers", "information retrieval", "ranking", "recommendation",
    "xgboost", "lightgbm", "deep learning", "machine learning", "mlops",
    "elasticsearch", "opensearch", "vector database", "langchain", "llm",
    "fine-tuning", "rag", "huggingface", "scikit-learn", "keras", "spacy",
    "opencv", "spark", "airflow", "docker", "kubernetes"}

STRONG_DESC_KEYWORDS = {"shipped", "deployed", "production", "ranking", "retrieval",
    "embeddings", "recommendation", "search", "vector", "faiss", "pinecone",
    "weaviate", "qdrant", "milvus", "xgboost", "lightgbm", "ndcg", "mrr",
    "a/b test", "fine-tun", "transformer", "bert", "sentence-transformer",
    "pytorch", "tensorflow", "inference", "latency", "throughput", "scale",
    "pipeline", "feature engineering", "model training", "evaluation"}

SERVICES_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant",
    "capgemini", "hcl", "tech mahindra", "mphasis", "mindtree", "ltimindtree",
    "persistent", "hexaware", "cyient", "zensar", "l&t infotech"}

BIG_TECH = {"microsoft", "google", "meta", "facebook", "amazon", "apple",
    "netflix", "uber", "oracle", "ibm", "intel", "nvidia", "salesforce", "adobe"}
```

---

## EXECUTION
```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
python validate_submission.py submission.csv
```

## STRICT RULES FOR IMPLEMENTING AGENT
- **DO NOT** add any AI/ML model, embedding model, or LLM call.
- **DO NOT** add dependencies beyond `numpy`. Use stdlib for everything else.
- **DO NOT** change the scoring formulas, weights, or thresholds.
- **DO NOT** change the honeypot detection heuristics.
- **DO NOT** change the reasoning generation logic.
- **DO NOT** change the file structure (rank.py, scorer.py, honeypot.py, reasoning.py, constants.py, train_xgboost.py).
- **DO** use a min-heap (heapq) of size 1000 for Stage 1 efficiency.
- **DO** ensure the XGBoost re-ranking falls back gracefully if the model file is missing.
- **DO** handle edge cases: missing fields, null values, empty arrays.
- **DO** ensure CSV passes `validate_submission.py`.
- **DO** test on `sample_candidates.json` first (treat it as jsonl: load as JSON array, iterate).
