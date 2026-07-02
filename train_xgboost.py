import json
import xgboost as xgb
import numpy as np
import random
from scorer import jd_fit_score, recruitability_score

def mock_llm_score(candidate):
    """
    In a real scenario, this would call an LLM API (GPT-4o / Claude 3.5 Sonnet) 
    with the Job Description and the Candidate JSON, and ask it to rate 0-4.
    Here we simulate it using our heuristic score + some noise.
    """
    base_score = jd_fit_score(candidate) * recruitability_score(candidate)
    
    # 0 to 4 ranking
    rank = int(base_score * 4) + random.randint(-1, 1)
    return max(0, min(4, rank))

def train_model(input_path="candidates.jsonl", num_samples=2000):
    print(f"Sampling {num_samples} candidates for offline LLM evaluation...")
    
    # Read a sample of candidates
    candidates = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for _ in range(num_samples):
            line = f.readline()
            if not line: break
            try:
                candidates.append(json.loads(line))
            except:
                continue

    print("Generating LLM ground truth scores (simulated)...")
    X = []
    y = []
    groups = [len(candidates)] # For XGBRanker, all in one query group for this example
    
    for c in candidates:
        # Re-compute features just like in rank.py
        profile = c.get("profile", {})
        career = c.get("career_history", [])
        skills = c.get("skills", [])
        signals = c.get("redrob_signals", {})
        
        jd_fit = jd_fit_score(c)
        rec = recruitability_score(c)
        
        num_roles = len(career)
        avg_tenure = sum(cr.get("duration_months", 0) for cr in career) / max(1, num_roles)
        
        is_job_hopper = num_roles >= 5 and avg_tenure < 18
        is_big_tech_lifer = num_roles == 1 and career and career[0].get("duration_months", 0) >= 72
        all_titles_desc = " ".join([cr.get("title", "") for cr in career] + [cr.get("description", "") for cr in career]).lower()
        is_research_only = "research" in all_titles_desc
        has_production_keywords = any(kw in all_titles_desc for kw in ["shipped", "deployed", "production"])
        
        # Exact same 11 features as rank.py
        features = [
            jd_fit,
            rec,
            float(profile.get("years_of_experience", 0)),
            1.0 if is_research_only else 0.0,
            1.0 if is_job_hopper else 0.0,
            1.0 if is_big_tech_lifer else 0.0,
            1.0 if has_production_keywords else 0.0,
            float(signals.get("notice_period_days", 0)),
            float(signals.get("recruiter_response_rate", 0)),
            float(len(skills)), # Simplified for training script
            1.0 if "pune" in profile.get("location", "").lower() else 0.0
        ]
        
        X.append(features)
        y.append(mock_llm_score(c))
        
    X = np.array(X)
    y = np.array(y)
    
    print("Training XGBRanker...")
    model = xgb.XGBRanker(
        tree_method="hist",
        objective="rank:ndcg",
        eval_metric="ndcg"
    )
    
    model.fit(X, y, qid=np.zeros(len(X))) # Single query group (qid=0)
    
    model.save_model("xgboost_model.json")
    print("Model saved to xgboost_model.json")

if __name__ == "__main__":
    import sys
    # For testing, we can use sample_candidates.json
    file_to_use = sys.argv[1] if len(sys.argv) > 1 else "sample_candidates.json"
    train_model(file_to_use, num_samples=50)
