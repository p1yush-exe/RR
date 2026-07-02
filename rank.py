import argparse
import json
import csv
import gzip
import heapq
from honeypot import is_honeypot
from scorer import jd_fit_score, recruitability_score
from reasoning import generate_reasoning
from constants import RELEVANT_SKILLS, BIG_TECH

def build_memo(candidate, jd_fit, rec, final):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    num_roles = len(career)
    avg_tenure = sum(c.get("duration_months", 0) for c in career) / max(1, num_roles)

    top_skills = [
        s.get("name")
        for s in skills
        if s.get("name", "").lower() in RELEVANT_SKILLS
    ][:3]

    best_career = career[0] if career else None
    strongest_career_entry = ""
    if best_career:
        title = best_career.get("title", "Role")
        company = best_career.get("company", "Company")
        dur = best_career.get("duration_months", 0)
        strongest_career_entry = f"{title} at {company} ({dur}mo)"

    yoe = profile.get("years_of_experience", 0)
    yoe_note = None
    if 5 <= yoe <= 9:
        yoe_note = f"{yoe} years (sweet spot)"
    elif yoe > 9:
        yoe_note = f"{yoe} years (senior)"

    loc = profile.get("location", "")
    location_note = None
    if any(city in loc.lower() for city in ["pune", "noida"]):
        location_note = f"{loc} (preferred)"

    is_job_hopper = num_roles >= 5 and avg_tenure < 18
    is_big_tech_lifer = (
        num_roles == 1
        and career
        and career[0].get("duration_months", 0) >= 72
        and any(bt in career[0].get("company", "").lower() for bt in BIG_TECH)
    )

    all_titles_desc = " ".join(
        [c.get("title", "") for c in career] + [c.get("description", "") for c in career]
    ).lower()
    is_research_only = "research" in all_titles_desc

    concerns = []
    np_days = signals.get("notice_period_days", 0)
    if np_days > 60:
        concerns.append(f"notice period {np_days} days")
    if "vector" not in all_titles_desc and not any(
        "vector" in s.get("name", "").lower()
        or "pinecone" in s.get("name", "").lower()
        or "faiss" in s.get("name", "").lower()
        for s in skills
    ):
        concerns.append("no vector DB experience")

    positive_signals = []
    if signals.get("recruiter_response_rate", 0) > 0.8:
        positive_signals.append(
            f"response rate {int(signals.get('recruiter_response_rate', 0) * 100)}%"
        )

    if is_job_hopper or is_big_tech_lifer:
        yoe_note = None

    return {
        "candidate_id": candidate.get("candidate_id"),
        "final_score": final,
        "jd_fit_score": jd_fit,
        "recruitability_score": rec,
        "current_title": profile.get("current_title", ""),
        "current_company": profile.get("current_company", ""),
        "yoe": yoe,
        "location": loc,
        "country": profile.get("country", ""),
        "top_matched_skills": top_skills,
        "strongest_career_entry": strongest_career_entry,
        "career_summary": f"{num_roles} roles",
        "concerns": concerns,
        "positive_signals": positive_signals,
        "yoe_note": yoe_note,
        "location_note": location_note,
        "is_research_only": is_research_only,
        "is_job_hopper": is_job_hopper,
        "is_big_tech_lifer": is_big_tech_lifer,
        "has_production_keywords": any(
            kw in all_titles_desc for kw in ["shipped", "deployed", "production"]
        ),
        "notice_days": np_days,
        "response_rate": signals.get("recruiter_response_rate", 0),
    }

def xgb_features_from_memo(memo):
    return [
        memo["jd_fit_score"],
        memo["recruitability_score"],
        float(memo["yoe"]),
        1.0 if memo["is_research_only"] else 0.0,
        1.0 if memo["is_job_hopper"] else 0.0,
        1.0 if memo["is_big_tech_lifer"] else 0.0,
        1.0 if memo["has_production_keywords"] else 0.0,
        float(memo["notice_days"]),
        float(memo["response_rate"]),
        float(len(memo["top_matched_skills"])),
        1.0 if memo["location_note"] else 0.0,
    ]

def output_score(score):
    return round(float(score), 8)

def process_candidates(input_path, output_path):
    top_1000 = []
    
    open_func = gzip.open if input_path.endswith('.gz') else open
    
    with open_func(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            if is_honeypot(candidate):
                continue
                
            jd_fit = jd_fit_score(candidate)
            rec = recruitability_score(candidate)
            final = jd_fit * rec

            memo = build_memo(candidate, jd_fit, rec, final)
            
            if len(top_1000) < 1000:
                heapq.heappush(top_1000, (final, candidate.get("candidate_id", ""), memo))
            else:
                heapq.heappushpop(top_1000, (final, candidate.get("candidate_id", ""), memo))
                
    # Sort top 1000 descending by final_score, breaking ties with candidate_id
    top_1000_sorted = sorted(top_1000, key=lambda x: (-x[0], x[1]))
    
    # --- STAGE 2: XGBRanker (if model exists) ---
    import os
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "xgboost_model.json")
    if os.path.exists(model_path):
        try:
            import xgboost as xgb
            import numpy as np
            model = xgb.XGBRanker()
            model.load_model(model_path)
            
            features = [xgb_features_from_memo(item[2]) for item in top_1000_sorted]
                
            X = np.array(features)
            preds = model.predict(X)
            
            # Re-sort by XGBoost predictions
            for i in range(len(top_1000_sorted)):
                # Overwrite the original score with the XGBoost score
                top_1000_sorted[i] = (float(preds[i]), top_1000_sorted[i][1], top_1000_sorted[i][2])
                
            # Re-sort using the new predictions
            top_1000_sorted.sort(key=lambda x: (-x[0], x[1]))
        except Exception:
            pass # Fall back to heuristic scoring if xgboost is missing or the model is invalid
            
    top_100 = sorted(top_1000_sorted[:100], key=lambda x: (-output_score(x[0]), x[1]))
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for rank, item in enumerate(top_100, 1):
            score, cid, memo = item
            reasoning_str = generate_reasoning(memo, rank)
            writer.writerow([cid, rank, f"{output_score(score):.8f}", reasoning_str])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    process_candidates(args.candidates, args.out)
