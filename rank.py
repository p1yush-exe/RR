import argparse
import json
import csv
import gzip
import heapq
from pathlib import Path
from honeypot import is_honeypot
from scorer import jd_fit_score, recruitability_score
from reasoning import generate_reasoning
from constants import RELEVANT_SKILLS, BIG_TECH


def iter_candidates(path):
    """Read candidates from either a JSON array file or a JSONL/JSONL.GZ file."""
    path = Path(path)
    open_func = gzip.open if str(path).endswith(".gz") else open
    with open_func(path, "rt", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            yield from json.load(f)
            return
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def match_relevant_skills(skills):
    """Substring match: a skill like 'Fine-tuning LLMs' matches 'fine-tuning'."""
    matched = []
    for s in skills:
        name = s.get("name", "").lower()
        if any(rs in name for rs in RELEVANT_SKILLS) or name in RELEVANT_SKILLS:
            matched.append(s.get("name"))
    return matched[:5]


def build_memo(candidate, jd_fit, rec, final):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})

    num_roles = len(career)
    avg_tenure = sum(c.get("duration_months", 0) for c in career) / max(1, num_roles)

    top_skills = match_relevant_skills(skills)

    # Pick strongest career entry (longest duration at a product company)
    best_career = max(career, key=lambda c: c.get("duration_months", 0)) if career else None
    strongest_career_entry = ""
    if best_career:
        title = best_career.get("title", "Role")
        company = best_career.get("company", "Company")
        dur = best_career.get("duration_months", 0)
        strongest_career_entry = f"{title} at {company} ({dur}mo)"

    yoe = profile.get("years_of_experience", 0)
    yoe_note = None
    if 5 <= yoe <= 9:
        yoe_note = f"{yoe} years experience (5-9 sweet spot for this JD)"
    elif yoe > 9:
        yoe_note = f"{yoe} years experience (senior, slightly above JD range)"

    loc = profile.get("location", "")
    location_note = None
    if any(city in loc.lower() for city in ["pune", "noida"]):
        location_note = f"based in {loc} (JD preferred location)"
    elif any(city in loc.lower() for city in ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurgaon"]):
        location_note = f"based in {loc} (Tier-1 Indian city)"

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
    is_research_only = "research" in all_titles_desc and "production" not in all_titles_desc

    concerns = []
    np_days = signals.get("notice_period_days", 0)
    if np_days > 90:
        concerns.append(f"high notice period ({np_days} days)")
    elif np_days > 60:
        concerns.append(f"notice period {np_days} days")
    if "vector" not in all_titles_desc and not any(
        any(kw in s.get("name", "").lower() for kw in ["vector", "pinecone", "faiss", "weaviate", "qdrant", "milvus"])
        for s in skills
    ):
        concerns.append("no explicit vector DB experience")

    positive_signals = []
    rr = signals.get("recruiter_response_rate", 0)
    if rr > 0.8:
        positive_signals.append(f"recruiter response rate {int(rr * 100)}%")
    icr = signals.get("interview_completion_rate", 0)
    if icr > 0.8:
        positive_signals.append(f"interview completion rate {int(icr * 100)}%")
    if signals.get("open_to_work_flag", False):
        positive_signals.append("actively open to work")

    if is_job_hopper or is_big_tech_lifer:
        yoe_note = None

    current_title = profile.get("current_title", "")
    current_company = profile.get("current_company", "")

    return {
        "candidate_id": candidate.get("candidate_id"),
        "final_score": final,
        "jd_fit_score": jd_fit,
        "recruitability_score": rec,
        "current_title": current_title,
        "current_company": current_company,
        "yoe": yoe,
        "location": loc,
        "country": profile.get("country", ""),
        "top_matched_skills": top_skills,
        "strongest_career_entry": strongest_career_entry,
        "career_summary": f"{num_roles} roles, avg tenure {int(avg_tenure)}mo",
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

    for candidate in iter_candidates(input_path):
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
            reranked = []
            for i in range(len(top_1000_sorted)):
                reranked.append((float(preds[i]), top_1000_sorted[i][1], top_1000_sorted[i][2]))
            reranked.sort(key=lambda x: (-x[0], x[1]))
            top_1000_sorted = reranked
        except Exception:
            pass  # Fall back to heuristic scoring

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
