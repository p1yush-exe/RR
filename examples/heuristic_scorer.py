import json
import re
from datetime import datetime
from collections import defaultdict

# Pre-defined sets for fast O(1) lookups
SERVICES_COMPANIES = {"tcs", "infosys", "wipro", "accenture", "cognizant", "capgemini", "hcl", "tech mahindra"}
AI_KEYWORDS = {"rag", "llm", "embedding", "pinecone", "weaviate", "qdrant", "milvus", "sentence-transformers", "pytorch", "tensorflow"}
AI_TITLES = {"ai engineer", "machine learning", "ml engineer", "search", "retrieval", "data scientist"}
NON_AI_TITLES = {"marketing", "hr", "sales", "manager", "support", "admin"}

def calculate_recruitability(signals):
    """
    Calculate recruitability score between 0 and 1.
    """
    # Base response rate
    response_rate = signals.get("recruiter_response_rate", 0.5)
    
    # Penalize if not open to work
    open_to_work = 1.0 if signals.get("open_to_work_flag", False) else 0.5
    
    # Interview completion penalty (if they ghost interviews)
    interview_completion = signals.get("interview_completion_rate", 1.0)
    
    # Recency of activity (closer to today is better)
    # Assuming 'today' is around mid-2024 for the context of this dataset
    try:
        last_active = datetime.strptime(signals.get("last_active_date", "2023-01-01"), "%Y-%m-%d")
        days_inactive = (datetime(2024, 6, 1) - last_active).days
        recency_multiplier = max(0.1, 1.0 - (max(0, days_inactive) / 365.0))
    except:
        recency_multiplier = 0.5

    score = response_rate * open_to_work * interview_completion * recency_multiplier
    return score

def is_honeypot(candidate):
    """
    Detect logical impossibilities indicating a honeypot.
    Returns True if honeypot, False otherwise.
    """
    profile = candidate.get("profile", {})
    yoe = profile.get("years_of_experience", 0)
    
    # 1. Experience vs Career History mismatch
    career = candidate.get("career_history", [])
    total_months_career = sum(job.get("duration_months", 0) for job in career)
    if total_months_career / 12 > yoe + 3: # Allow some buffer
        return True
        
    # 2. Skill duration > YOE
    skills = candidate.get("skills", [])
    for skill in skills:
        duration_months = skill.get("duration_months", 0)
        # If they claim to have used a skill longer than their total YOE (with a 2-year buffer for internships)
        if duration_months / 12 > yoe + 2:
            return True
            
    # 3. 10+ expert skills but 0 duration
    expert_zero_duration_count = sum(1 for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0)
    if expert_zero_duration_count >= 5:
        return True

    return False

def calculate_jd_fit(candidate):
    """
    Calculate a fast heuristic JD fit score.
    """
    score = 0.0
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    
    # 1. Title matching (Current & Past)
    title_text = (profile.get("current_title", "") + " " + " ".join([c.get("title", "") for c in career])).lower()
    
    if any(ai_title in title_text for ai_title in AI_TITLES):
        score += 3.0
    if any(non_ai_title in title_text for non_ai_title in NON_AI_TITLES):
        score -= 5.0 # Heavy penalty for keyword stuffers
        
    # 2. Services companies penalty
    companies_text = (profile.get("current_company", "") + " " + " ".join([c.get("company", "") for c in career])).lower()
    if any(svc in companies_text for svc in SERVICES_COMPANIES):
        # We don't disqualify, but JD says "only worked at consulting firms" is a red flag
        # We'll penalize it, but allow them to overcome it if they have product company exp too
        score -= 2.0
        
    # 3. YOE Match (Target 5-9)
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9:
        score += 2.0
    elif yoe > 9:
        score += 1.0 # Overqualified is better than underqualified
        
    # 4. Keyword matches in skills & descriptions
    desc_text = " ".join([c.get("description", "") for c in career]).lower()
    skill_text = " ".join([s.get("name", "") for s in skills]).lower()
    combined_text = desc_text + " " + skill_text
    
    keyword_hits = sum(1 for kw in AI_KEYWORDS if kw in combined_text)
    score += (keyword_hits * 0.5)

    return max(0.0, score)

def process_candidates(filepath):
    top_candidates = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        # Check if jsonl or json
        if filepath.endswith('.jsonl'):
            for line in f:
                candidate = json.loads(line)
                evaluate_candidate(candidate, top_candidates)
        else:
            # For sample_candidates.json which is a JSON array
            candidates = json.load(f)
            for candidate in candidates:
                evaluate_candidate(candidate, top_candidates)
                
    # Sort by final score
    top_candidates.sort(key=lambda x: x['final_score'], reverse=True)
    return top_candidates[:1000]

def evaluate_candidate(candidate, top_candidates):
    if is_honeypot(candidate):
        return
        
    recruitability = calculate_recruitability(candidate.get("redrob_signals", {}))
    jd_fit = calculate_jd_fit(candidate)
    
    final_score = recruitability * jd_fit
    
    top_candidates.append({
        "candidate_id": candidate["candidate_id"],
        "recruitability": recruitability,
        "jd_fit": jd_fit,
        "final_score": final_score,
        "name": candidate.get("profile", {}).get("anonymized_name", "")
    })

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else 'sample_candidates.json'
    results = process_candidates(filepath)
    
    print(f"Processed top {len(results)} candidates:")
    for i, res in enumerate(results[:10]):
        print(f"{i+1}. {res['candidate_id']} | Score: {res['final_score']:.2f} (Fit: {res['jd_fit']:.2f}, Recr: {res['recruitability']:.2f})")
