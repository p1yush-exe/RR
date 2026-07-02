from datetime import datetime
from constants import (
    REFERENCE_DATE, POSITIVE_TITLES, NEGATIVE_TITLES, 
    RELEVANT_SKILLS, STRONG_DESC_KEYWORDS, SERVICES_COMPANIES, BIG_TECH
)

def jd_fit_score(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    education = candidate.get("education", [])
    
    score = 0.0
    
    # Sub-score 1: Title Match (3.0)
    titles = [profile.get("current_title", "")] + [c.get("title", "") for c in career]
    all_titles_lower = " ".join(t for t in titles if t).lower()
    
    has_neg = any(neg in all_titles_lower for neg in NEGATIVE_TITLES)
    if has_neg:
        skill_has_ai = any(any(kw in s.get("name", "").lower() for kw in POSITIVE_TITLES) for s in skills)
        title_score = -1.0 if skill_has_ai else 0.0
    elif any(pos in all_titles_lower for pos in POSITIVE_TITLES):
        title_score = 1.0
    else:
        title_score = 0.0
    score += title_score * 3.0
    
    # Sub-score 2: YOE Match (2.0)
    yoe = profile.get("years_of_experience", 0)
    if 5 <= yoe <= 9: yoe_score = 1.0
    elif 4 <= yoe < 5 or 9 < yoe <= 12: yoe_score = 0.6
    elif 3 <= yoe < 4 or 12 < yoe <= 15: yoe_score = 0.3
    else: yoe_score = 0.0
    score += yoe_score * 2.0
    
    # Sub-score 3: Career Description Quality (3.0)
    all_descriptions = " ".join(c.get("description", "") for c in career if c.get("description")).lower()
    hit_count = sum(1 for kw in STRONG_DESC_KEYWORDS if kw in all_descriptions)
    desc_score = min(1.0, hit_count / 5.0)
    score += desc_score * 3.0
    
    # Sub-score 4: Product Company Experience (2.0)
    companies = [c.get("company", "").lower() for c in career if c.get("company")]
    has_services = any(any(svc in comp for svc in SERVICES_COMPANIES) for comp in companies)
    all_services = len(companies) > 0 and all(any(svc in comp for svc in SERVICES_COMPANIES) for comp in companies)
    if all_services:
        comp_score = -1.0
    elif has_services:
        comp_score = 0.3
    else:
        comp_score = 1.0
    score += comp_score * 2.0
    
    # Sub-score 5: Career Trajectory (2.0)
    num_roles = len(career)
    avg_tenure = sum(c.get("duration_months", 0) for c in career) / max(1, num_roles)
    has_big_tech_only = num_roles > 0 and all(any(bt in c.get("company", "").lower() for bt in BIG_TECH) for c in career if c.get("company"))
    is_single_long = num_roles == 1 and career[0].get("duration_months", 0) >= 72 if num_roles > 0 else False
    
    if 2 <= num_roles <= 4 and 24 <= avg_tenure <= 48: traj_score = 1.0
    elif num_roles >= 5 and avg_tenure < 18: traj_score = -0.5
    elif is_single_long and has_big_tech_only: traj_score = -0.25
    elif is_single_long: traj_score = 0.0
    else: traj_score = 0.5
    score += traj_score * 2.0
    
    # Sub-score 6: Pre-2022 ML (1.5)
    has_pre_2022 = False
    for c in career:
        sd = c.get("start_date")
        title = c.get("title", "").lower()
        if sd and sd < "2022-01-01" and any(kw in title for kw in POSITIVE_TITLES):
            has_pre_2022 = True
            break
    score += (1.0 if has_pre_2022 else 0.0) * 1.5
    
    # Sub-score 7: Location Match (1.5)
    loc = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing = signals.get("willing_to_relocate", False)
    
    if country != "india" and not willing: loc_score = -0.5
    elif any(city in loc for city in ["pune", "noida"]): loc_score = 1.0
    elif any(city in loc for city in ["hyderabad", "mumbai", "delhi", "bangalore", "bengaluru", "gurgaon", "gurugram", "chennai"]): loc_score = 0.6
    elif country == "india": loc_score = 0.3
    else: loc_score = 0.0
    score += loc_score * 1.5
    
    # Sub-score 8: Relevant Skill Depth (1.5)
    relevant = [s for s in skills if s.get("name", "").lower() in RELEVANT_SKILLS
                and s.get("proficiency", "").lower() in ("advanced", "expert")
                and s.get("duration_months", 0) >= 12]
    skill_score = min(1.0, len(relevant) / 3.0)
    score += skill_score * 1.5
    
    # Sub-score 9: Education Tier (0.5)
    best_tier = min([e.get("tier", "unknown") for e in education if e.get("tier")], default="unknown")
    if best_tier == "tier_1": edu_score = 1.0
    elif best_tier == "tier_2": edu_score = 0.6
    else: edu_score = 0.0
    score += edu_score * 0.5
    
    return score

def recruitability_score(candidate):
    s = candidate.get("redrob_signals", {})
    
    completeness = s.get("profile_completeness_score", 0) / 100.0
    
    last_active = s.get("last_active_date")
    recency = 0.1
    if last_active:
        try:
            la_dt = datetime.strptime(last_active[:10], "%Y-%m-%d").date()
            days_inactive = (REFERENCE_DATE - la_dt).days
            if days_inactive <= 30: recency = 1.0
            elif days_inactive <= 90: recency = 0.8
            elif days_inactive <= 180: recency = 0.5
            elif days_inactive <= 365: recency = 0.2
        except ValueError:
            pass
            
    open_to_work = 1.0 if s.get("open_to_work_flag", False) else 0.6
    views = max(0.05, min(1.0, s.get("profile_views_received_30d", 0) / 100.0))
    
    apps = s.get("applications_submitted_30d", 0)
    if 1 <= apps <= 15: apps_score = 1.0
    elif apps == 0: apps_score = 0.7
    else: apps_score = 0.5
    
    response_rate = max(0.05, s.get("recruiter_response_rate", 0.0))
    
    rt = s.get("avg_response_time_hours", 100)
    if rt <= 6: response_time = 1.0
    elif rt <= 24: response_time = 0.9
    elif rt <= 48: response_time = 0.7
    elif rt <= 72: response_time = 0.5
    else: response_time = 0.3
    
    assessments = s.get("skill_assessment_scores", {})
    if assessments:
        avg_assessment = sum(assessments.values()) / max(1, len(assessments)) / 100.0
    else:
        avg_assessment = 0.5
        
    connections = max(0.05, min(1.0, s.get("connection_count", 0) / 500.0))
    endorsements = max(0.05, min(1.0, s.get("endorsements_received", 0) / 100.0))
    
    np_days = s.get("notice_period_days", 100)
    if np_days <= 30: notice = 1.0
    elif np_days <= 60: notice = 0.8
    elif np_days <= 90: notice = 0.6
    elif np_days <= 120: notice = 0.4
    else: notice = 0.3
    
    sal_range = s.get("expected_salary_range_inr_lpa", {})
    sal_min = sal_range.get("min", 0)
    sal_max = sal_range.get("max", 0)
    sal_mid = (sal_min + sal_max) / 2.0
    if sal_min > sal_max: salary = 0.7
    elif 20 <= sal_mid <= 65: salary = 1.0
    elif sal_mid < 10: salary = 0.4
    else: salary = 0.7
    
    mode = s.get("preferred_work_mode", "").lower()
    if mode in ("hybrid", "flexible"): work_mode = 1.0
    elif mode == "onsite": work_mode = 0.8
    else: work_mode = 0.6
    
    relocate = 1.0 if s.get("willing_to_relocate", False) else 0.7
    
    gh = s.get("github_activity_score", -1)
    if gh == -1: github = 0.5
    elif gh >= 50: github = 1.0
    elif gh >= 30: github = 0.8
    else: github = 0.6
    
    search_app = max(0.05, min(1.0, s.get("search_appearance_30d", 0) / 200.0))
    saved = max(0.05, min(1.0, s.get("saved_by_recruiters_30d", 0) / 20.0))
    interview = max(0.05, s.get("interview_completion_rate", 0.0))
    
    oar = s.get("offer_acceptance_rate", -1)
    if oar == -1: offer_accept = 0.5
    elif oar >= 0.5: offer_accept = 1.0
    elif oar >= 0.3: offer_accept = 0.7
    else: offer_accept = 0.4
    
    email_v = 1.0 if s.get("verified_email", False) else 0.8
    phone_v = 1.0 if s.get("verified_phone", False) else 0.8
    linkedin = 1.0 if s.get("linkedin_connected", False) else 0.7
    
    availability = (recency * open_to_work * notice * work_mode * relocate) ** (1/5)
    responsiveness = (response_rate * response_time * interview * offer_accept) ** (1/4)
    credibility = (completeness * email_v * phone_v * linkedin * github) ** (1/5)
    market_signal = (views * search_app * saved * connections * endorsements * apps_score) ** (1/6)
    
    return availability * responsiveness * credibility * market_signal * salary * avg_assessment
