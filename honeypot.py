from datetime import datetime

def is_honeypot(candidate):
    profile = candidate.get("profile", {})
    career = candidate.get("career_history", [])
    skills = candidate.get("skills", [])
    signals = candidate.get("redrob_signals", {})
    
    yoe = profile.get("years_of_experience", 0)
    
    # Rule 1: duration_months vs YOE mismatch
    total_months = sum(c.get("duration_months", 0) for c in career)
    if abs(total_months / 12.0 - yoe) > 3.0:
        return True
        
    # Rule 2: expert skills with 0 duration
    expert_zero_count = sum(
        1 for s in skills 
        if s.get("proficiency", "").lower() == "expert" and s.get("duration_months", 0) == 0
    )
    if expert_zero_count >= 5:
        return True
        
    # Rule 3: Any career entry duration_months > actual date diff + 3
    for c in career:
        dur = c.get("duration_months", 0)
        start = c.get("start_date")
        end = c.get("end_date")
        if start and end:
            try:
                start_dt = datetime.strptime(start[:10], "%Y-%m-%d")
                end_dt = datetime.strptime(end[:10], "%Y-%m-%d")
                diff_months = (end_dt.year - start_dt.year) * 12 + (end_dt.month - start_dt.month)
                if dur > diff_months + 3:
                    return True
            except ValueError:
                pass
                
    # Rule 4: Any skill duration_months > YOE * 12 * 1.5
    for s in skills:
        if s.get("duration_months", 0) > yoe * 12 * 1.5:
            return True
            
    # Rule 5: temporal paradox
    signup = candidate.get("signup_date")
    last_active = signals.get("last_active_date")
    if signup and last_active:
        try:
            signup_dt = datetime.strptime(signup[:10], "%Y-%m-%d")
            last_active_dt = datetime.strptime(last_active[:10], "%Y-%m-%d")
            if signup_dt > last_active_dt:
                return True
        except ValueError:
            pass
            
    return False
