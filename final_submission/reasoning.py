def generate_reasoning(memo, rank):
    """Generate a fact-based, varied reasoning string from the memo bag.
    Every claim comes directly from the candidate's profile data.
    No invented facts. No hallucination."""
    parts = []

    # 1. Lead with current role context
    ct = memo.get("current_title", "")
    cc = memo.get("current_company", "")
    if ct and cc:
        parts.append(f"Currently {ct} at {cc}")
    elif ct:
        parts.append(f"Currently {ct}")

    # 2. Strongest career entry (longest tenure)
    if memo.get("strongest_career_entry"):
        parts.append(memo["strongest_career_entry"])

    # 3. Career trajectory summary
    if memo.get("career_summary"):
        parts.append(memo["career_summary"])

    # 4. Matched skills (up to 4)
    skills = memo.get("top_matched_skills", [])
    if skills:
        parts.append(f"relevant skills: {', '.join(skills[:4])}")

    # 5. YOE context with JD connection
    if memo.get("yoe_note"):
        parts.append(memo["yoe_note"])

    # 6. Location with JD connection
    if memo.get("location_note"):
        parts.append(memo["location_note"])

    # 7. Production keyword match (connects to JD's "shipper over researcher")
    if memo.get("has_production_keywords") and rank <= 50:
        parts.append("career history mentions production deployment (JD priority)")

    # 8. Behavioral positives (top 40)
    if rank <= 40 and memo.get("positive_signals"):
        parts.extend(memo["positive_signals"][:2])

    # 9. Concerns (rank 40+)
    if rank >= 40 and memo.get("concerns"):
        parts.append(f"concern: {memo['concerns'][0]}")

    # 10. Research flag
    if memo.get("is_research_only"):
        parts.append("career leans research-focused with limited production evidence (JD explicitly flags this)")

    # 11. Job hopper flag
    if memo.get("is_job_hopper"):
        parts.append("frequent role changes — JD penalizes title-chasers switching every 1.5 years")

    # 12. Big tech lifer flag
    if memo.get("is_big_tech_lifer"):
        parts.append("single long tenure at big tech — JD warns this profile may not fit a Series A pace")

    # Fallback
    if not parts:
        return "Profile matches general requirements for the role."

    return "; ".join(parts) + "."
