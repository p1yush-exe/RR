def generate_reasoning(memo, rank):
    parts = []

    # 1. Lead with career
    if memo.get("strongest_career_entry"):
        parts.append(memo["strongest_career_entry"])

    # 2. Matched skills (max 3, from actual profile)
    if memo.get("top_matched_skills"):
        parts.append(f"production experience in {', '.join(memo['top_matched_skills'][:3])}")

    # 3. YOE (only if meaningful)
    if memo.get("yoe_note"):
        parts.append(memo["yoe_note"])

    # 4. Location (only if positive)
    if memo.get("location_note"):
        parts.append(memo["location_note"])

    # 5. Behavioral positives (top 30 only)
    if rank <= 30 and memo.get("positive_signals"):
        parts.append(memo["positive_signals"][0])

    # 6. Concerns (rank 50+ only)
    if rank >= 50 and memo.get("concerns"):
        parts.append(f"concern: {memo['concerns'][0]}")

    # 7. Research flag (rank 40+ only)
    if memo.get("is_research_only") and rank >= 40:
        parts.append("career leans research-focused")

    # 8. Job hopper flag
    if memo.get("is_job_hopper") and rank >= 40:
        parts.append("frequent role changes")

    # If nothing was added
    if not parts:
        return "Profile fits general requirements."

    return "; ".join(parts) + "."
