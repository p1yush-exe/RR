import json
import argparse
from pathlib import Path

from honeypot import is_honeypot
from scorer import jd_fit_score, recruitability_score
from rank import iter_candidates

def run_phase1_and_sort(input_path, output_path):
    print(f"Reading {input_path} and running Phase 1 on ALL candidates...")
    
    processed = 0
    scored_candidates = []
    
    for candidate in iter_candidates(input_path):
        processed += 1
        
        # 1. Check honeypot
        if is_honeypot(candidate):
            continue
            
        # 2. Check research-only constraint
        career = candidate.get("career_history", [])
        all_titles_desc = " ".join(
            [c.get("title", "") for c in career] + [c.get("description", "") for c in career]
        ).lower()
        if "research" in all_titles_desc and "production" not in all_titles_desc:
            continue
            
        jd_fit = jd_fit_score(candidate)
        rec = recruitability_score(candidate)
        final_score = jd_fit * rec
        
        candidate["_phase1_scores"] = {
            "jd_fit_score": jd_fit,
            "recruitability_score": rec,
            "final_score": final_score
        }
        
        scored_candidates.append(candidate)
        
        if processed % 10000 == 0:
            print(f"Processed {processed} candidates...")

    print("Phase 1 filtering complete. Sorting valid candidates...")
    
    # Python's built-in Timsort (list.sort) is implemented in C and is mathematically
    # the fastest general-purpose sorting algorithm available in Python (O(N log N)), 
    # vastly outperforming any manual insertion sort or heap sort written in pure Python.
    scored_candidates.sort(key=lambda c: c["_phase1_scores"]["final_score"], reverse=True)
    
    print(f"Writing sorted candidates to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for c in scored_candidates:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
            
    print(f"Done! Saved {len(scored_candidates)} sorted candidates.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="candidates.jsonl")
    parser.add_argument("--out", default="candidatesphase2.jsonl")
    args = parser.parse_args()
    
    run_phase1_and_sort(args.candidates, args.out)
