# grader.py
import sys

def grade_episode(episode_data: dict) -> float:
    # Print debug info to stderr (visible in validator logs)
    print(f"[GRADER] Received data: {episode_data}", file=sys.stderr)

    # Extract values with safe defaults
    survival = episode_data.get("final_survival", 0.5)
    final_score = episode_data.get("final_score", 0.0)   # average reward
    task = episode_data.get("task", "easy")
    early_death = episode_data.get("early_death", False)

    # Ensure survival is strictly between 0 and 1
    if survival <= 0.0:
        survival = 0.01
    elif survival >= 1.0:
        survival = 0.99

    # Base score from survival (max 0.8)
    score = survival * 0.8

    # Reward bonus from final_score (max 0.19)
    reward_bonus = min(0.19, final_score / 10)
    score += reward_bonus

    # Penalty for early death
    if early_death:
        score *= 0.8

    # Task difficulty multiplier
    mult = {"easy": 1.0, "medium": 1.05, "hard": 1.1}
    score *= mult.get(task, 1.0)

    # Final clamp to ensure score is strictly between 0 and 1
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    print(f"[GRADER] Calculated score: {score}", file=sys.stderr)
    return score