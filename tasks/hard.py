def grade_episode(episode_data: dict) -> float:
    survival = float(episode_data.get("survival", 0.5))
    final_score = float(episode_data.get("final_score", 0.0))
    instability = float(episode_data.get("instability", 0.5))
    early_death = bool(episode_data.get("early_death", False))

    score = survival * 0.5 + (1 - instability) * 0.3 + min(0.2, final_score / 10.0)

    if early_death:
        score *= 0.9

    score = max(0.01, min(0.99, score))
    return float(score)

TASK_CONFIG = {
    "name": "hard",
    "description": "Critical ICU condition",
    "grader": grade_episode
}
