def grade_episode(episode_data: dict) -> float:
    survival = float(episode_data.get("survival", 0.5))
    instability = float(episode_data.get("instability", 0.5))
    final_score = float(episode_data.get("final_score", 0.0))

    score = survival * 0.6 + (1 - instability) * 0.25 + min(0.2, final_score / 10.0)

    score = max(0.01, min(0.99, score))
    return float(score)


TASK_CONFIG = {
    "name": "medium",
    "description": "Moderate ICU instability",
    "grader": grade_episode
}
