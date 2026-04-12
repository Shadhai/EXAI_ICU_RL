def grade_episode(episode_data: dict) -> float:
    survival = float(episode_data.get("survival", 0.5))
    instability = float(episode_data.get("instability", 0.5))

    score = survival * 0.6 + (1 - instability) * 0.3
    return max(0.01, min(0.99, score))


TASK_CONFIG = {
    "name": "medium",
    "description": "Moderate ICU instability",
    "grader": grade_episode
}
