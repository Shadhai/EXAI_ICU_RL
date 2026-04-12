def grade_episode(episode_data: dict) -> float:
    survival = float(episode_data.get("survival", 0.5))
    instability = float(episode_data.get("instability", 0.6))

    score = survival * 0.5 + (1 - instability) * 0.4
    return max(0.01, min(0.99, score))


TASK_CONFIG = {
    "name": "hard",
    "description": "Critical ICU condition",
    "grader": grade_episode
}
