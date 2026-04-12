def grade_episode(episode_data: dict) -> float:
    survival = float(episode_data.get("survival", 0.5))
    final_score = float(episode_data.get("final_score", 0.0))

    score = survival * 0.7 + min(0.25, final_score / 10)
    return max(0.01, min(0.99, score))


TASK_CONFIG = {
    "name": "easy",
    "description": "Stable ICU patient",
    "grader": grade_episode
}
