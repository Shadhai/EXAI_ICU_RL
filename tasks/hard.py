def grader(episode_data: dict) -> float:
    survival = episode_data.get("final_survival", 0.5)
    final_score = episode_data.get("final_score", 0.0) 
    task = episode_data.get("task", "hard")
    early_death = episode_data.get("early_death", False)
    if survival <= 0.0:
        survival = 0.01
    elif survival >= 1.0:
        survival = 0.99
    score = survival * 0.8
    reward_bonus = min(0.19, final_score / 10)
    score += reward_bonus
    if early_death:
        score *= 0.8
    mult = {"easy": 1.0, "medium": 1.05, "hard": 1.1}
    score *= mult.get(task, 1.0)
    score = max(0.01, min(0.99, score))
    return float(score)

TASK_CONFIG = {
    "name": "hard",
    "description": "ICU crisis, limited resources, GCS ≤12, high FiO2 required",
    "grader": grader
}