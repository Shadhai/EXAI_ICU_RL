# grader.py
def grade_episode(episode_data: dict) -> float:
    """
    Grade an episode. Expects:
    - final_survival: float (0-1)
    - total_reward: float
    - task: str (easy, medium, hard)
    - early_death: bool (optional)
    Returns a score strictly between 0 and 1.
    """
    survival = episode_data.get("final_survival", 0.0)
    total_reward = episode_data.get("total_reward", 0.0)
    task = episode_data.get("task", "easy")
    early_death = episode_data.get("early_death", False)

    # Clamp survival to avoid exact 0 or 1
    if survival <= 0.0:
        survival = 0.01
    elif survival >= 1.0:
        survival = 0.99

    # Base score from survival (max 0.8)
    score = survival * 0.8

    # Reward bonus (max 0.19 to avoid reaching 1.0)
    reward_bonus = min(0.19, total_reward / 20)
    score += reward_bonus

    # Penalty for early death
    if early_death:
        score *= 0.8

    # Difficulty multiplier
    mult = {"easy": 1.0, "medium": 1.05, "hard": 1.1}
    score *= mult.get(task, 1.0)

    # Final clamp strictly between 0.01 and 0.99
    if score <= 0.0:
        score = 0.01
    elif score >= 1.0:
        score = 0.99

    return score