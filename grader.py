"""
Grader function for OpenEnv evaluation.
Returns a score between 0 and 1 based on episode performance.
"""
def grade_episode(episode_data: dict) -> float:
    """
    episode_data contains: final_survival, total_reward, steps_taken, task_difficulty, etc.
    """
    survival = episode_data.get("final_survival", 0)
    total_reward = episode_data.get("total_reward", 0)
    steps = episode_data.get("steps", 30)
    task = episode_data.get("task", "easy")

    # Base score from survival
    score = survival

    # Reward bonus (normalized)
    reward_bonus = min(0.2, total_reward / 10)
    score += reward_bonus

    # Penalty if episode ended early due to death
    if episode_data.get("early_death", False):
        score *= 0.5

    # Task difficulty multiplier
    difficulty_mult = {"easy": 1.0, "medium": 1.2, "hard": 1.5}
    score *= difficulty_mult.get(task, 1.0)

    return min(1.0, max(0.0, score))