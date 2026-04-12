def grade_episode(episode_data: dict) -> float:
    try:
        survival = float(episode_data.get("survival") or 0.5)
        final_score = float(episode_data.get("final_score") or 0.0)
        task = str(episode_data.get("task") or "easy")
        early_death = bool(episode_data.get("early_death") or False)

        if survival <= 0.0:
            survival = 0.01
        elif survival >= 1.0:
            survival = 0.99

        score = survival * 0.8
        reward_bonus = min(0.19, final_score / 10.0)
        score += reward_bonus

        if early_death:
            score *= 0.8

        mult = {"easy": 1.0, "medium": 1.05, "hard": 1.1}
        score *= mult.get(task, 1.0)
        
        score = max(0.01, min(0.99, score))
        return float(score)
    except Exception as e:
        import sys
        print(f"[GRADER EXCEPTION]: {e}", file=sys.stderr)
        return 0.5