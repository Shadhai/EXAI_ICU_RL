import json
import sys


def grade_episode(episode_data: dict) -> float:
    survival = episode_data.get("survival", 0.5)
    final_score = episode_data.get("final_score", 0.0)
    task = episode_data.get("task", "easy")
    early_death = episode_data.get("early_death", False)

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


# Evaluator may call grade() directly
def grade(episode_data: dict) -> float:
    return grade_episode(episode_data)


if __name__ == "__main__":
    try:
        if len(sys.argv) > 1:
            with open(sys.argv[1], "r") as f:
                data = json.load(f)
        else:
            data = json.load(sys.stdin)
        result = grade_episode(data)
        print(result)
    except Exception:
        print(0.5)
