class RewardEngine:
    @staticmethod
    def compute_reward(state: dict, prev_state: dict, action: str, action_valid: bool) -> float:
        if not state["alive"] or not action_valid:
            return 0.0
        survival_gain = state["survival_probability"] - prev_state["survival_probability"]
        reward = survival_gain * 4.0
        if action == "escalate_care":
            reward += 0.3
        elif action != "request_lab":
            reward += 0.1
        return max(0.0, min(1.0, reward + 0.1))