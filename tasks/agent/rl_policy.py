import os
import numpy as np
from typing import Tuple, Dict, Any
from stable_baselines3 import PPO

class RLPolicy:
    def __init__(self, model_path="models/ppo_icu_final.zip"):
        self.model = None
        if os.path.exists(model_path):
            self.model = PPO.load(model_path)
            self.use_rl = True
        else:
            self.use_rl = False
            print("RL model not found, falling back to heuristic.")

    def _get_obs(self, state: Dict[str, Any]) -> np.ndarray:
        v = state["vitals"]
        l = state["labs"]
        r = state["resources"]
        obs = np.array([
            v["HR"], v["BP"], v["SpO2"], v["Temp"], v["RR"], v["GCS"],
            l["lactate"], l["pH"], l["WBC"],
            r["icu_beds"], r["ventilators"], r["fio2"],
            state["qSOFA"], state["survival_probability"]
        ], dtype=np.float32)
        return obs

    def decide(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        if self.use_rl:
            obs = self._get_obs(state)
            action_int, _ = self.model.predict(obs, deterministic=True)
            action_map = {0: "administer_drug", 1: "adjust_ventilator", 2: "request_lab", 3: "escalate_care"}
            action = action_map[action_int]
            # Generate explanation from XAI module
            from agent.xai_module import ExplainabilityModule
            xai = ExplainabilityModule()
            explanation = xai.explain(state, action)
            return action, explanation
        else:
            # Fallback to heuristic policy
            from agent.policy import PolicyAgent
            return PolicyAgent().decide(state)