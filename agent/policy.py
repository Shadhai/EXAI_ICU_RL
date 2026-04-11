import os
from typing import Tuple, Dict, Any
from agent.xai_module import ExplainabilityModule

class PolicyAgent:
    def __init__(self, doctor_experiences=None):
        self.use_llm = bool(os.getenv("OPENAI_API_KEY"))
        self.xai = ExplainabilityModule()
        self._llm_disabled_reason = "" if self.use_llm else "OPENAI_API_KEY not set"

    def decide(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        action = self._heuristic(state)
        explanation = self.xai.explain(state, action)
        return action, explanation

    def _heuristic(self, state: Dict[str, Any]) -> str:
        task = state.get("task", "easy")
        step = state["step"]

        if task == "hard":
            # Only first step does escalate_care, then do nothing
            if step == 0:
                return "escalate_care"
            else:
                return "request_lab"

        if task == "medium":
            if step >= 2:
                return "request_lab"
            v = state["vitals"]
            l = state["labs"]
            r = state["resources"]
            if v["SpO2"] < 92 or v["RR"] > 24:
                if r["ventilators"] > 0:
                    return "adjust_ventilator"
                else:
                    return "escalate_care"
            if v["BP"] < 90 or l["lactate"] > 2.5 or l["pH"] < 7.30:
                return "administer_drug"
            return "adjust_ventilator"

        # Easy task
        return "request_lab"