import os
from typing import Tuple, Dict, Any
from agent.xai_module import ExplainabilityModule

class PolicyAgent:
    def __init__(self, doctor_experiences=None):
        # Check for LiteLLM proxy environment variables
        self.api_base = os.getenv("API_BASE_URL")
        self.api_key = os.getenv("API_KEY")
        self.use_llm = bool(self.api_base and self.api_key)
        self.xai = ExplainabilityModule()
        self._llm_disabled_reason = ""
        self.client = None

        if self.use_llm:
            try:
                from openai import OpenAI
                self.client = OpenAI(base_url=self.api_base, api_key=self.api_key)
                self._llm_disabled_reason = ""
            except Exception as e:
                self.use_llm = False
                self._llm_disabled_reason = f"Failed to initialise OpenAI client: {e}"
        else:
            self._llm_disabled_reason = "API_BASE_URL or API_KEY not set"

    def decide(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        if self.use_llm and self.client:
            action = self._llm_decision(state)
        else:
            action = self._heuristic(state)
        explanation = self.xai.explain(state, action)
        return action, explanation

    def _llm_decision(self, state: Dict[str, Any]) -> str:
        # Build a prompt from the state
        prompt = f"""You are an ICU AI agent. Based on the following patient state, choose one action: administer_drug, adjust_ventilator, request_lab, escalate_care.

State:
- SpO2: {state['vitals']['SpO2']}%
- BP: {state['vitals']['BP']} mmHg
- Heart rate: {state['vitals']['HR']} bpm
- Respiratory rate: {state['vitals']['RR']} /min
- GCS: {state['vitals']['GCS']}
- Lactate: {state['labs']['lactate']} mmol/L
- pH: {state['labs']['pH']}
- qSOFA: {state.get('qSOFA', 0)}
- Survival probability: {state['survival_probability']:.2f}
- Resources: ICU beds={state['resources']['icu_beds']}, ventilators={state['resources']['ventilators']}

Return only the action name."""
        try:
            response = self.client.chat.completions.create(
                model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )
            action = response.choices[0].message.content.strip().lower()
            if action not in ["administer_drug", "adjust_ventilator", "request_lab", "escalate_care"]:
                action = "request_lab"
            return action
        except Exception:
            # Fallback to heuristic on error
            return self._heuristic(state)

    def _heuristic(self, state: Dict[str, Any]) -> str:
        # Your existing heuristic (same as before)
        v = state["vitals"]
        l = state["labs"]
        r = state["resources"]
        task = state.get("task", "easy")
        step = state["step"]

        if task == "hard":
            if step == 0:
                return "escalate_care"
            else:
                return "request_lab"
        if task == "medium":
            if step >= 2:
                return "request_lab"
            if v["SpO2"] < 92 or v["RR"] > 24:
                if r["ventilators"] > 0:
                    return "adjust_ventilator"
                else:
                    return "escalate_care"
            if v["BP"] < 90 or l["lactate"] > 2.5 or l["pH"] < 7.30:
                return "administer_drug"
            return "adjust_ventilator"
        return "request_lab"
        