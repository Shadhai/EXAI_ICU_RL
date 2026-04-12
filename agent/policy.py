import os
from typing import Tuple, Dict, Any
from agent.xai_module import ExplainabilityModule


class PolicyAgent:
    def __init__(self, doctor_experiences=None):
        self.xai = ExplainabilityModule()
        # NOTE: Do NOT read env vars or build the OpenAI client here.
        # The evaluator injects API_BASE_URL / API_KEY AFTER the server
        # process starts, so module-level initialisation always sees them
        # as empty.  We resolve them lazily inside decide() instead.

    def _get_client(self):
        """Build an OpenAI client from env vars at call-time, not import-time."""
        api_base = os.environ.get("API_BASE_URL")
        api_key  = os.environ.get("API_KEY")
        if not api_base or not api_key:
            return None
        try:
            from openai import OpenAI
            return OpenAI(base_url=api_base, api_key=api_key)
        except Exception as e:
            print(f"[WARN] PolicyAgent: failed to init OpenAI client: {e}")
            return None

    def decide(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
        client = self._get_client()
        if client:
            action = self._llm_decision(state, client)
        else:
            action = self._heuristic(state)
        explanation = self.xai.explain(state, action)
        return action, explanation

    def _llm_decision(self, state: Dict[str, Any], client) -> str:
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
            response = client.chat.completions.create(
                model=os.environ.get("MODEL_NAME", "gpt-4o-mini"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=10
            )
            action = response.choices[0].message.content.strip().lower()
            if action not in ["administer_drug", "adjust_ventilator", "request_lab", "escalate_care"]:
                action = "request_lab"
            return action
        except Exception as e:
            print(f"[WARN] LLM call failed: {e}. Falling back to heuristic.")
            return self._heuristic(state)

    def _heuristic(self, state: Dict[str, Any]) -> str:
        v = state["vitals"]
        l = state["labs"]
        r = state["resources"]
        task = state.get("task", "easy")
        step = state.get("step", 0)

        if task == "hard":
            return "escalate_care" if step == 0 else "request_lab"
        if task == "medium":
            if step >= 2:
                return "request_lab"
            if v["SpO2"] < 92 or v["RR"] > 24:
                return "adjust_ventilator" if r["ventilators"] > 0 else "escalate_care"
            if v["BP"] < 90 or l["lactate"] > 2.5 or l["pH"] < 7.30:
                return "administer_drug"
            return "adjust_ventilator"
        return "request_lab"
