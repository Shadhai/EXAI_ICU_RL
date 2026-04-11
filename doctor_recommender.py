# doctor_recommender.py
from typing import Dict, Any
from doctor_experience import DoctorExperience

class DoctorRecommender:
    def __init__(self, doctor_experiences: Dict[str, DoctorExperience]):
        self.doctor_exps = doctor_experiences

    def identify_primary_condition(self, state: Dict[str, Any]) -> str:
        v = state["vitals"]
        l = state["labs"]
        if v["BP"] < 90:
            return "hypotension"
        if l["lactate"] > 2.0 or l["pH"] < 7.35:
            return "sepsis"
        if v["SpO2"] < 92 or v["RR"] > 20:
            return "respiratory_failure"
        if v["GCS"] < 15:
            return "neurological"
        return "sepsis"  # default

    def best_doctor_for_state(self, state: Dict[str, Any]) -> tuple:
        """Returns (doctor_id, expected_improvement) for the best doctor."""
        cond = self.identify_primary_condition(state)
        best_id = None
        best_eff = -1.0
        for doc_id, exp in self.doctor_exps.items():
            eff = exp.expected_efficacy(cond)
            if eff > best_eff:
                best_eff = eff
                best_id = doc_id
        return best_id, best_eff