# doctor_experience.py
import json
import os
from typing import Dict, List, Tuple

class DoctorExperience:
    def __init__(self, doctor_id: str, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.doctor_id = doctor_id
        # Beta prior for each condition (alpha = successes, beta = failures)
        self.priors = {
            "hypotension": {"alpha": prior_alpha, "beta": prior_beta},
            "sepsis": {"alpha": prior_alpha, "beta": prior_beta},
            "respiratory_failure": {"alpha": prior_alpha, "beta": prior_beta},
            "neurological": {"alpha": prior_alpha, "beta": prior_beta},
        }
        self.history: List[Tuple[str, float]] = []  # (condition, improvement)

    def expected_efficacy(self, condition: str) -> float:
        """Return mean of Beta distribution = alpha/(alpha+beta)"""
        a = self.priors[condition]["alpha"]
        b = self.priors[condition]["beta"]
        return a / (a + b) if (a + b) > 0 else 0.5

    def update(self, condition: str, improvement: float):
        """
        improvement: 0..1, e.g., (actual_survival_gain / max_possible_gain)
        Convert to pseudo‑counts and add to alpha/beta.
        """
        success = max(0.0, min(1.0, improvement))
        self.priors[condition]["alpha"] += success
        self.priors[condition]["beta"] += (1 - success)
        self.history.append((condition, improvement))

    def save(self, filepath: str):
        data = {
            "doctor_id": self.doctor_id,
            "priors": self.priors,
            "history": self.history
        }
        with open(filepath, "w") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, filepath: str):
        with open(filepath, "r") as f:
            data = json.load(f)
        exp = cls(data["doctor_id"])
        exp.priors = data["priors"]
        exp.history = data["history"]
        return exp