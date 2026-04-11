import numpy as np
from typing import Dict, Any, Optional

class StateManager:
    def __init__(self, seed: Optional[int] = None, max_steps: int = 30):
        self.rng = np.random.RandomState(seed)
        self.max_steps = max_steps
        self.reset()

    def reset(self, task: str = "easy", difficulty: str = "easy", task_config: Optional[Dict] = None):
        self.step = 0
        self.alive = True
        self.task = task
        self.difficulty = difficulty

        # Normal values
        self.vitals = {"HR": 75.0, "BP": 120.0, "SpO2": 96.0, "Temp": 37.0, "RR": 16.0, "GCS": 15.0}
        self.labs = {"lactate": 1.5, "WBC": 8.0, "pH": 7.40}
        self.resources = {"icu_beds": 5, "ventilators": 3, "fio2": 0.21}
        self.care_team = None

        if task == "hard":
            self.vitals["BP"] = 85.0
            self.vitals["SpO2"] = 88.0
            self.vitals["Temp"] = 38.5
            self.vitals["RR"] = 28.0
            self.vitals["GCS"] = 12.0
            self.labs["lactate"] = 4.0
            self.labs["pH"] = 7.20
            self.resources["icu_beds"] = 1
            self.resources["ventilators"] = 1
            self.resources["fio2"] = 0.6
        elif task == "medium":
            self.vitals["BP"] = 95.0
            self.vitals["SpO2"] = 91.0
            self.vitals["RR"] = 22.0
            self.vitals["GCS"] = 14
            self.labs["lactate"] = 2.5
            self.labs["pH"] = 7.30
            self.resources["fio2"] = 0.4

        self._update_survival_from_labs()

    def _compute_qSOFA(self) -> int:
        score = 0
        if self.vitals["RR"] >= 22: score += 1
        if self.vitals["BP"] <= 100: score += 1
        if self.vitals["GCS"] < 15: score += 1
        return score

    def _update_survival_from_labs(self):
        spo2 = self.vitals["SpO2"]
        bp = self.vitals["BP"]
        lactate = self.labs["lactate"]
        ph = self.labs["pH"]
        gcs = self.vitals["GCS"]
        rr = self.vitals["RR"]

        survival = 0.95
        if spo2 < 90: survival -= 0.25
        elif spo2 < 94: survival -= 0.10
        if bp < 80: survival -= 0.30
        elif bp < 100: survival -= 0.15
        if lactate > 4.0: survival -= 0.35
        elif lactate > 2.0: survival -= 0.15
        if ph < 7.25: survival -= 0.30
        elif ph < 7.35: survival -= 0.15
        if gcs < 13: survival -= 0.25
        elif gcs < 15: survival -= 0.10
        if rr > 28: survival -= 0.20
        elif rr > 24: survival -= 0.10

        self.survival_probability = max(0.05, min(0.95, survival))

    def get_state(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "max_steps": self.max_steps,
            "alive": self.alive,
            "survival_probability": self.survival_probability,
            "task": self.task,
            "difficulty": self.difficulty,
            "vitals": self.vitals.copy(),
            "labs": self.labs.copy(),
            "resources": self.resources.copy(),
            "care_team": self.care_team,
            "qSOFA": self._compute_qSOFA(),
        }

    def update_vital(self, key: str, delta: float):
        if key in self.vitals:
            self.vitals[key] += delta
            if key == "HR": self.vitals[key] = max(30, min(180, self.vitals[key]))
            elif key == "BP": self.vitals[key] = max(40, min(200, self.vitals[key]))
            elif key == "SpO2": self.vitals[key] = max(60, min(100, self.vitals[key]))
            elif key == "Temp": self.vitals[key] = max(34, min(42, self.vitals[key]))
            elif key == "RR": self.vitals[key] = max(8, min(40, self.vitals[key]))
            elif key == "GCS": self.vitals[key] = max(3, min(15, self.vitals[key]))

    def update_lab(self, key: str, delta: float):
        if key in self.labs:
            self.labs[key] += delta
            if key == "lactate": self.labs[key] = max(0.5, min(20, self.labs[key]))
            elif key == "pH": self.labs[key] = max(6.8, min(7.6, self.labs[key]))
            elif key == "WBC": self.labs[key] = max(1, min(50, self.labs[key]))

    def update_resource(self, key: str, delta: float):
        if key in self.resources:
            self.resources[key] += delta
            if key == "icu_beds": self.resources[key] = max(0, min(10, self.resources[key]))
            elif key == "ventilators": self.resources[key] = max(0, min(5, self.resources[key]))
            elif key == "fio2": self.resources[key] = max(0.21, min(1.0, self.resources[key]))