import uuid
from typing import Dict, Any, Optional
from app.state_manager import StateManager
from app.transition_engine import TransitionEngine
from app.reward_engine import RewardEngine
from app.action_handler import ActionHandler
from doctor_experience import DoctorExperience
from doctor_recommender import DoctorRecommender

DOCTOR_REGISTRY = {
    "dr_aditi": {"doctor_id": "dr_aditi", "doctor_name": "Dr. Aditi Sharma", "specialty": "Critical Care", "experience_level": "Senior"},
    "dr_james": {"doctor_id": "dr_james", "doctor_name": "Dr. James Carter", "specialty": "Pulmonology", "experience_level": "Intermediate"},
    "dr_lin": {"doctor_id": "dr_lin", "doctor_name": "Dr. Mei Lin", "specialty": "Infectious Disease", "experience_level": "Junior"},
    "dr_samir": {"doctor_id": "dr_samir", "doctor_name": "Dr. Samir Mehta", "specialty": "Cardiology", "experience_level": "Senior"},
    "dr_ananya": {"doctor_id": "dr_ananya", "doctor_name": "Dr. Ananya Rao", "specialty": "Neurocritical Care", "experience_level": "Intermediate"},
}

class ICUEnvironment:
    def __init__(self, seed: int = 42, max_steps: int = 30):
        self.seed = seed
        self.max_steps = max_steps
        self.state_manager = StateManager(seed=seed, max_steps=max_steps)
        self.transition = TransitionEngine(self.state_manager, self)
        self.reward = RewardEngine()
        self.current_episode_id = None
        self.initial_survival = None
        # Load doctor experiences
        self.doctor_experiences = {}
        for doc_id in DOCTOR_REGISTRY.keys():
            try:
                exp = DoctorExperience.load(f"doctor_history_{doc_id}.json")
            except FileNotFoundError:
                exp = DoctorExperience(doc_id)
            self.doctor_experiences[doc_id] = exp
        self.recommender = DoctorRecommender(self.doctor_experiences)
        self.current_doctor_exp = None

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
        if seed is not None:
            self.state_manager = StateManager(seed=seed, max_steps=self.max_steps)
            self.transition = TransitionEngine(self.state_manager, self)   # critical
        self.state_manager.reset(task=task)
        self.initial_survival = self.state_manager.survival_probability
        self.current_episode_id = str(uuid.uuid4())
        return {
            "episode_id": self.current_episode_id,
            "state": self._serialize_state(),
            "message": f"Reset with task={task}"
        }

    def step(self, action: str) -> Dict[str, Any]:
        if not self.state_manager.alive:
            raise RuntimeError("Episode finished. Call reset first.")
        prev_state = self.state_manager.get_state()
        valid = ActionHandler.is_valid(action)
        if not valid:
            return {
                "state": self._serialize_state(),
                "reward": 0.0,
                "done": self.state_manager.step >= self.state_manager.max_steps or not self.state_manager.alive,
                "info": {"error": "invalid_action", "action_info": {}}
            }
        self.transition.apply_action(action)
        new_state = self.state_manager.get_state()
        reward = self.reward.compute_reward(new_state, prev_state, action, True)
        done = (self.state_manager.step >= self.state_manager.max_steps) or (not self.state_manager.alive)
        if done:
            self._end_episode()
        return {
            "state": self._serialize_state(),
            "reward": reward,
            "done": done,
            "info": {"success": True, "action_info": {}}
        }

    def state(self) -> Dict[str, Any]:
        return {"state": self._serialize_state()}

    def assign_doctor(self, doctor_id: str) -> Dict[str, Any]:
        if doctor_id not in DOCTOR_REGISTRY:
            raise ValueError(f"Unknown doctor {doctor_id}")
        self.state_manager.care_team = DOCTOR_REGISTRY[doctor_id].copy()
        self.current_doctor_exp = self.doctor_experiences[doctor_id]
        return {
            "assigned_doctor": self.state_manager.care_team,
            "state": self._serialize_state(),
            "message": f"Assigned {self.state_manager.care_team['doctor_name']}"
        }

    def _end_episode(self):
        if self.current_doctor_exp and self.initial_survival is not None:
            final_survival = self.state_manager.survival_probability
            improvement = max(0.0, final_survival - self.initial_survival)
            state = self.state_manager.get_state()
            cond = self.recommender.identify_primary_condition(state)
            self.current_doctor_exp.update(cond, improvement)
            self.current_doctor_exp.save(f"doctor_history_{self.current_doctor_exp.doctor_id}.json")

    def _serialize_state(self) -> Dict[str, Any]:
        s = self.state_manager.get_state()
        return {
            "step": s["step"],
            "max_steps": s["max_steps"],
            "alive": s["alive"],
            "survival_probability": s["survival_probability"],
            "task": s["task"],
            "difficulty": s["difficulty"],
            "vitals": s["vitals"],
            "labs": s["labs"],
            "resources": s["resources"],
            "care_team": s.get("care_team"),
            "qSOFA": s["qSOFA"],
        }