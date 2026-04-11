# generate_doctor_history.py
import numpy as np
from doctor_experience import DoctorExperience

DOCTOR_IDS = ["dr_aditi", "dr_james", "dr_lin", "dr_samir", "dr_ananya"]
CONDITIONS = ["hypotension", "sepsis", "respiratory_failure", "neurological"]

# Define a “true” underlying efficacy for each doctor (unknown to the AI)
TRUE_EFFICACY = {
    "dr_aditi":   {"hypotension": 0.25, "sepsis": 0.30, "respiratory_failure": 0.20, "neurological": 0.15},
    "dr_james":   {"hypotension": 0.10, "sepsis": 0.15, "respiratory_failure": 0.35, "neurological": 0.05},
    "dr_lin":     {"hypotension": 0.05, "sepsis": 0.25, "respiratory_failure": 0.08, "neurological": 0.10},
    "dr_samir":   {"hypotension": 0.35, "sepsis": 0.10, "respiratory_failure": 0.10, "neurological": 0.05},
    "dr_ananya":  {"hypotension": 0.05, "sepsis": 0.15, "respiratory_failure": 0.10, "neurological": 0.40},
}

def simulate_past_episodes(doctor_id: str, num_episodes: int = 100) -> DoctorExperience:
    exp = DoctorExperience(doctor_id, prior_alpha=1.0, prior_beta=1.0)
    true_eff = TRUE_EFFICACY[doctor_id]

    for _ in range(num_episodes):
        # Randomly choose a condition based on realistic prevalence
        condition = np.random.choice(CONDITIONS, p=[0.3, 0.3, 0.2, 0.2])
        true_eff_val = true_eff[condition]
        # Simulated improvement: drawn from a Beta distribution centered on true efficacy
        alpha_noise = max(0.5, true_eff_val * 10)
        beta_noise = max(0.5, (1 - true_eff_val) * 10)
        improvement = np.random.beta(alpha_noise, beta_noise)
        exp.update(condition, improvement)
    return exp

if __name__ == "__main__":
    for doc_id in DOCTOR_IDS:
        exp = simulate_past_episodes(doc_id, num_episodes=100)
        exp.save(f"doctor_history_{doc_id}.json")
        print(f"Saved history for {doc_id} – efficacy estimates:")
        for cond in CONDITIONS:
            print(f"  {cond}: {exp.expected_efficacy(cond):.2f}")