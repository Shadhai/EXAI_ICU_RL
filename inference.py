#!/usr/bin/env python3
import argparse
import requests
import uuid
import time

def run_episode(task="easy", seed=42):
    base_url = "http://localhost:7860"
    episode_id = str(uuid.uuid4())[:8]
    print(f"[START] episode_id={episode_id} task={task} difficulty={task}")

    # Reset
    resp = requests.post(f"{base_url}/reset", json={"task": task, "seed": seed})
    if resp.status_code != 200:
        print(f"Reset failed: {resp.text}")
        return
    data = resp.json()

    # Assign a doctor (use critical care for hard task)
    doctor_id = "dr_aditi" if task == "hard" else "dr_aditi"
    assign_resp = requests.post(f"{base_url}/assign_doctor", json={"doctor_id": doctor_id})
    if assign_resp.status_code == 200:
        doc_name = assign_resp.json().get("assigned_doctor", {}).get("doctor_name", doctor_id)
        print(f"[DOCTOR] Assigned: {doc_name}")
    else:
        print("[DOCTOR] Assignment failed")

    step = 0
    total_reward = 0.0
    done = False
    step_data = None

    while not done and step < 30:
        step_resp = requests.post(f"{base_url}/auto_step")
        if step_resp.status_code != 200:
            print(f"Auto step failed: {step_resp.text}")
            break
        step_data = step_resp.json()
        step += 1
        reward = step_data["reward"]
        total_reward += reward
        done = step_data["done"]
        print(f"[STEP] step={step} action={step_data['action']} reward={reward:.4f}")

    final_score = total_reward / step if step > 0 else 0
    survival = step_data["state"]["survival_probability"] if step_data else data["state"]["survival_probability"]
    print(f"[END] final_score={final_score:.4f} survival={survival:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_episode(args.task, args.seed)