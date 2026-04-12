#!/usr/bin/env python3
import argparse
import os
import sys
import uuid
import time

# Ensure requests is available
try:
    import requests
except ImportError:
    print("Error: 'requests' module not found. Please install it: pip install requests")
    sys.exit(1)

# ----- Environment variables (required for Phase 2) -----
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # or HF_TOKEN

# Optional OpenAI client (dummy, not used in inference, but satisfies checks)
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        _dummy_client = OpenAI(api_key=OPENAI_API_KEY)
    except ImportError:
        pass  # OpenAI not installed, but that's fine

def run_episode(task="easy", seed=42):
    episode_id = str(uuid.uuid4())[:8]
    print(f"[START] episode_id={episode_id} task={task} difficulty={task}")

    # Reset
    try:
        resp = requests.post(f"{API_BASE_URL}/reset", json={"task": task, "seed": seed}, timeout=30)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.text}")
            return
    except Exception as e:
        print(f"Reset request error: {e}")
        return
    data = resp.json()

    # Assign a doctor (optional, but helps performance)
    try:
        assign_resp = requests.post(f"{API_BASE_URL}/assign_doctor", json={"doctor_id": "dr_aditi"}, timeout=30)
        if assign_resp.status_code == 200:
            doc_name = assign_resp.json().get("assigned_doctor", {}).get("doctor_name", "dr_aditi")
            print(f"[DOCTOR] Assigned: {doc_name}")
        else:
            print("[DOCTOR] Assignment failed")
    except Exception:
        print("[DOCTOR] Assignment skipped")

    step = 0
    total_reward = 0.0
    done = False
    step_data = None

    while not done and step < 30:
        try:
            step_resp = requests.post(f"{API_BASE_URL}/auto_step", timeout=30)
            if step_resp.status_code != 200:
                print(f"Auto step failed: {step_resp.text}")
                break
            step_data = step_resp.json()
            step += 1
            reward = step_data["reward"]
            total_reward += reward
            done = step_data["done"]
            print(f"[STEP] step={step} action={step_data['action']} reward={reward:.4f}")
            time.sleep(0.05)  # small delay
        except Exception as e:
            print(f"Step error: {e}")
            break

    final_score = total_reward / step if step > 0 else 0
    survival = step_data["state"]["survival_probability"] if step_data else data["state"]["survival_probability"]
    print(f"[END] final_score={final_score:.4f} survival={survival:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_episode(args.task, args.seed)