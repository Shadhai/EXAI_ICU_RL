#!/usr/bin/env python3
import argparse
import os
import sys
import uuid
import time
import requests
from openai import OpenAI

# Environment variables for the proxy (injected by evaluator)
API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# URL of your environment (the Space)
ENV_URL = os.getenv("ENV_URL", "http://localhost:7860")

# Determine if we should use LLM (Phase 2)
USE_LLM = API_BASE_URL and API_KEY
client = None
if USE_LLM:
    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print("[INFO] Using LLM proxy", file=sys.stderr)
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}", file=sys.stderr)
        sys.exit(1)

def llm_decision(state):
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
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=10
    )
    action = response.choices[0].message.content.strip().lower()
    if action not in ["administer_drug", "adjust_ventilator", "request_lab", "escalate_care"]:
        action = "request_lab"
    return action

def heuristic_decision(state):
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

def run_episode(task="easy", seed=42):
    episode_id = str(uuid.uuid4())[:8]
    print(f"[START] episode_id={episode_id} task={task} difficulty={task}")

    # Reset environment
    try:
        resp = requests.post(f"{ENV_URL}/reset", json={"task": task, "seed": seed}, timeout=30)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.text}")
            return
    except Exception as e:
        print(f"Reset error: {e}")
        return
    data = resp.json()

    # Assign a doctor (optional)
    try:
        assign_resp = requests.post(f"{ENV_URL}/assign_doctor", json={"doctor_id": "dr_aditi"}, timeout=30)
        if assign_resp.status_code == 200:
            doc_name = assign_resp.json().get("assigned_doctor", {}).get("doctor_name", "dr_aditi")
            print(f"[DOCTOR] Assigned: {doc_name}")
    except:
        pass

    step = 0
    total_reward = 0.0
    done = False
    step_data = None

    while not done and step < 30:
        # Get current state
        try:
            state_resp = requests.get(f"{ENV_URL}/state", timeout=30)
            if state_resp.status_code != 200:
                print("Failed to get state")
                break
            current_state = state_resp.json()["state"]
        except Exception as e:
            print(f"State error: {e}")
            break

        # Decide action (LLM if in Phase 2, else heuristic)
        if USE_LLM:
            try:
                action = llm_decision(current_state)
            except Exception as e:
                print(f"LLM error: {e}, using heuristic", file=sys.stderr)
                action = heuristic_decision(current_state)
        else:
            action = heuristic_decision(current_state)

        # Execute step with the chosen action
        try:
            step_resp = requests.post(f"{ENV_URL}/step", json={"action": action}, timeout=30)
            if step_resp.status_code != 200:
                print(f"Step failed: {step_resp.text}")
                break
            step_data = step_resp.json()
            step += 1
            reward = step_data["reward"]
            total_reward += reward
            done = step_data["done"]
            print(f"[STEP] step={step} action={action} reward={reward:.4f}")
            time.sleep(0.05)
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
    