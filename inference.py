#!/usr/bin/env python3
import argparse
import os
import sys
import uuid
import time
import requests

# Lazy OpenAI client
_client = None

def get_openai_client():
    global _client
    if _client is not None:
        return _client
    api_base = os.getenv("API_BASE_URL")
    api_key = os.getenv("API_KEY")
    if not api_base or not api_key:
        return None
    try:
        from openai import OpenAI
        _client = OpenAI(base_url=api_base, api_key=api_key)
        return _client
    except Exception:
        return None

def llm_decision(state):
    client = get_openai_client()
    if client is None:
        return heuristic_decision(state)
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
        return heuristic_decision(state)

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
    base_url = os.getenv("API_BASE_URL", "http://localhost:7860")
    episode_id = str(uuid.uuid4())[:8]
    print(f"[START] episode_id={episode_id} task={task} difficulty={task}")

    try:
        resp = requests.post(f"{base_url}/reset", json={"task": task, "seed": seed}, timeout=30)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.text}")
            return
    except Exception as e:
        print(f"Reset error: {e}")
        return
    data = resp.json()

    # Optional doctor assignment
    try:
        assign_resp = requests.post(f"{base_url}/assign_doctor", json={"doctor_id": "dr_aditi"}, timeout=30)
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
        try:
            state_resp = requests.get(f"{base_url}/state", timeout=30)
            if state_resp.status_code != 200:
                break
            current_state = state_resp.json()["state"]
        except Exception:
            break

        action = llm_decision(current_state)

        try:
            step_resp = requests.post(f"{base_url}/step", json={"action": action}, timeout=30)
            if step_resp.status_code != 200:
                break
            step_data = step_resp.json()
            step += 1
            reward = step_data["reward"]
            total_reward += reward
            done = step_data["done"]
            print(f"[STEP] step={step} action={action} reward={reward:.4f}")
            time.sleep(0.05)
        except Exception:
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