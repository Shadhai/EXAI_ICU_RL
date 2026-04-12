#!/usr/bin/env python3
import argparse
import os
import sys
import uuid
import time
import requests

# Try to import OpenAI
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Static config
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

print("[INFO] inference.py loaded. OPENAI_AVAILABLE=%s" % OPENAI_AVAILABLE, file=sys.stderr)


def get_llm_client():
    """Build OpenAI client from env vars at call-time (not import-time)."""
    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY")
    if not api_base or not api_key or not OPENAI_AVAILABLE:
        return None
    try:
        return OpenAI(base_url=api_base, api_key=api_key)
    except Exception as e:
        print(f"[WARN] Failed to init LLM client: {e}", file=sys.stderr)
        return None


def llm_decision(state):
    """Call the hackathon LLM proxy and return an action."""
    client = get_llm_client()
    if client is None:
        print("[WARN] llm_decision: no client, falling back to heuristic", file=sys.stderr)
        return heuristic_decision(state)

    model = os.environ.get("MODEL_NAME", MODEL_NAME)
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
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10
        )
        action = response.choices[0].message.content.strip().lower()
        print(f"[LLM] action={action!r}", file=sys.stderr)
        if action not in ["administer_drug", "adjust_ventilator", "request_lab", "escalate_care"]:
            action = "request_lab"
        return action
    except Exception as e:
        print(f"[WARN] LLM call failed: {e}. Falling back to heuristic.", file=sys.stderr)
        return heuristic_decision(state)


def heuristic_decision(state):
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


def run_episode(task="easy", seed=42):
    episode_id = str(uuid.uuid4())[:8]
    print(f"[START] episode_id={episode_id} task={task} difficulty={task}")

    # Determine mode at RUNTIME, not import time
    api_base = os.environ.get("API_BASE_URL")
    api_key  = os.environ.get("API_KEY")
    use_llm  = bool(api_base and api_key and OPENAI_AVAILABLE)
    print(f"[INFO] use_llm={use_llm} API_BASE_URL={'SET' if api_base else 'NOT SET'}", file=sys.stderr)

    # Reset environment
    try:
        resp = requests.post(f"{ENV_BASE_URL}/reset", json={"task": task, "seed": seed}, timeout=30)
        if resp.status_code != 200:
            print(f"Reset failed: {resp.text}")
            return
    except Exception as e:
        print(f"Reset error: {e}")
        return
    data = resp.json()

    # Optional: assign a doctor
    try:
        assign_resp = requests.post(
            f"{ENV_BASE_URL}/assign_doctor", json={"doctor_id": "dr_aditi"}, timeout=30
        )
        if assign_resp.status_code == 200:
            doc_name = assign_resp.json().get("assigned_doctor", {}).get("doctor_name", "dr_aditi")
            print(f"[DOCTOR] Assigned: {doc_name}")
    except Exception:
        pass

    step = 0
    total_reward = 0.0
    done = False
    step_data = None

    while not done and step < 30:
        # Get current state
        try:
            state_resp = requests.get(f"{ENV_BASE_URL}/state", timeout=30)
            if state_resp.status_code != 200:
                print("Failed to get state")
                break
            current_state = state_resp.json()["state"]
        except Exception as e:
            print(f"State error: {e}")
            break

        # Inject step/task context for heuristic fallback
        current_state["step"] = step
        current_state["task"] = task

        # Choose action
        if use_llm:
            action = llm_decision(current_state)
        else:
            action = heuristic_decision(current_state)

        # Execute action
        try:
            step_resp = requests.post(f"{ENV_BASE_URL}/step", json={"action": action}, timeout=30)
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
    survival = (
        step_data["state"]["survival_probability"]
        if step_data
        else data["state"]["survival_probability"]
    )
    print(f"[END] final_score={final_score:.4f} survival={survival:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="easy", choices=["easy", "medium", "hard"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_episode(args.task, args.seed)
