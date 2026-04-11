# test_env_direct.py
from app.environment import ICUEnvironment

env = ICUEnvironment(seed=42, max_steps=30)
print("=== RESETTING HARD TASK ===")
reset_data = env.reset(task="hard", seed=42)
state = reset_data["state"]
print(f"Initial state: SpO2={state['vitals']['SpO2']}, BP={state['vitals']['BP']}, lactate={state['labs']['lactate']}, pH={state['labs']['pH']}, GCS={state['vitals']['GCS']}, survival={state['survival_probability']:.2f}")

# Take a sequence of actions
actions = ["escalate_care", "administer_drug", "adjust_ventilator"]
for i, action in enumerate(actions):
    result = env.step(action)
    state = result["state"]
    print(f"After {action}: SpO2={state['vitals']['SpO2']:.1f}, BP={state['vitals']['BP']:.1f}, survival={state['survival_probability']:.2f}, reward={result['reward']:.3f}, done={result['done']}")
    if result["done"]:
        break

# Also test what happens if we do nothing (request_lab) repeatedly
print("\n=== DO NOTHING (request_lab) ===")
env.reset(task="hard", seed=42)
for _ in range(5):
    result = env.step("request_lab")
    state = result["state"]
    print(f"Step {state['step']}: survival={state['survival_probability']:.2f}, reward={result['reward']:.3f}")
    if result["done"]:
        break