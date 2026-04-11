from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from agent.policy import PolicyAgent
from app.environment import ICUEnvironment
from app.models import (
    ResetRequest, ResetResponse, StepRequest, StepResponse,
    StateResponse, AutoStepResponse, AssignDoctorRequest, AssignDoctorResponse
)

app = FastAPI(title="ICU Monitor with Clinical Predictors")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

env = ICUEnvironment(seed=42, max_steps=30)
policy_agent = PolicyAgent(doctor_experiences=env.doctor_experiences)
UI_PATH = Path(__file__).resolve().parent / "ui" / "index.html"

@app.get("/")
async def root():
    return FileResponse(UI_PATH)

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    try:
        payload = env.reset(task=request.task, seed=request.seed)
        return ResetResponse(**payload)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/step", response_model=StepResponse)
def step(request: StepRequest):
    try:
        payload = env.step(request.action)
        return StepResponse(**payload)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

@app.get("/state", response_model=StateResponse)
def state():
    try:
        payload = env.state()
        return StateResponse(**payload)
    except RuntimeError as e:
        raise HTTPException(409, str(e))

@app.post("/auto_step", response_model=AutoStepResponse)
def auto_step():
    try:
        current_state = env.state()["state"]
        action, trace = policy_agent.decide(current_state)
        payload = env.step(action)
        payload["info"]["policy"] = {
            "llm_enabled": policy_agent.use_llm,
            "llm_disabled_reason": getattr(policy_agent, "_llm_disabled_reason", ""),
        }
        return AutoStepResponse(
            action=action,
            explainability=trace,
            state=payload["state"],
            reward=payload["reward"],
            done=payload["done"],
            info=payload["info"],
        )
    except RuntimeError as e:
        raise HTTPException(409, str(e))

@app.get("/doctors")
def doctors():
    from app.environment import DOCTOR_REGISTRY
    return {"doctors": list(DOCTOR_REGISTRY.values())}

@app.post("/assign_doctor", response_model=AssignDoctorResponse)
def assign_doctor(request: AssignDoctorRequest):
    try:
        payload = env.assign_doctor(request.doctor_id)
        return AssignDoctorResponse(**payload)
    except (RuntimeError, ValueError) as e:
        raise HTTPException(409 if isinstance(e, RuntimeError) else 400, str(e))