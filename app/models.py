from pydantic import BaseModel
from typing import Optional, Dict, Any

class ResetRequest(BaseModel):
    task: str = "easy"
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action: str

class AssignDoctorRequest(BaseModel):
    doctor_id: str

class Vitals(BaseModel):
    HR: float
    BP: float
    SpO2: float
    Temp: float
    RR: float
    GCS: float          # Glasgow Coma Scale 3-15

class Labs(BaseModel):
    lactate: float
    WBC: float
    pH: float

class Resources(BaseModel):
    icu_beds: int
    ventilators: int
    fio2: float         # Fraction of inspired oxygen (0.21-1.0)

class CareTeam(BaseModel):
    doctor_id: str
    doctor_name: str
    specialty: str
    experience_level: str

class StateInfo(BaseModel):
    step: int
    max_steps: int
    alive: bool
    survival_probability: float
    task: str
    difficulty: str
    vitals: Vitals
    labs: Labs
    resources: Resources
    care_team: Optional[CareTeam]
    qSOFA: int

class ResetResponse(BaseModel):
    episode_id: str
    state: StateInfo
    message: str

class StepResponse(BaseModel):
    state: StateInfo
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    state: StateInfo

class AutoStepResponse(BaseModel):
    action: str
    explainability: Dict[str, str]
    state: StateInfo
    reward: float
    done: bool
    info: Dict[str, Any]

class AssignDoctorResponse(BaseModel):
    assigned_doctor: Dict[str, Any]
    state: StateInfo
    message: str