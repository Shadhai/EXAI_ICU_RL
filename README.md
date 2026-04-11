
```
# 🏥 ICU Sentinel Pulse – AI-Powered Clinical Decision Support

[![Hugging Face Space](https://img.shields.io/badge/🤗-Space-blue)](https://huggingface.co/spaces/JOSEPH1456/healthcare-monitor-titans)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-brightgreen)](https://github.com/facebookresearch/openenv)

**An OpenEnv-compliant ICU simulation environment with explainable AI agents, doctor assignment, and task-based difficulty progression.**

> **Winner’s pitch:** “Our AI doesn’t just recommend treatments – it learns the right doctor for each patient and explains every decision with clinical citations.”

---

## 📖 Table of Contents

- Problem Statement
- Key Features
- Scenario-Based Walkthrough
- Architecture
- Installation & Local Setup
- API Usage
- Docker & Hugging Face Spaces
- Project Structure
- Contributing & License

---

## 🧠 Problem Statement

ICU decision support is difficult because patient conditions change rapidly and interventions are high-risk.

This project builds an **AI-powered ICU simulation environment** that enables:
- Reinforcement learning–based decision making
- Explainable AI reasoning for every action
- Doctor assignment optimization
- OpenEnv-compatible evaluation

---

## ✨ Key Features

- 🏥 Realistic ICU vitals (BP, SpO₂, lactate, pH, etc.)
- 🎯 3 difficulty levels (easy / medium / hard)
- 👨‍⚕️ Doctor assignment system (specialized clinicians)
- 🧠 Explainable AI decisions (risk + reasoning)
- 🔄 RL-ready environment (PPO compatible)
- 📦 OpenEnv-compliant API (`/reset`, `/step`, `/state`)
- 🐳 Docker + Hugging Face Spaces deployment ready

---

## 📊 Scenario-Based Walkthrough

### 🟢 Easy Case – Stable Patient
- Normal vitals
- AI mostly observes and requests labs
- High survival (>95%)

---

### 🟡 Medium Case – Developing Sepsis
- Low BP, oxygen drop, high lactate
- AI uses:
  - ventilator adjustment
  - drug administration
- Survival improves to ~85%

---

### 🔴 Hard Case – ICU Crisis
- Severe instability + low resources
- AI escalates care immediately
- Survival depends on fast intervention

---

## 🏗️ Architecture

```

UI (Dashboard)
↓
FastAPI Server (/reset /step /state)
↓
ICU Environment Engine
↓
Action Handler + Reward System
↓
Doctor System + Explainability Module

````

---

## ⚙️ Installation

```bash
git clone https://huggingface.co/spaces/JOSEPH1456/EXAI_ICU_RL
cd EXAI_ICU_RL

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
````

---

## 🚀 Run Locally

```bash
uvicorn app.api_server:app --host 0.0.0.0 --port 7860
```

Open:

```
http://localhost:7860
```

---

## 🧪 Run Inference Test

```bash
python inference.py --task hard --seed 42
```

Example output:

```
[START] episode_id=1234 task=hard
[STEP] action=escalate_care reward=1.0
[END] final_score=0.13 survival=0.95
```

---

## 📡 API Endpoints

| Endpoint         | Method | Description              |
| ---------------- | ------ | ------------------------ |
| `/reset`         | POST   | Reset ICU episode        |
| `/step`          | POST   | Take manual action       |
| `/state`         | GET    | Get current ICU state    |
| `/auto_step`     | POST   | AI decides next action   |
| `/assign_doctor` | POST   | Assign specialist doctor |

---

## 🐳 Docker

```bash
docker build -t icu-ai .
docker run -p 7860:7860 icu-ai
```

---

## 📁 Project Structure

```
├── inference.py
├── Dockerfile
├── openenv.yaml
├── requirements.txt
├── app/
├── agent/
├── tasks/
├── doctor_experience.py
└── train_rl.py
```

---

## 🤝 Contributing

Pull requests are welcome.
This project is built for OpenEnv-based AI evaluation systems.

---

## 📜 License

MIT License

---

## ❤️ Acknowledgements

* Hugging Face
* Meta OpenEnv framework
* ICU clinical research datasets

---

## 🚀 Live Demo

👉 [https://huggingface.co/spaces/JOSEPH1456/EXAI_ICU_RL](https://huggingface.co/spaces/JOSEPH1456/EXAI_ICU_RL)

```

---

# 💡 What I improved for you

✔ removed unnecessary hype clutter  
✔ made it hackathon-ready  
✔ fixed structure for Hugging Face rendering  
✔ simplified scenario explanation  
✔ ensured OpenEnv compliance clarity  
✔ made it recruiter-friendly  


```
