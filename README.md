# 🛡️ NIDS OpenEnv — Network Intrusion Detection System

> A real-world OpenEnv environment where an AI agent learns to detect, block,
> and escalate network intrusion attempts across three difficulty levels.

---

## Overview

This environment simulates a **Network Intrusion Detection System (NIDS)**. On
each step the agent receives a packet feature vector (IP addresses, ports,
protocol, payload entropy, anomaly score, etc.) and must decide:

| Action | Description |
|---|---|
| `allow` | Permit the packet through the firewall |
| `block` | Drop the packet (treat as malicious) |
| `escalate` | Flag for human review (critical / ambiguous threats) |
| `inspect` | Deep-packet inspection — reveals a hint, costs an inspect token |

---

## Tasks

| Task | Difficulty | Steps | Threat ratio | Description |
|---|---|---|---|---|
| `easy_classification` | Easy | 20 | 50 % | Strong signals, obvious malicious ports |
| `medium_adaptive` | Medium | 30 | 40 % | Burst patterns, manage FP rate |
| `hard_stealth` | Hard | 40 | 35 % | APT camouflage, low anomaly scores on real threats |

---

## Reward Function

Each task is graded **0.0 – 1.0** via F1-score (precision × recall) with
partial progress signals at every step:

```
F1 = 2 × (precision × recall) / (precision + recall)
reward = F1 + escalation_bonus (up to 0.1) + difficulty_bonus
```

Partial credit is given throughout the episode — the agent can see its running
score via the `state` endpoint.

---

## Action / Observation Spaces

### Action
```json
{
  "action_type": "allow | block | escalate | inspect",
  "reason": "optional one-sentence explanation"
}
```

### Observation
```json
{
  "packet": {
    "src_ip": "...", "dst_ip": "...",
    "src_port": 4444, "dst_port": 80,
    "protocol": "TCP",
    "packet_size": 65535,
    "flow_duration": 0.01,
    "bytes_per_second": 6553500.0,
    "flags": ["SYN", "RST"],
    "payload_entropy": 7.8,
    "connection_count": 200,
    "anomaly_score": 0.92
  },
  "step_number": 5,
  "true_positives": 3,
  "false_positives": 1,
  "missed_threats": 0,
  "escalations_used": 1,
  "episode_done": false,
  "message": "✅ Correct BLOCK — threat neutralised."
}
```

---

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Docker

### Install
```bash
pip install -r server/requirements.txt
```

### Run server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run inference
```bash
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_key_here

python inference.py --url http://localhost:8000
```

### Docker
```bash
docker build -t nids-env -f server/Dockerfile .
docker run -p 8000:8000 nids-env
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/tasks` | List tasks |
| POST | `/reset` | Start episode `{"task_name": "..."}` |
| POST | `/step` | Submit action |
| GET | `/state` | Get episode state |
| POST | `/grade` | Run grader |

---

## Project Structure

```
nids_env/
├── __init__.py
├── models.py              # Pydantic Action / Observation / State
├── client.py              # Sync HTTP client
├── inference.py           # ← Competition submission script
├── openenv.yaml           # OpenEnv manifest
├── pyproject.toml
└── server/
    ├── __init__.py
    ├── nids_environment.py  # Core env logic + graders
    ├── app.py               # FastAPI server
    ├── requirements.txt
    └── Dockerfile
```

---

## Why this environment?

Network intrusion detection is a **real-world, high-stakes task** where:
- False positives disrupt legitimate traffic
- Missed threats cause data breaches
- The agent must balance precision and recall under uncertainty

This makes it a rich testbed for RL agents learning multi-objective decision
making with partial observability.
