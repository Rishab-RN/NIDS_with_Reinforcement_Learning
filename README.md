---
title: NIDS OpenEnv
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
pinned: false
---

# NIDS OpenEnv — Network Intrusion Detection System

> A real-world OpenEnv environment where an AI agent learns to detect, block,
> and escalate network intrusion attempts across five difficulty levels.

## Motivation

Security Operations Centers (SOCs) process **11,000+ alerts per day** on average
(Ponemon Institute, 2024), with false positive rates exceeding 40%. Human
analysts suffer from alert fatigue, leading to missed critical threats.

This environment simulates the core NIDS decision loop — an agent receives a
stream of network packet features and must make real-time decisions: **allow**
benign traffic, **block** malicious packets, **escalate** ambiguous threats for
human review, or spend limited **inspect** tokens for deep-packet analysis.

The multi-objective nature (balancing precision and recall under uncertainty)
makes this a rich testbed for RL post-training, directly applicable to
real-world SOC automation.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   inference.py (Agent)                    │
│  LLM-based decision engine + heuristic fallback          │
│  Outputs: [START] / [STEP] / [END] structured logs       │
└──────────────┬───────────────────────────────────────────┘
               │ HTTP (reset / step / state / grade)
┌──────────────▼───────────────────────────────────────────┐
│              server/app.py (FastAPI)                      │
│  OpenEnv-compatible endpoints                            │
│  Pydantic request/response validation                    │
└──────────────┬───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│        server/nids_environment.py (Environment)          │
│  Packet generation, action evaluation, reward shaping    │
│  5 tasks × difficulty-specific traffic simulation        │
└──────────────────────────────────────────────────────────┘
```

## Tasks

| Task | Difficulty | Steps | Threat % | Inspect Tokens | Challenge |
|---|---|---|---|---|---|
| `easy_classification` | Easy | 20 | 50% | 5 | Strong signals, obvious malicious ports |
| `medium_adaptive` | Medium | 30 | 40% | 3 | Burst patterns, manage false-positive rate |
| `hard_stealth` | Hard | 40 | 35% | 2 | APT camouflage, low anomaly scores on threats |
| `expert_zeroday` | Expert | 50 | 30% | 1 | Zero-day: anomaly scores near-identical across classes |
| `adaptive_evasion` | Adaptive | 35 | 40% | 2 | Evasive attacker mimics benign traffic patterns |

## Reward Function

Composite reward with multiple partial-progress signals:

```
F1        = 2 × (precision × recall) / (precision + recall)
streak    = min(consecutive_correct × 0.01, 0.05)
escalate  = min(escalations × 0.05, 0.10)
efficiency = (1 - steps_used / max_steps) × 0.05
difficulty = {easy: 0.0, medium: 0.05, hard: 0.1, expert: 0.15, adaptive: 0.08}

reward = clamp(F1 + streak + escalate + efficiency + difficulty, 0.01, 0.99)
```

Scores are strictly bounded within (0, 1) exclusive.

## Baseline Scores

| Task | Random Agent | Heuristic Agent | LLM Agent (Qwen-72B) |
|---|---|---|---|
| `easy_classification` | ~0.35 | ~0.72 | ~0.85 |
| `medium_adaptive` | ~0.30 | ~0.65 | ~0.78 |
| `hard_stealth` | ~0.22 | ~0.50 | ~0.68 |
| `expert_zeroday` | ~0.18 | ~0.35 | ~0.55 |
| `adaptive_evasion` | ~0.25 | ~0.55 | ~0.70 |

Frontier models should achieve >0.80 on easy and >0.55 on expert.

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
    "src_ip": "10.0.1.42", "dst_ip": "192.168.1.1",
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
  "message": "Correct BLOCK — threat neutralised."
}
```

## Setup & Running Locally

### Prerequisites
- Python 3.10+
- Docker (optional)

### Install
```bash
pip install -e .
```

### Run server
```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Run inference
```bash
export HF_TOKEN=your_key_here
python inference.py --url http://localhost:8000
```

### Docker
```bash
docker build -t nids-env .
docker run -p 8000:8000 nids-env
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/tasks` | List all 5 tasks with descriptions |
| POST | `/reset` | Start episode `{"task_name": "..."}` |
| POST | `/step` | Submit action, receive observation |
| GET | `/state` | Get current episode metadata |
| POST | `/grade` | Compute final score for episode |

## Project Structure

```
nids_env/
├── __init__.py
├── models.py               # Pydantic Action / Observation / State
├── client.py               # Sync HTTP client (EnvClient)
├── inference.py            # Competition submission script
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml
├── Dockerfile
└── server/
    ├── nids_environment.py  # Core env logic + 5 tasks + graders
    ├── app.py               # FastAPI server (OpenEnv endpoints)
    └── requirements.txt
```

## Why This Environment?

Network intrusion detection is a **real-world, high-stakes task** where:
- False positives disrupt legitimate traffic and cause alert fatigue
- Missed threats lead to data breaches costing $4.45M on average (IBM, 2023)
- The agent must balance precision and recall under uncertainty
- Resource constraints (inspect tokens) require strategic decision-making

This makes it a uniquely rich testbed for training RL agents on multi-objective
decision making with partial observability and resource management.
