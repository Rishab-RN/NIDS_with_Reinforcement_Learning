"""
inference.py — NIDS OpenEnv inference script.

Runs an LLM agent against all 3 tasks and emits structured stdout logs
in the exact [START] / [STEP] / [END] format required by the judges.

Environment variables required:
  API_BASE_URL  — LLM API base URL  (e.g. https://api.openai.com/v1)
  MODEL_NAME    — model identifier  (e.g. gpt-4o-mini)
  HF_TOKEN      — Hugging Face / API key

Usage:
  python inference.py [--url http://localhost:8000]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

TASKS = ["easy_classification", "medium_adaptive", "hard_stealth"]

# ---------------------------------------------------------------------------
# LLM agent
# ---------------------------------------------------------------------------

client_llm = OpenAI(api_key=HF_TOKEN or "dummy", base_url=API_BASE_URL)

SYSTEM_PROMPT = """
You are an AI network security analyst acting as an agent inside a Network
Intrusion Detection System (NIDS).

On each step you receive a JSON packet observation. You must respond with a
single JSON object containing ONLY these two keys:
  {
    "action_type": "<allow|block|escalate|inspect>",
    "reason": "<one sentence>"
  }

DECISION RULES (apply in order, stop at first match):

1. BLOCK immediately if ALL of these are true:
   - src_port is one of {4444, 31337, 6666, 1337, 9999}
   - payload_entropy > 6.0
   - connection_count > 40

2. BLOCK if anomaly_score > 0.70

3. ESCALATE if anomaly_score between 0.45 and 0.70 AND any of:
   - src_port in malicious list above
   - flags contain RST, URG, or PSH
   - bytes_per_second > 500000

4. ESCALATE if connection_count > 60 AND src_port in malicious list (stealth APT pattern)

5. ALLOW if anomaly_score < 0.35 AND dst_port in {80, 443, 22, 25, 53, 8080}

6. INSPECT if you have tokens left and signals are mixed/contradictory

7. Default: ALLOW if anomaly_score < 0.40, else ESCALATE

IMPORTANT:
- On hard difficulty, real threats are CAMOUFLAGED with lower anomaly scores.
  Use multi-feature reasoning (port + entropy + connection_count + flags together).
- Minimise false positives (F1 scoring — precision and recall matter equally).
- Never output anything except the JSON object. No markdown, no preamble.
""".strip()


def llm_decide(obs_json: dict) -> dict:
    """Ask the LLM what action to take given the current observation."""
    packet = obs_json.get("observation", obs_json).get("packet", {})
    user_msg = json.dumps(packet, indent=2)

    try:
        resp = client_llm.chat.completions.create(
            model      = MODEL_NAME,
            messages   = [
                {"role": "system",  "content": SYSTEM_PROMPT},
                {"role": "user",    "content": user_msg},
            ],
            temperature = 0.1,
            max_tokens  = 128,
        )
        raw = resp.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        action = json.loads(raw.strip())
    except Exception:
        # Fallback: heuristic decision from anomaly score + port
        score     = packet.get("anomaly_score", 0.5)
        src_port  = packet.get("src_port", 0)
        dst_port  = packet.get("dst_port", 0)
        mal_ports = {4444, 31337, 6666, 1337, 9999}
        if score > 0.65 or src_port in mal_ports:
            action = {"action_type": "block",   "reason": "Fallback: high anomaly or malicious port."}
        elif score < 0.35 and dst_port in {80, 443, 22, 25, 53, 8080}:
            action = {"action_type": "allow",   "reason": "Fallback: low anomaly, benign port."}
        else:
            action = {"action_type": "escalate","reason": "Fallback: ambiguous signals."}
    return action


# ---------------------------------------------------------------------------
# Environment HTTP helpers
# ---------------------------------------------------------------------------

def env_reset(base_url: str, task_name: str) -> dict:
    r = requests.post(f"{base_url}/reset",
                      json={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()


def env_step(base_url: str, action: dict) -> dict:
    r = requests.post(f"{base_url}/step", json=action, timeout=30)
    r.raise_for_status()
    return r.json()


def env_grade(base_url: str, task_name: str) -> dict:
    r = requests.post(f"{base_url}/grade",
                      params={"task_name": task_name}, timeout=30)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Run one task episode  — strict [START] / [STEP] / [END] log format
# ---------------------------------------------------------------------------

def _log(tag: str, payload: dict) -> None:
    """Emit a single judge-format log line: [TAG] <json>"""
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


def run_task(base_url: str, task_name: str) -> dict:
    # ── [START] ──────────────────────────────────────────────────────────────
    _log("START", {
        "task":      task_name,
        "model":     MODEL_NAME,
        "timestamp": time.time(),
    })

    obs     = env_reset(base_url, task_name)
    step_no = 0
    done    = False

    while not done:
        action = llm_decide(obs)
        result = env_step(base_url, action)
        done   = result.get("done", False)
        reward = result.get("reward", 0.0)
        step_no += 1

        # ── [STEP] ───────────────────────────────────────────────────────────
        _log("STEP", {
            "task":        task_name,
            "step":        step_no,
            "action_type": action.get("action_type"),
            "reason":      action.get("reason", ""),
            "reward":      reward,
            "done":        done,
            "message":     result.get("observation", {}).get("message", ""),
        })

        obs = result

    grade = env_grade(base_url, task_name)

    # ── [END] ────────────────────────────────────────────────────────────────
    _log("END", {
        "task":           task_name,
        "score":          grade["score"],
        "reward":         grade["reward"],
        "true_positives": grade["true_positives"],
        "false_positives":grade["false_positives"],
        "missed_threats": grade["missed_threats"],
        "grade":          grade["grade"],
        "steps_taken":    step_no,
        "timestamp":      time.time(),
    })

    return grade


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url",
        default=os.environ.get("ENV_URL", "http://localhost:8000"),
        help="Base URL of the running NIDS environment server",
    )
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Verify server is up
    try:
        h = requests.get(f"{base_url}/health", timeout=10)
        h.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Cannot reach environment at {base_url}: {e}", file=sys.stderr)
        sys.exit(1)

    all_scores = []
    for task_name in TASKS:
        grade = run_task(base_url, task_name)
        all_scores.append(grade["score"])
        time.sleep(0.5)

    avg = sum(all_scores) / len(all_scores)
    _log("SUMMARY", {
        "tasks":   TASKS,
        "scores":  all_scores,
        "average": round(avg, 4),
        "model":   MODEL_NAME,
    })


if __name__ == "__main__":
    main()
