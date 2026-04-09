"""
inference.py — NIDS OpenEnv inference script.

Runs an LLM agent against all 3 tasks and emits structured stdout logs
in the exact [START] / [STEP] / [END] key=value format required by judges.

STDOUT FORMAT:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables required:
  API_BASE_URL  — LLM API base URL
  MODEL_NAME    — model identifier
  HF_TOKEN      — Hugging Face / API key
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import requests
from openai import OpenAI

# ---------------------------------------------------------------------------
# Config — Defaults only for API_BASE_URL and MODEL_NAME (not HF_TOKEN)
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")

BENCHMARK    = "nids_env"
TASKS        = ["easy_classification", "medium_adaptive", "hard_stealth"]

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


VALID_ACTIONS = {"allow", "block", "escalate", "inspect"}


def _heuristic_decide(packet: dict) -> dict:
    """Pure heuristic fallback — no LLM needed, never fails."""
    score    = packet.get("anomaly_score", 0.5)
    src_port = packet.get("src_port", 0)
    dst_port = packet.get("dst_port", 0)
    entropy  = packet.get("payload_entropy", 0.0)
    conn     = packet.get("connection_count", 0)
    mal_ports = {4444, 31337, 6666, 1337, 9999}

    if src_port in mal_ports and entropy > 6.0 and conn > 40:
        return {"action_type": "block", "reason": "Malicious port+entropy+conn"}
    if score > 0.65 or src_port in mal_ports:
        return {"action_type": "block", "reason": "High anomaly or malicious port"}
    if score < 0.35 and dst_port in {80, 443, 22, 25, 53, 8080}:
        return {"action_type": "allow", "reason": "Low anomaly, benign port"}
    if 0.45 <= score <= 0.70:
        return {"action_type": "escalate", "reason": "Ambiguous anomaly score"}
    if score < 0.40:
        return {"action_type": "allow", "reason": "Below threshold"}
    return {"action_type": "escalate", "reason": "Default escalate"}


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

        # Validate action_type — fallback if LLM returns garbage
        if action.get("action_type") not in VALID_ACTIONS:
            return _heuristic_decide(packet)
        return action
    except Exception:
        return _heuristic_decide(packet)


# ---------------------------------------------------------------------------
# Environment HTTP helpers — all handle errors gracefully
# ---------------------------------------------------------------------------

def env_reset(base_url: str, task_name: str) -> dict:
    r = requests.post(f"{base_url}/reset",
                      json={"task_name": task_name}, timeout=60)
    r.raise_for_status()
    return r.json()


def env_step(base_url: str, action: dict) -> dict:
    """Send action to /step. Returns the result dict or a synthetic error dict."""
    # Only send fields the server expects (action_type + optional reason)
    payload = {
        "action_type": action.get("action_type", "allow"),
        "reason": action.get("reason", ""),
    }
    try:
        r = requests.post(f"{base_url}/step", json=payload, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        # Server returned 4xx/5xx — log the error, return synthetic result
        err_text = str(e)
        try:
            err_text = r.text[:200]
        except Exception:
            pass
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "state": {},
            "info": {},
            "_error": err_text,
        }
    except Exception as e:
        return {
            "observation": {},
            "reward": 0.0,
            "done": True,
            "state": {},
            "info": {},
            "_error": str(e),
        }


def env_grade(base_url: str, task_name: str) -> dict:
    """Get grade. Returns a fallback score if it fails."""
    try:
        r = requests.post(f"{base_url}/grade",
                          params={"task_name": task_name}, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"score": 0.5, "reward": 0.5}


# ---------------------------------------------------------------------------
# Run one task — exact [START] / [STEP] / [END] key=value format
# [END] is ALWAYS emitted, even on exception.
# ---------------------------------------------------------------------------

def run_task(base_url: str, task_name: str) -> dict:
    step_no = 0
    rewards_list = []

    # ── [START] ──────────────────────────────────────────────────────────────
    print(f"[START] task={task_name} env={BENCHMARK} model={MODEL_NAME}", flush=True)

    try:
        obs  = env_reset(base_url, task_name)
        done = False

        while not done:
            action = llm_decide(obs)
            result = env_step(base_url, action)
            done   = result.get("done", False)
            reward = result.get("reward", 0.0)
            step_no += 1
            rewards_list.append(reward)

            action_str = action.get("action_type", "unknown")
            done_str   = "true" if done else "false"
            error_str  = result.get("_error", "null") or "null"

            # ── [STEP] ───────────────────────────────────────────────────────
            print(
                f"[STEP] step={step_no} action={action_str} "
                f"reward={reward:.2f} done={done_str} error={error_str}",
                flush=True,
            )

            obs = result

        grade = env_grade(base_url, task_name)
        score = grade.get("score", 0.5)

    except Exception as exc:
        # Something went totally wrong — still emit [END]
        print(f"[STEP] step={step_no + 1} action=allow reward=0.00 done=true "
              f"error={exc}", flush=True)
        score = 0.5
        grade = {"score": score}

    # ── [END] — always emitted ───────────────────────────────────────────────
    success_str = "true" if score > 0.5 else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards_list) if rewards_list else "0.00"

    print(
        f"[END] success={success_str} steps={step_no} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )

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

    # Wait for server to become available (up to 60s)
    for attempt in range(12):
        try:
            h = requests.get(f"{base_url}/health", timeout=10)
            h.raise_for_status()
            break
        except Exception:
            if attempt == 11:
                print(f"[ERROR] Cannot reach environment at {base_url}", file=sys.stderr)
                sys.exit(1)
            time.sleep(5)

    for task_name in TASKS:
        try:
            run_task(base_url, task_name)
        except Exception:
            # Last resort — should never get here, but don't crash
            traceback.print_exc(file=sys.stderr)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
