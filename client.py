"""
client.py — Lightweight synchronous HTTP client for NIDS OpenEnv.
"""
from __future__ import annotations

import requests
from typing import Optional
from models import NIDSAction, NIDSObservation, NIDSState


class NIDSEnvClient:
    """
    Synchronous client that wraps the NIDS FastAPI server.

    Usage:
        client = NIDSEnvClient("http://localhost:8000")
        obs    = client.reset("easy_classification")
        result = client.step(NIDSAction(action_type="block"))
        state  = client.state()
    """

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def reset(self, task_name: str = "easy_classification") -> dict:
        r = requests.post(f"{self.base_url}/reset",
                          json={"task_name": task_name}, timeout=30)
        r.raise_for_status()
        return r.json()

    def step(self, action: NIDSAction) -> dict:
        r = requests.post(f"{self.base_url}/step",
                          json=action.model_dump(), timeout=30)
        r.raise_for_status()
        return r.json()

    def state(self) -> dict:
        r = requests.get(f"{self.base_url}/state", timeout=30)
        r.raise_for_status()
        return r.json()

    def grade(self, task_name: str) -> dict:
        r = requests.post(f"{self.base_url}/grade",
                          params={"task_name": task_name}, timeout=30)
        r.raise_for_status()
        return r.json()

    def health(self) -> dict:
        r = requests.get(f"{self.base_url}/health", timeout=10)
        r.raise_for_status()
        return r.json()
