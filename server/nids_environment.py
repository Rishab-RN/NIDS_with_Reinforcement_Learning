"""
nids_environment.py — Server-side NIDS Environment implementation.

Three tasks (easy → medium → hard), each with its own grader.
Reward range: 0.0 – 1.0 (partial credit given throughout the episode).
"""
from __future__ import annotations

import random
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from models import (
    ActionType, NIDSAction, NIDSObservation, NIDSState,
    PacketFeatures, PacketProtocol, ThreatLevel,
)


# ---------------------------------------------------------------------------
# Packet generator
# ---------------------------------------------------------------------------

_BENIGN_PORTS   = [80, 443, 22, 25, 53, 8080]
_MALICIOUS_PORTS = [4444, 31337, 6666, 1337, 9999]
_PROTOCOLS      = list(PacketProtocol)


def _random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _make_packet(malicious: bool, difficulty: str) -> Tuple[PacketFeatures, bool]:
    """Generate a synthetic packet. Returns (features, is_malicious)."""

    if malicious:
        src_port      = random.choice(_MALICIOUS_PORTS)
        dst_port      = random.randint(1, 65535)
        protocol      = random.choice(_PROTOCOLS)
        packet_size   = random.randint(1400, 65535)
        flow_duration = random.uniform(0.001, 0.5)
        entropy       = random.uniform(6.5, 8.0)   # high → obfuscated
        conn_count    = random.randint(50, 500)
        anomaly_score = random.uniform(0.6, 1.0)
        flags         = random.sample(["SYN", "FIN", "RST", "PSH", "URG"],
                                      k=random.randint(2, 4))

        # Hard mode: camouflage some signals
        if difficulty == "hard":
            anomaly_score = max(0.0, anomaly_score - random.uniform(0.2, 0.4))
            entropy       = max(0.0, entropy - random.uniform(1.0, 2.0))
            conn_count    = max(1, conn_count - random.randint(30, 45))
    else:
        src_port      = random.randint(1024, 65535)
        dst_port      = random.choice(_BENIGN_PORTS)
        protocol      = random.choice([PacketProtocol.TCP, PacketProtocol.HTTP,
                                       PacketProtocol.DNS])
        packet_size   = random.randint(64, 1400)
        flow_duration = random.uniform(0.1, 10.0)
        entropy       = random.uniform(3.0, 5.5)
        conn_count    = random.randint(1, 20)
        anomaly_score = random.uniform(0.0, 0.35)
        flags         = random.sample(["SYN", "ACK"], k=random.randint(1, 2))

        # Hard mode: add a little noise to benign packets
        if difficulty == "hard":
            anomaly_score = min(1.0, anomaly_score + random.uniform(0.1, 0.2))

    return PacketFeatures(
        src_ip           = _random_ip(),
        dst_ip           = _random_ip(),
        src_port         = src_port,
        dst_port         = dst_port,
        protocol         = protocol,
        packet_size      = packet_size,
        flow_duration    = flow_duration,
        bytes_per_second = packet_size / max(flow_duration, 1e-6),
        flags            = flags,
        payload_entropy  = round(entropy, 3),
        connection_count = conn_count,
        anomaly_score    = round(anomaly_score, 3),
    ), malicious


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

TASKS: Dict[str, dict] = {
    "easy_classification": {
        "difficulty":       "easy",
        "max_steps":        20,
        "threat_ratio":     0.5,   # 50 % of packets are malicious
        "inspect_tokens":   5,     # free deep-inspections per episode
        "description": (
            "Binary classification: identify malicious vs benign packets. "
            "Signals are strong — high anomaly scores, obvious ports."
        ),
    },
    "medium_adaptive": {
        "difficulty":       "medium",
        "max_steps":        30,
        "threat_ratio":     0.4,
        "inspect_tokens":   3,
        "description": (
            "Adaptive attacker: threats arrive in bursts. Agent must learn "
            "temporal patterns and manage false-positive rate."
        ),
    },
    "hard_stealth": {
        "difficulty":       "hard",
        "max_steps":        40,
        "threat_ratio":     0.35,
        "inspect_tokens":   2,
        "description": (
            "Stealth APT simulation: attackers camouflage traffic. Low anomaly "
            "scores on real threats. Agent must use multi-feature reasoning."
        ),
    },
}


# ---------------------------------------------------------------------------
# Graders
# ---------------------------------------------------------------------------

def _compute_reward(
    tp: int, fp: int, missed: int, escalations: int,
    max_steps: int, difficulty: str,
) -> float:
    """
    Reward function with partial progress signals.

    Components:
      - Precision-like:  penalise false positives
      - Recall-like:     penalise missed threats
      - Escalation bonus: appropriate use of escalate
      - Normalised to [0, 1]
    """
    total_decisions = max(tp + fp + missed, 1)

    precision_score = tp / max(tp + fp, 1)          # 0–1
    recall_score    = tp / max(tp + missed, 1)       # 0–1
    f1              = (2 * precision_score * recall_score /
                       max(precision_score + recall_score, 1e-9))

    # Escalation bonus (capped — reward appropriate use, not spamming)
    escalation_bonus = min(escalations * 0.05, 0.1)

    # Difficulty multiplier (harder tasks → same raw score = higher reward)
    diff_bonus = {"easy": 0.0, "medium": 0.05, "hard": 0.1}[difficulty]

    raw = f1 + escalation_bonus + diff_bonus
    # Clamp strictly within (0, 1) — validator rejects exactly 0.0 and 1.0
    clamped = max(0.0001, min(raw, 0.9999))
    return round(clamped, 4)


def grade_task(task_name: str, episode_data: dict) -> dict:
    """
    Public grader called at episode end.
    Returns a dict with score (0.0–1.0) and breakdown.
    """
    task   = TASKS[task_name]
    diff   = task["difficulty"]
    tp     = episode_data["true_positives"]
    fp     = episode_data["false_positives"]
    missed = episode_data["missed_threats"]
    esc    = episode_data["escalations_used"]
    steps  = episode_data["step_count"]

    score = _compute_reward(tp, fp, missed, esc, task["max_steps"], diff)

    return {
        "task":             task_name,
        "difficulty":       diff,
        "score":            score,
        "reward":           score,
        "true_positives":   tp,
        "false_positives":  fp,
        "missed_threats":   missed,
        "escalations_used": esc,
        "steps_taken":      steps,
        "grade": (
            "EXCELLENT" if score >= 0.85 else
            "GOOD"      if score >= 0.65 else
            "FAIR"      if score >= 0.45 else
            "POOR"
        ),
    }


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class NIDSEnvironment:
    """
    Network Intrusion Detection OpenEnv environment.
    Compatible with OpenEnv step() / reset() / state() contract.
    """

    def __init__(self, task_name: str = "easy_classification"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS)}")
        self.task_name  = task_name
        # Deep-copy so per-episode mutations (e.g. inspect max_steps bump) never
        # corrupt the global TASKS registry.
        self.task_cfg   = dict(TASKS[task_name])
        self._reset_internals()

    # ------------------------------------------------------------------
    def _reset_internals(self) -> None:
        self.episode_id      = str(uuid.uuid4())[:8]
        self.step_count      = 0
        self.true_positives  = 0
        self.false_positives = 0
        self.missed_threats  = 0
        self.escalations     = 0
        self.inspect_tokens  = self.task_cfg["inspect_tokens"]
        self._packet_queue: List[Tuple[PacketFeatures, bool]] = []
        self._pre_generate_episode()

    def _pre_generate_episode(self) -> None:
        """Generate the full packet sequence for this episode."""
        max_steps    = self.task_cfg["max_steps"]
        threat_ratio = self.task_cfg["threat_ratio"]
        difficulty   = self.task_cfg["difficulty"]
        rng          = random.Random()

        for _ in range(max_steps):
            is_malicious = rng.random() < threat_ratio
            pkt, label   = _make_packet(is_malicious, difficulty)
            self._packet_queue.append((pkt, label))

    # ------------------------------------------------------------------
    def reset(self) -> NIDSObservation:
        self._reset_internals()
        pkt, _ = self._packet_queue[0]
        return NIDSObservation(
            packet           = pkt,
            step_number      = 0,
            packets_seen     = 0,
            true_positives   = 0,
            false_positives  = 0,
            missed_threats   = 0,
            escalations_used = 0,
            episode_done     = False,
            message          = (
                f"Episode started. Task: {self.task_name} | "
                f"Difficulty: {self.task_cfg['difficulty']} | "
                f"Max steps: {self.task_cfg['max_steps']}"
            ),
        )

    # ------------------------------------------------------------------
    def step(self, action: NIDSAction) -> NIDSObservation:
        if self.step_count >= self.task_cfg["max_steps"]:
            return self._terminal_obs("Episode already finished.")

        pkt, is_malicious = self._packet_queue[self.step_count]
        self.step_count  += 1

        # --- evaluate action ---
        msg_parts = []
        if action.action_type == ActionType.INSPECT:
            if self.inspect_tokens > 0:
                self.inspect_tokens -= 1
                # Reveal true label hint after inspection
                hint = "MALICIOUS" if is_malicious else "BENIGN"
                msg_parts.append(f"[INSPECT] Deep scan hint: likely {hint}.")
                # Inspection doesn't count as a blocking decision — re-queue
                self._packet_queue.insert(self.step_count, (pkt, is_malicious))
                self.task_cfg["max_steps"] += 1  # compensate extra step
            else:
                msg_parts.append("[INSPECT] No tokens left. Packet auto-allowed.")
                if is_malicious:
                    self.missed_threats += 1
        elif action.action_type == ActionType.BLOCK:
            if is_malicious:
                self.true_positives += 1
                msg_parts.append("✅ Correct BLOCK — threat neutralised.")
            else:
                self.false_positives += 1
                msg_parts.append("⚠️  False positive — benign packet blocked.")
        elif action.action_type == ActionType.ALLOW:
            if is_malicious:
                self.missed_threats += 1
                msg_parts.append("❌ Missed threat — malicious packet allowed through!")
            else:
                msg_parts.append("✅ Correct ALLOW — benign packet passed.")
        elif action.action_type == ActionType.ESCALATE:
            self.escalations += 1
            if is_malicious:
                self.true_positives += 1
                msg_parts.append("🚨 ESCALATED — threat correctly flagged for review.")
            else:
                msg_parts.append("⚠️  False escalation — benign packet escalated.")

        done = self.step_count >= self.task_cfg["max_steps"]

        # Next packet
        next_pkt = (
            self._packet_queue[self.step_count][0]
            if not done
            else pkt  # episode over, repeat last for structure
        )

        return NIDSObservation(
            packet           = next_pkt,
            step_number      = self.step_count,
            packets_seen     = self.step_count,
            true_positives   = self.true_positives,
            false_positives  = self.false_positives,
            missed_threats   = self.missed_threats,
            escalations_used = self.escalations,
            episode_done     = done,
            message          = " | ".join(msg_parts) or "Action processed.",
        )

    # ------------------------------------------------------------------
    def state(self) -> NIDSState:
        score = _compute_reward(
            self.true_positives, self.false_positives,
            self.missed_threats, self.escalations,
            self.task_cfg["max_steps"], self.task_cfg["difficulty"],
        )
        return NIDSState(
            episode_id     = self.episode_id,
            task_name      = self.task_name,
            difficulty     = self.task_cfg["difficulty"],
            step_count     = self.step_count,
            max_steps      = self.task_cfg["max_steps"],
            current_score  = score,
            threat_budget  = max(0, self.task_cfg["max_steps"] - self.step_count),
            inspect_tokens = self.inspect_tokens,
        )

    # ------------------------------------------------------------------
    def _terminal_obs(self, message: str) -> NIDSObservation:
        pkt, _ = self._packet_queue[-1]
        return NIDSObservation(
            packet           = pkt,
            step_number      = self.step_count,
            packets_seen     = self.step_count,
            true_positives   = self.true_positives,
            false_positives  = self.false_positives,
            missed_threats   = self.missed_threats,
            escalations_used = self.escalations,
            episode_done     = True,
            message          = message,
        )

    # ------------------------------------------------------------------
    def get_episode_data(self) -> dict:
        return {
            "true_positives":   self.true_positives,
            "false_positives":  self.false_positives,
            "missed_threats":   self.missed_threats,
            "escalations_used": self.escalations,
            "step_count":       self.step_count,
        }
