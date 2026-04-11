"""
nids_environment.py — Server-side NIDS Environment implementation.
Five tasks (easy → expert), each with its own grader.
"""
from __future__ import annotations

import random
import uuid
from typing import Dict, List, Tuple

from models import (
    ActionType, NIDSAction, NIDSObservation, NIDSState,
    PacketFeatures, PacketProtocol, ThreatLevel,
)


_BENIGN_PORTS    = [80, 443, 22, 25, 53, 8080]
_MALICIOUS_PORTS = [4444, 31337, 6666, 1337, 9999]
_PROTOCOLS       = list(PacketProtocol)


def _random_ip() -> str:
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def _make_packet(malicious: bool, difficulty: str) -> Tuple[PacketFeatures, bool]:
    if malicious:
        src_port      = random.choice(_MALICIOUS_PORTS)
        dst_port      = random.randint(1, 65535)
        protocol      = random.choice(_PROTOCOLS)
        packet_size   = random.randint(1400, 65535)
        flow_duration = random.uniform(0.001, 0.5)
        entropy       = random.uniform(6.5, 8.0)
        conn_count    = random.randint(50, 500)
        anomaly_score = random.uniform(0.6, 1.0)
        flags         = random.sample(["SYN", "FIN", "RST", "PSH", "URG"],
                                      k=random.randint(2, 4))

        if difficulty == "hard":
            anomaly_score = max(0.0, anomaly_score - random.uniform(0.2, 0.4))
            entropy       = max(0.0, entropy - random.uniform(1.0, 2.0))
            conn_count    = max(1, conn_count - random.randint(30, 45))

        elif difficulty == "expert":
            anomaly_score = random.uniform(0.05, 0.30)
            entropy       = random.uniform(3.5, 5.0)
            conn_count    = random.randint(5, 25)
            src_port      = random.choice(_BENIGN_PORTS + _MALICIOUS_PORTS)
            flags         = random.sample(["SYN", "ACK", "FIN", "RST", "PSH"],
                                          k=random.randint(1, 3))

        elif difficulty == "adaptive":
            if random.random() < 0.3:
                src_port      = random.choice(_BENIGN_PORTS)
                anomaly_score = random.uniform(0.15, 0.40)
                entropy       = random.uniform(4.0, 5.5)
            else:
                anomaly_score = max(0.0, anomaly_score - random.uniform(0.1, 0.25))
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

        if difficulty == "hard":
            anomaly_score = min(1.0, anomaly_score + random.uniform(0.1, 0.2))

        elif difficulty == "expert":
            anomaly_score = min(1.0, anomaly_score + random.uniform(0.15, 0.35))
            entropy       = random.uniform(4.5, 6.5)
            conn_count    = random.randint(10, 60)

        elif difficulty == "adaptive":
            anomaly_score = min(1.0, anomaly_score + random.uniform(0.05, 0.15))

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


TASKS: Dict[str, dict] = {
    "easy_classification": {
        "difficulty":       "easy",
        "max_steps":        20,
        "threat_ratio":     0.5,
        "inspect_tokens":   5,
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
    "expert_zeroday": {
        "difficulty":       "expert",
        "max_steps":        50,
        "threat_ratio":     0.30,
        "inspect_tokens":   1,
        "description": (
            "Zero-day attacks with novel signatures. Anomaly scores are nearly "
            "indistinguishable between benign and malicious. Agent must rely "
            "on behavioral patterns across multiple features simultaneously."
        ),
    },
    "adaptive_evasion": {
        "difficulty":       "adaptive",
        "max_steps":        35,
        "threat_ratio":     0.40,
        "inspect_tokens":   2,
        "description": (
            "Evasive attacker that mimics benign traffic patterns. Some "
            "malicious packets use benign ports and low entropy to evade "
            "detection. Demands high precision under deception."
        ),
    },
}


def _compute_reward(
    tp: int, fp: int, missed: int, escalations: int,
    max_steps: int, difficulty: str,
    streak: int = 0, steps_taken: int = 0,
) -> float:
    precision_score = tp / max(tp + fp, 1)
    recall_score    = tp / max(tp + missed, 1)
    f1              = (2 * precision_score * recall_score /
                       max(precision_score + recall_score, 1e-9))

    escalation_bonus = min(escalations * 0.05, 0.1)

    diff_bonus = {
        "easy": 0.0, "medium": 0.05, "hard": 0.1,
        "expert": 0.15, "adaptive": 0.08,
    }[difficulty]

    streak_bonus = min(streak * 0.01, 0.05)

    efficiency = 1.0
    if max_steps > 0 and steps_taken > 0:
        efficiency = min(steps_taken / max_steps, 1.0)
    efficiency_bonus = (1.0 - efficiency) * 0.05

    raw = f1 + escalation_bonus + diff_bonus + streak_bonus + efficiency_bonus
    clamped = max(0.01, min(raw, 0.99))
    return round(clamped, 4)


def grade_task(task_name: str, episode_data: dict) -> dict:
    task   = TASKS[task_name]
    diff   = task["difficulty"]
    tp     = episode_data["true_positives"]
    fp     = episode_data["false_positives"]
    missed = episode_data["missed_threats"]
    esc    = episode_data["escalations_used"]
    steps  = episode_data["step_count"]
    streak = episode_data.get("best_streak", 0)

    score = _compute_reward(
        tp, fp, missed, esc, task["max_steps"], diff,
        streak=streak, steps_taken=steps,
    )

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
        "best_streak":      streak,
        "grade": (
            "EXCELLENT" if score >= 0.85 else
            "GOOD"      if score >= 0.65 else
            "FAIR"      if score >= 0.45 else
            "POOR"
        ),
    }


class NIDSEnvironment:
    def __init__(self, task_name: str = "easy_classification"):
        if task_name not in TASKS:
            raise ValueError(f"Unknown task: {task_name}. Choose from {list(TASKS)}")
        self.task_name  = task_name
        self.task_cfg   = dict(TASKS[task_name])
        self._reset_internals()

    def _reset_internals(self) -> None:
        self.episode_id      = str(uuid.uuid4())[:8]
        self.step_count      = 0
        self.true_positives  = 0
        self.false_positives = 0
        self.missed_threats  = 0
        self.escalations     = 0
        self.inspect_tokens  = self.task_cfg["inspect_tokens"]
        self.current_streak  = 0
        self.best_streak     = 0
        self._packet_queue: List[Tuple[PacketFeatures, bool]] = []
        self._pre_generate_episode()

    def _pre_generate_episode(self) -> None:
        max_steps    = self.task_cfg["max_steps"]
        threat_ratio = self.task_cfg["threat_ratio"]
        difficulty   = self.task_cfg["difficulty"]
        rng          = random.Random()

        for _ in range(max_steps):
            is_malicious = rng.random() < threat_ratio
            pkt, label   = _make_packet(is_malicious, difficulty)
            self._packet_queue.append((pkt, label))

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

    def step(self, action: NIDSAction) -> NIDSObservation:
        if self.step_count >= self.task_cfg["max_steps"]:
            return self._terminal_obs("Episode already finished.")

        pkt, is_malicious = self._packet_queue[self.step_count]
        self.step_count  += 1

        correct = False
        msg_parts = []

        if action.action_type == ActionType.INSPECT:
            if self.inspect_tokens > 0:
                self.inspect_tokens -= 1
                hint = "MALICIOUS" if is_malicious else "BENIGN"
                msg_parts.append(f"[INSPECT] Deep scan hint: likely {hint}.")
                self._packet_queue.insert(self.step_count, (pkt, is_malicious))
                self.task_cfg["max_steps"] += 1
            else:
                msg_parts.append("[INSPECT] No tokens left. Packet auto-allowed.")
                if is_malicious:
                    self.missed_threats += 1

        elif action.action_type == ActionType.BLOCK:
            if is_malicious:
                self.true_positives += 1
                correct = True
                msg_parts.append("Correct BLOCK — threat neutralised.")
            else:
                self.false_positives += 1
                msg_parts.append("False positive — benign packet blocked.")

        elif action.action_type == ActionType.ALLOW:
            if is_malicious:
                self.missed_threats += 1
                msg_parts.append("Missed threat — malicious packet allowed through.")
            else:
                correct = True
                msg_parts.append("Correct ALLOW — benign packet passed.")

        elif action.action_type == ActionType.ESCALATE:
            self.escalations += 1
            if is_malicious:
                self.true_positives += 1
                correct = True
                msg_parts.append("ESCALATED — threat correctly flagged for review.")
            else:
                msg_parts.append("False escalation — benign packet escalated.")

        if correct:
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.current_streak = 0

        done = self.step_count >= self.task_cfg["max_steps"]

        next_pkt = (
            self._packet_queue[self.step_count][0]
            if not done
            else pkt
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

    def state(self) -> NIDSState:
        score = _compute_reward(
            self.true_positives, self.false_positives,
            self.missed_threats, self.escalations,
            self.task_cfg["max_steps"], self.task_cfg["difficulty"],
            streak=self.best_streak, steps_taken=self.step_count,
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

    def get_episode_data(self) -> dict:
        return {
            "true_positives":   self.true_positives,
            "false_positives":  self.false_positives,
            "missed_threats":   self.missed_threats,
            "escalations_used": self.escalations,
            "step_count":       self.step_count,
            "best_streak":      self.best_streak,
        }
