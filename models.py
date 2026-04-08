"""
models.py — Typed Action / Observation / State models for NIDS environment.
"""
from __future__ import annotations
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class ActionType(str, Enum):
    ALLOW   = "allow"    # Let packet through
    BLOCK   = "block"    # Drop / firewall the packet
    ESCALATE = "escalate"  # Flag for human review (critical threats)
    INSPECT = "inspect"  # Deep-packet inspect (costs 1 extra step)


class ThreatLevel(str, Enum):
    NONE     = "none"
    LOW      = "low"
    MEDIUM   = "medium"
    HIGH     = "high"
    CRITICAL = "critical"


class PacketProtocol(str, Enum):
    TCP  = "TCP"
    UDP  = "UDP"
    ICMP = "ICMP"
    HTTP = "HTTP"
    DNS  = "DNS"


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class NIDSAction(BaseModel):
    """Action the agent takes on the current packet."""
    action_type: ActionType = Field(..., description="What to do with this packet")
    reason: Optional[str]   = Field(None,  description="Optional reasoning (for logging)")


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------

class PacketFeatures(BaseModel):
    """Simulated packet feature vector exposed to the agent."""
    src_ip:            str
    dst_ip:            str
    src_port:          int
    dst_port:          int
    protocol:          PacketProtocol
    packet_size:       int              # bytes
    flow_duration:     float            # seconds
    bytes_per_second:  float
    flags:             List[str]        # e.g. ["SYN", "ACK"]
    payload_entropy:   float            # 0.0–8.0  (high = obfuscated)
    connection_count:  int              # connections from src in last 60 s
    anomaly_score:     float            # pre-computed heuristic 0.0–1.0


class NIDSObservation(BaseModel):
    """Full observation returned after each step."""
    packet:           PacketFeatures
    step_number:      int
    packets_seen:     int
    true_positives:   int               # correctly blocked threats so far
    false_positives:  int               # benign packets wrongly blocked
    missed_threats:   int               # threats let through
    escalations_used: int
    episode_done:     bool
    message:          str               # human-readable feedback


# ---------------------------------------------------------------------------
# State  (episode metadata)
# ---------------------------------------------------------------------------

class NIDSState(BaseModel):
    """Internal episode state (for introspection / debugging)."""
    episode_id:       str
    task_name:        str
    difficulty:       str
    step_count:       int
    max_steps:        int
    current_score:    float
    threat_budget:    int               # threats remaining in episode
    inspect_tokens:   int               # remaining free inspections
