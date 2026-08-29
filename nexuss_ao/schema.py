"""Versioned, auditable evidence objects passed between modality experts and the hub."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any, Mapping

SCHEMA_VERSION = "nexuss-ao.evidence.v1"

@dataclass(frozen=True)
class EvidenceEvent:
    modality: str
    segment_id: str
    content: str
    confidence: float | None = None
    start_s: float | None = None
    end_s: float | None = None
    fields: Mapping[str, Any] = field(default_factory=dict)
    model: str = "unknown"
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.modality not in {"text", "image", "audio", "video", "system"}:
            raise ValueError(f"unsupported modality: {self.modality}")
        if not self.segment_id:
            raise ValueError("segment_id is required")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if (self.start_s is None) != (self.end_s is None):
            raise ValueError("start_s and end_s must be provided together")

@dataclass(frozen=True)
class EvidencePacket:
    schema_version: str
    request_id: str
    events: tuple[EvidenceEvent, ...]
    available_modalities: tuple[str, ...]
    missing_modalities: tuple[str, ...]
    raw_media_sha256: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "events": [asdict(e) for e in self.events],
            "available_modalities": list(self.available_modalities),
            "missing_modalities": list(self.missing_modalities),
            "raw_media_sha256": dict(self.raw_media_sha256),
        }

def media_hash(value: bytes | str) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return sha256(data).hexdigest()

def packet_from_events(request_id: str, events: list[EvidenceEvent], raw_media_sha256: Mapping[str, str]) -> EvidencePacket:
    available = tuple(sorted({e.modality for e in events if e.modality != "system"}))
    all_modalities = ("text", "image", "audio", "video")
    missing = tuple(m for m in all_modalities if m not in available)
    return EvidencePacket(SCHEMA_VERSION, request_id, tuple(events), available, missing, raw_media_sha256)
