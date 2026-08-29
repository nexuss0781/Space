"""Nexuss-AO late-fusion hub: one API, separate experts, auditable evidence."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Mapping

from .adapters import Specialist
from .schema import EvidenceEvent, EvidencePacket, media_hash, packet_from_events

@dataclass(frozen=True)
class NexussRequest:
    text: str = ""
    image: str | None = None
    audio: str | None = None
    video: str | None = None
    request_id: str | None = None

@dataclass
class NexussAO:
    text_model: Specialist
    vision_model: Specialist | None = None
    audio_model: Specialist | None = None
    video_model: Specialist | None = None
    min_confidence: float = 0.0
    _last_packet: EvidencePacket | None = field(default=None, init=False, repr=False)

    def _run(self, model: Specialist | None, modality: str, media: str | None, prompt: str) -> list[EvidenceEvent]:
        if not media or model is None:
            return []
        events = model.infer(media, prompt)
        return [e for e in events if e.confidence is None or e.confidence >= self.min_confidence]

    def prepare_evidence(self, request: NexussRequest) -> EvidencePacket:
        request_id = request.request_id or str(uuid.uuid4())
        events: list[EvidenceEvent] = []
        if request.text:
            events.append(EvidenceEvent("text", "text-0", request.text, 1.0, model="request"))
        events += self._run(self.vision_model, "image", request.image, request.text)
        events += self._run(self.audio_model, "audio", request.audio, request.text)
        events += self._run(self.video_model, "video", request.video, request.text)
        hashes = {m: media_hash(v.encode()) for m, v in (("image", request.image), ("audio", request.audio), ("video", request.video)) if v}
        self._last_packet = packet_from_events(request_id, events, hashes)
        return self._last_packet

    def build_hub_prompt(self, packet: EvidencePacket) -> str:
        blocks = ["You are Nexuss-AO, an evidence-grounded multimodal assistant.", "Use only the typed evidence below."]
        for event in packet.events:
            if event.modality == "text":
                continue
            meta = {"modality": event.modality, "segment_id": event.segment_id, "confidence": event.confidence, "model": event.model, "fields": dict(event.fields)}
            tag = f"{event.modality}_evidence"
            blocks.append(f"<{tag}>\n{json.dumps(meta, sort_keys=True)}\n{event.content}\n</{tag}>")
        if packet.missing_modalities:
            blocks.append("Missing modalities: " + ", ".join(packet.missing_modalities) + ". Do not invent their contents.")
        blocks.append("Answer the user's text request concisely; state uncertainty or request missing evidence when exactness depends on it.")
        return "\n\n".join(blocks)

    def answer(self, request: NexussRequest) -> dict[str, Any]:
        packet = self.prepare_evidence(request)
        prompt = self.build_hub_prompt(packet)
        result = self.text_model.infer(None, prompt)
        answer = result[0].content if result else "I cannot answer without a configured text model."
        return {"request_id": packet.request_id, "answer": answer, "evidence": packet.to_dict(), "hub_prompt": prompt}
