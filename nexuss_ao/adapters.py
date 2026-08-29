"""Adapters keep heterogeneous model files separate and make provenance explicit."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .schema import EvidenceEvent, media_hash

class Specialist(Protocol):
    modality: str
    name: str
    def infer(self, media: str | None, prompt: str) -> list[EvidenceEvent]: ...

@dataclass
class StaticSpecialist:
    modality: str
    name: str
    response: str
    confidence: float = 1.0
    def infer(self, media: str | None, prompt: str) -> list[EvidenceEvent]:
        digest = media_hash(Path(media).read_bytes()) if media and Path(media).exists() else None
        return [EvidenceEvent(self.modality, "0", self.response, self.confidence, model=self.name, source_sha256=digest)]

@dataclass
class CommandSpecialist:
    """Run a pinned local CLI. The command must print JSON or plain text to stdout."""
    modality: str
    name: str
    command: str
    timeout_s: int = 600
    def infer(self, media: str | None, prompt: str) -> list[EvidenceEvent]:
        if not self.command:
            raise ValueError(f"no command configured for {self.name}")
        argv = shlex.split(self.command)
        env = dict(os.environ)
        env.update({"NEXUSS_MEDIA": media or "", "NEXUSS_PROMPT": prompt})
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=self.timeout_s, env=env, check=True)
        raw = proc.stdout.strip()
        confidence: float | None = None
        content = raw
        fields: dict[str, Any] = {}
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                content = str(obj.get("text", obj.get("response", raw)))
                confidence = obj.get("confidence")
                fields = {k: v for k, v in obj.items() if k not in {"text", "response", "confidence"}}
        except json.JSONDecodeError:
            pass
        digest = media_hash(Path(media).read_bytes()) if media and Path(media).exists() else None
        return [EvidenceEvent(self.modality, "0", content, confidence, fields=fields, model=self.name, source_sha256=digest)]
