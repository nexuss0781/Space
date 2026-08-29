import json
from pathlib import Path

from nexuss_ao import NexussAO, NexussRequest
from nexuss_ao.adapters import StaticSpecialist
from nexuss_ao.schema import EvidenceEvent, packet_from_events


def test_packet_is_typed_and_tracks_missing_modalities():
    packet = packet_from_events("r1", [EvidenceEvent("audio", "0", "hello", .9, model="asr")], {})
    assert packet.schema_version == "nexuss-ao.evidence.v1"
    assert packet.available_modalities == ("audio",)
    assert "image" in packet.missing_modalities
    assert packet.to_dict()["events"][0]["confidence"] == .9


def test_late_fusion_preserves_specialist_outputs_and_abstains_on_missing():
    hub = NexussAO(
        text_model=StaticSpecialist("text", "smollm2-360m", "Grounded answer"),
        vision_model=StaticSpecialist("image", "smolvlm2-500m", "A red square", .95),
        audio_model=StaticSpecialist("audio", "qwen3-asr-0.6b", "hello world", .99),
    )
    result = hub.answer(NexussRequest(text="What is present?", image="not-a-file", audio="not-a-file", request_id="smoke-1"))
    assert result["request_id"] == "smoke-1"
    assert result["answer"] == "Grounded answer"
    assert "Missing modalities" in result["hub_prompt"]
    assert result["evidence"]["events"][0]["modality"] == "text"


def test_fixture_smoke_report(tmp_path):
    hub = NexussAO(
        text_model=StaticSpecialist("text", "smollm2-360m", "The answer is grounded."),
        vision_model=StaticSpecialist("image", "smolvlm2-500m", "A blue triangle", .9),
    )
    result = hub.answer(NexussRequest(text="Describe it.", image="fixture://blue-triangle", request_id="fixture-1"))
    report = {"implementation": "late-fusion", "request": "fixture-1", "answer": result["answer"], "event_count": len(result["evidence"]["events"]), "missing": result["evidence"]["missing_modalities"]}
    out = tmp_path / "smoke.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    assert report["event_count"] == 2
    assert out.exists()
