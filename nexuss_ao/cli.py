#!/usr/bin/env python3
"""CLI for Nexuss-AO. Real experts are supplied as command templates via flags/env."""
from __future__ import annotations
import argparse, json
from .adapters import CommandSpecialist, StaticSpecialist
from .hub import NexussAO, NexussRequest

def main() -> None:
    ap = argparse.ArgumentParser(description="Nexuss-AO auditable multimodal late-fusion hub")
    ap.add_argument("--text", default="")
    ap.add_argument("--image")
    ap.add_argument("--audio")
    ap.add_argument("--video")
    ap.add_argument("--vision-command", help="CLI; receives NEXUSS_MEDIA and NEXUSS_PROMPT")
    ap.add_argument("--audio-command", help="CLI; receives NEXUSS_MEDIA and NEXUSS_PROMPT")
    ap.add_argument("--video-command", help="CLI; receives NEXUSS_MEDIA and NEXUSS_PROMPT")
    ap.add_argument("--text-command", help="CLI; receives NEXUSS_PROMPT")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    text = CommandSpecialist("text", "text-command", args.text_command) if args.text_command else StaticSpecialist("text", "text-fallback", "No text command configured.")
    vision = CommandSpecialist("image", "vision-command", args.vision_command) if args.vision_command else None
    audio = CommandSpecialist("audio", "audio-command", args.audio_command) if args.audio_command else None
    video = CommandSpecialist("video", "video-command", args.video_command) if args.video_command else None
    if args.smoke:
        text = StaticSpecialist("text", "smollm2-360m-fixture", "Smoke test answer grounded in evidence.")
        vision = StaticSpecialist("image", "smolvlm2-500m-fixture", "A blue triangle.")
    print(json.dumps(NexussAO(text, vision, audio, video).answer(NexussRequest(args.text, args.image, args.audio, args.video)), indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
