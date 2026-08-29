# SPACE / Nexuss-AO

SPACE benchmarks compact text, vision, and audio models for always-on deployment. The selected micro specialists are **SmolLM2-360M-Instruct** for text, **SmolVLM2-500M-Video-Instruct** for image/video, and **Qwen3-ASR-0.6B** for speech. Existing local measurements are retained below; they are throughput/RSS measurements, not cross-modal quality scores.

## Nexuss-AO production baseline

The repository now includes [`nexuss_ao/`](nexuss_ao/) as the production multimodal base **system**. It exposes one request API while deliberately keeping heterogeneous specialists separate:

```text
text + image/video + audio
          |
  specialist inference
          |
 versioned typed evidence + provenance
          |
 SmolLM2 text-centric hub response
```

This is the safest quality-preserving fusion available without training a new checkpoint. It does **not** claim that incompatible GGUF files have been merged, that SmolLM2 natively consumes pixels/waveforms, or that the system generates images/audio. Raw media hashes, specialist names, confidence, evidence events, missing modalities, and the final hub prompt are retained for audit and fallback.

### Run locally

```bash
python -m pip install pytest
python -m pytest -q
python -m nexuss_ao.cli --smoke --text "Describe the supplied evidence."
```

Real local model CLIs can be connected without changing the hub by passing command templates. Each command receives `NEXUSS_MEDIA` and `NEXUSS_PROMPT` and may emit plain text or JSON such as `{"text":"...", "confidence":0.9}`:

```bash
python -m nexuss_ao.cli \
  --text "What is shown and said?" \
  --image path/to/image.jpg --audio path/to/audio.wav \
  --vision-command 'your-pinned-vision-cli' \
  --audio-command 'your-pinned-audio-cli' \
  --text-command 'your-pinned-text-cli'
```

The bounded CPU workflow [`.github/workflows/nexuss-ao-smoke.yml`](.github/workflows/nexuss-ao-smoke.yml) runs tests, a deterministic fixture inference, JSON validation, and uploads only diagnostic artifacts. It is intentionally an **inference smoke test**, not a training run. Existing full candidate benchmarks remain manual and are not silently converted into unified-model quality claims.

## Fusion research conclusion

Direct tensor averaging, vocabulary surgery, or merging already-quantized GGUF weights is rejected because the selected models have incompatible computation graphs, tokenizers, hidden representation bases, positional schemes, output heads, and quantization behavior. The recommended future unified-checkpoint experiment is staged modality-specific projector/resampler training into the actual SmolLM2 decoder width, with frozen experts, typed modality markers, teacher/logit/feature distillation, text replay, modality dropout, and specialist residual fallbacks. Gated cross-attention should be added only if ablations demonstrate lost grounding; sparse MoE should be considered only after measured mixed-data interference.

The research record is available in [`research/final-synthesis.md`](research/final-synthesis.md), with detailed notes in [`research/fusion.md`](research/fusion.md), [`research/training.md`](research/training.md), [`research/evaluation.md`](research/evaluation.md), and [`research/github-actions.md`](research/github-actions.md). Proposed release gates include at least 95% specialist retention, no more than 2% text replay regression, measurable held-out cross-modal gain, missing-modality robustness, objective WER/OCR/grounding scores, and reproducible latency/RSS bounds. These are engineering gates, not results already achieved.

## Existing benchmark results

### Text (10 prompts, 256 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 prompts) |
|---|---:|---:|---:|---:|
| **SmolLM2-360M-Instruct** (q4_k_m) | 0.27 GB | **57.73** | 474 MB | 37.7s |
| SmolLM3-3B (q4_k_m) | 1.92 GB | 13.18 | 3,436 MB | 196.6s |
| Qwen2.5-3B-Instruct (q4_k_m) | 1.93 GB | 13.43 | 3,336 MB | 161.6s |

### Vision (10 image tasks, 128 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 imgs) |
|---|---:|---:|---:|---:|
| **SmolVLM2-500M-Video-Instruct** (q8_0) | 0.44 GB | **67.15** | 1,026 MB | 73.7s |
| SmolVLM-500M-Instruct (q8_0) | 0.44 GB | 65.06 | 1,026 MB | 77.6s |
| SmolVLM2-2.2B-Instruct (q4_k_m) | 1.11 GB | 19.15 | 4,260 MB | 363.3s |
| Moondream2 (q8_0) | 1.51 GB | 14.86 | 4,000 MB | 236.1s |
| Gemma 3 4B IT (q4_k_m) | 2.49 GB | 13.19 | 5,309 MB | 602.6s |

### Audio (10 speech clips, 128 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 clips) |
|---|---|---:|---:|---:|
| **Qwen3-ASR-0.6B** (q8_0) | 0.80 GB | **37.17** | 2,073 MB | 24.1s |
| Ultravox 1B (q4_k_m) | 0.81 GB | 35.68 | 2,969 MB | 209.2s |
| Qwen2.5-Omni-3B (q4_k_m) | 2.10 GB | 15.90 | 6,302 MB | 435.4s |
| Qwen3-ASR-1.7B (q8_0) | 2.17 GB | 12.68 | 3,510 MB | 45.6s |
| Ultravox 8B (q4_k_m) | 4.92 GB | 7.34 | 10,456 MB | 448.1s |
