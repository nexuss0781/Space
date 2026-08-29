# SPACE

## Text (10 prompts, 256 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 prompts) |
|---|---:|---:|---:|---:|
| **SmolLM2-360M-Instruct** (q4_k_m) | 0.27 GB | **57.73** | 474 MB | 37.7s |
| SmolLM3-3B (q4_k_m) | 1.92 GB | 13.18 | 3,436 MB | 196.6s |
| Qwen2.5-3B-Instruct (q4_k_m) | 1.93 GB | 13.43 | 3,336 MB | 161.6s |

## Vision (10 image tasks, 128 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 imgs) |
|---|---:|---:|---:|---:|
| **SmolVLM2-500M-Video-Instruct** (q8_0) | 0.44 GB | **67.15** | 1,026 MB | 73.7s |
| SmolVLM-500M-Instruct (q8_0) | 0.44 GB | 65.06 | 1,026 MB | 77.6s |
| SmolVLM2-2.2B-Instruct (q4_k_m) | 1.11 GB | 19.15 | 4,260 MB | 363.3s |
| Moondream2 (q8_0) | 1.51 GB | 14.86 | 4,000 MB | 236.1s |
| Gemma 3 4B IT (q4_k_m) | 2.49 GB | 13.19 | 5,309 MB | 602.6s |

## Audio (10 speech clips, 128 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 clips) |
|---|---|---:|---:|---:|
| **Qwen3-ASR-0.6B** (q8_0) | 0.80 GB | **37.17** | 2,073 MB | 24.1s |
| Ultravox 1B (q4_k_m) | 0.81 GB | 35.68 | 2,969 MB | 209.2s |
| Qwen2.5-Omni-3B (q4_k_m) | 2.10 GB | 15.90 | 6,302 MB | 435.4s |
| Qwen3-ASR-1.7B (q8_0) | 2.17 GB | 12.68 | 3,510 MB | 45.6s |
| Ultravox 8B (q4_k_m) | 4.92 GB | 7.34 | 10,456 MB | 448.1s |

## Selected

| Modality | Micro (always-on) | Quality |
|---|---|---|
| Text | **SmolLM2-360M-Instruct** (q4_k_m, 57.7 tok/s) | **Qwen2.5-3B-Instruct** (q4_k_m, 13.4 tok/s) |
| Vision | **SmolVLM2-500M-Video-Instruct** (q8_0, 67.2 tok/s) | **SmolVLM2-2.2B-Instruct** (q4_k_m, 19.2 tok/s) |
| Audio | **Qwen3-ASR-0.6B** (q8_0, 37.2 tok/s) | **Qwen3-ASR-1.7B** (q8_0, 12.7 tok/s) |