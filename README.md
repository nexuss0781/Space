# SPACE

Micro-tier perception stack for a self-improving AI: small GGUF models, quantized, run on CPU via [llama.cpp](https://github.com/ggml-org/llama.cpp), benchmarked on GitHub Actions.

Public runtime models live on HuggingFace: https://huggingface.co/Nexuss0781/SPACE

## Benchmarks

All runs executed on the same GitHub Actions runner (Ubuntu, AMD EPYC 7763, 4 vCPU, 16.77 GB RAM). Text measured with `llama.cpp` GGUF q4_k_m; vision measured with `llama-mtmd-cli` (mmproj, one model per job). Full reports: `reports/vision-benchmark.md`.

### Text leg (10 prompts, 256 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 prompts) |
|---|---:|---:|---:|---:|
| **SmolLM2-360M-Instruct** (q4_k_m) | 0.27 GB | **57.73** | 474 MB | 37.7s |
| SmolLM3-3B (q4_k_m) | 1.92 GB | 13.18 | 3,436 MB | 196.6s |
| Qwen2.5-3B-Instruct (q4_k_m) | 1.93 GB | 13.43 | 3,336 MB | 161.6s |

Earlier bf16 baseline for the micro tier: 2.26 tok/s, 6,397 MB peak RSS (~25x slower, ~13x heavier than the q4 micro model).

### Vision leg (10 image tasks, 128 max new tokens)

| Model | Weights | Avg tok/s | Peak RSS | Wall (10 imgs) |
|---|---:|---:|---:|---:|
| **SmolVLM2-500M-Video-Instruct** (q8_0) | 0.44 GB | **67.15** | 1,026 MB | 73.7s |
| SmolVLM-500M-Instruct (q8_0) | 0.44 GB | 65.06 | 1,026 MB | 77.6s |
| SmolVLM2-2.2B-Instruct (q4_k_m) | 1.11 GB | 19.15 | 4,260 MB | 363.3s |
| Moondream2 (q8_0) | 1.51 GB | 14.86 | 4,000 MB | 236.1s |
| Gemma 3 4B IT (q4_k_m) | 2.49 GB | 13.19 | 5,309 MB | 602.6s |

Result: the micro-tier models are ~4.5x faster than the large ones on both legs. Working pair: **SmolLM2-360M / SmolVLM2-500M** as the always-on edge runtimes, with a 3B-class model (Qwen2.5-3B / SmolVLM2-2.2B) in the swappable quality slot.

## Layout

- `benchmarks/` — harnesses and task sets (text + vision), run by GitHub Actions
- `reports/` — consolidated benchmark papers
- `models/upload/` — GGUF upload tooling for `Nexuss0781/SPACE`
- `prompts/` — prompt sets