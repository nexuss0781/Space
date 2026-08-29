# Multimodal fusion research notes

**Date:** 2026-08-30  
**Scope:** Practical ways to combine separately trained text, vision, and audio systems into one multimodal system, with emphasis on shared token spaces, projectors/adapters, cross-attention, mixture-of-experts (MoE), and the limits of direct weight merging.

## Executive finding

A “shared space” is not one thing. A joint embedding space (for example, ImageBind) is useful for retrieval, similarity, and routing, but does not by itself make a language model able to autoregressively understand or generate every modality. A shared **autoregressive sequence space** (for example, AnyGPT’s discrete modality tokens) goes further: each modality is tokenized/de-tokenized and one language model applies next-token prediction to the interleaved sequence. The most practical near-term design for SPACE is neither direct weight averaging nor a from-scratch any-to-any model. It is a **modular, text-centric hub**: keep the selected micro models frozen, run vision and audio encoders, train small modality-specific adapters/projectors to a common SmolLM2 hidden interface, and initially expose outputs as explicitly tagged semantic text. Add learned cross-attention or a sparse router only after paired multimodal training data and evaluation exist.

## What is in SPACE

The repository README reports CPU-friendly quantized models and makes a clear speed/quality tradeoff. The selected text micro model is **SmolLM2-360M-Instruct** (q4\_k\_m, 0.27 GB, 57.73 tok/s); the selected vision micro model is **SmolVLM2-500M-Video-Instruct** (q8\_0, 0.44 GB, 67.15 tok/s). The quality alternatives are Qwen2.5-3B-Instruct and SmolVLM2-2.2B-Instruct. The audio benchmark includes **Qwen3-ASR-0.6B** (q8\_0, 0.80 GB, 37.17 tok/s) and larger alternatives, but the README’s “Selected” table does not yet designate an audio micro model. The benchmark scripts use separate model files and, for several audio checkpoints, an `mmproj` file, which is evidence that current deployment is adapter/projector-aware rather than a single interchangeable tensor graph.

The official SmolVLM2 card says it is an image/multi-image/video/text model, built with a **SigLIP image encoder and SmolLM2 text decoder**, and supports interleaved media and text. Thus the vision micro model already demonstrates the “encoder + connector + language decoder” pattern. It is not evidence that an independently trained Qwen3-ASR transformer can be spliced into it by replacing weights. The official Qwen3-ASR repository describes 0.6B and 1.7B speech recognition models with offline/streaming inference and 52 languages/dialects; their stable public interface is speech-to-text, not a drop-in universal audio-token interface.

## Fusion patterns

| Pattern | How it works | Practical value | Main caveat |
|---|---|---|---|
| **Joint embedding / shared latent space** | Train modality encoders so paired items have nearby normalized vectors; use contrastive loss, retrieval, routing, or conditioning. ImageBind learns a joint embedding for image, text, audio, depth, thermal, and IMU using image-paired data. | Excellent first layer for cross-modal retrieval, deduplication, confidence checks, and selecting which expert to run. Encoders can remain frozen. | Embeddings are not autoregressive tokens. Similarity does not guarantee compositional reasoning, temporal fidelity, or generation. A projection into an LLM hidden dimension still needs supervised alignment. |
| **Continuous projector / adapter** | Map encoder features `z_m` through a learned linear/MLP, resampler, Q-Former, or LoRA-conditioned connector into the decoder hidden width: `h_m = P_m(z_m)`. Insert the resulting soft tokens beside text embeddings. | Lowest-risk way to reuse separate frozen encoders; parameter- and memory-efficient; compatible with quantized base models if the connector runs at adequate precision. LLaVA shows a simple vision-to-LLM connector; BLIP-2 shows a stronger query bottleneck. | The output dimension matching the LLM is only a shape constraint. The connector must learn semantics, scale, normalization, token count, and placement from paired data. |
| **Cross-attention** | Keep modality features as a separate memory `K,V`; add decoder cross-attention `Attn(Q_text,K_m,V_m)` at selected layers, optionally behind a learned gate. Flamingo uses a Perceiver Resampler followed by newly initialized gated cross-attention layers interleaved with a frozen LM; BLIP-2’s Q-Former uses learned queries and cross-attends to frozen image features. | Handles variable-length images, video, and long audio without pretending raw features are text tokens; supports interleaved media and selective access. A gated residual can start near the text-only behavior and reduce catastrophic disruption. | Extra attention and KV memory; more training and serving complexity; cross-modal data must teach when and where the decoder should look. A frozen language model may underuse a weakly trained connector. |
| **Discrete modality tokens / unified autoregression** | A modality tokenizer compresses image/audio/music into discrete semantic codes; interleave modality-specific code ranges with text tokens, train one decoder with next-token prediction, then de-tokenize generated codes. AnyGPT uses this architecture without changing the LLM block structure. | One sequence model can reason over and generate several modalities; text tooling (sampling, caching, sequence objectives) transfers directly. | Tokenizer/codebook quality and sequence length dominate cost. Semantic tokens may discard high-frequency detail; codebooks and de-tokenizers add substantial training/data requirements. “Discrete” does not mean all modalities share the same vocabulary or semantics automatically. |
| **Mixture-of-experts** | Use modality-specific encoders/connectors, then a common LLM trunk with shared attention and several FFN experts. A learned router selects top-k experts per token or segment; only selected experts execute. Uni-MoE follows this pattern and progressively trains connectors, modality experts, then the mixed model with LoRA. | Preserves specialization while allowing shared reasoning; conditional compute can keep activated parameters lower than a dense enlarged model. Useful when text, image, and audio distributions conflict. | Router collapse, modality starvation, load imbalance, expert duplication, communication overhead, and difficult quantization. It is a new trained architecture, not a parameter-average operation. |
| **Weight merging / task arithmetic** | Average compatible checkpoints or add/subtract fine-tuning deltas. Task Arithmetic constructs a vector by subtracting a common pretrained model from its fine-tuned version. | Can cheaply combine compatible fine-tunes that share architecture, initialization, tokenizer, parameterization, and a common basin. | Not a general modality fusion method. Independently trained text/vision/audio models usually violate compatibility assumptions, and direct averaging can destroy features in every branch. |

## Evidence from primary sources

### Shared embedding space is alignment, not a language interface

ImageBind [1] learns one joint embedding across six modalities and reports that image-paired data can bind the other modalities. The practical lesson is to use a frozen or lightly tuned shared embedding as an **alignment and retrieval service**, not to assume it is a replacement for a multimodal decoder. For SPACE, an ImageBind-like service could identify whether an audio clip and image refer to a similar event or choose a vision/audio expert; it does not turn SmolLM2 into an image/audio generator.

AnyGPT [2] makes the stronger unified-sequence choice. Its tokenizers convert non-text data into discrete semantic sequences, a core LLM performs next-token prediction, and de-tokenizers reconstruct modalities. The paper explicitly frames this as data-level preprocessing with unchanged LLM architecture and training objective. Its table illustrates an important engineering cost: image uses 32 tokens per image, while speech uses about 50 tokens per second and music about 200 tokens per second. This is attractive for true any-to-any generation but is not a cheap retrofit for the existing SPACE checkpoints.

### Projectors and query bottlenecks are the best first bridge

LLaVA [3] connects a pretrained vision encoder to an LLM and then instruction-tunes on multimodal data. It is a useful minimal baseline: project visual features into the LLM interface, preserve the LLM, and train the connector/instruction path rather than merging full checkpoints.

BLIP-2 [4] gives a more robust adapter design. Its Q-Former has learned query vectors, cross-attention to a frozen image encoder, and a fixed number of output features independent of input resolution. It is trained in two stages: representation alignment with the frozen image encoder, then vision-to-language generation with the frozen LLM. This staged recipe is directly applicable to audio: use a small learned query/resampler module over audio encoder frames, then train it to produce hidden states that the text decoder can interpret. A shared target width should be taken from the actual SmolLM2 decoder configuration, not guessed from the source encoder.

### Cross-attention preserves modularity

Flamingo [5] uses a Perceiver Resampler to convert spatiotemporal vision features into a fixed number of visual tokens, then inserts newly initialized cross-attention layers between pretrained LM layers. The paper emphasizes frozen vision and language components and arbitrarily interleaved visual/text input. For SPACE, a lightweight variant can add one or a few gated cross-attention blocks to a common decoder. Use separate modality memories (vision patches, audio frames) and modality/type/position embeddings; do not concatenate unnormalized raw features and expect self-attention to discover the interface.

### MoE is trained conditional specialization, not merged weights

Uni-MoE [6] explicitly combines modality-specific encoders and connectors with a sparse MoE language model. The common blocks retain shared self-attention while FFN experts and a sparse router handle token-level specialization. Its progressive recipe is: (1) cross-modality alignment through connectors, (2) train modality-specific experts with cross-modal instruction data, and (3) tune the complete mixture on mixed multimodal instructions with LoRA. This sequence is a useful blueprint if SPACE eventually needs one process covering many modalities and workloads. Start with top-1 or top-2 routing and monitor per-modality expert usage; do not add a router before there is mixed training data.

## Why naive weight merging fails

1. **Parameter tensors are coordinate systems, not semantic objects.** Two models can have equal-shaped matrices while their hidden features differ by rotations, permutations, scaling, or layerwise reparameterizations. Averaging coordinates assumes the bases are already aligned.
2. **The checkpoints are not the same function family.** SmolLM2, SmolVLM2, and Qwen3-ASR have different module graphs, tokenizers/codebooks, embedding widths, positional schemes, normalization statistics, attention layouts, and output heads. Most tensors cannot even be meaningfully paired. An `mmproj` file is a learned interface, not proof that arbitrary model weights are mergeable.
3. **Independent minima and task interference create destructive updates.** A text model’s update may use a feature direction that an audio model uses for a different function. Averaging can cancel useful directions or move the result across a high-loss barrier. Task Arithmetic [7] obtains deltas relative to the **same pretrained model**, and Model Soups [8] reports success mainly when fine-tuned models occupy a single low-error basin; those conditions do not hold for separately pretrained modality models.
4. **Vocabularies and sequence semantics conflict.** Text token IDs do not encode image patches or acoustic frames. Renumbering or unioning vocabularies does not provide a learned mapping from visual/acoustic content to language semantics. Directly merging embedding/output matrices can corrupt both token likelihoods and modality codebooks.
5. **Quantization magnifies the risk.** Averaging q4/q8 tensors or dequantizing/requantizing after an incompatible merge introduces rounding and scale errors on top of functional interference. Keep quantized specialist inference separate while training adapters in fp16/bf16/fp32 as hardware allows.

Importantly, merging is not impossible in all settings. Sung et al. [9] empirically studied multimodal merging and found that initialization, architecture, and merging mechanism matter; their non-naive recipe outperformed naive merging on VQA, retrieval, NLVR2, Flickr30k, and ADE20k. This supports a narrow conclusion: if SPACE later trains modality branches from one common initialization and identical architecture, test task-vector/TIES-like methods as an experiment, but do not make them the primary fusion plan for today’s heterogeneous checkpoints.

## Concrete recommendation for SPACE

### Phase 0: ship a reliable late-fusion hub

Keep **SmolLM2-360M-Instruct**, **SmolVLM2-500M-Video-Instruct**, and **Qwen3-ASR-0.6B** as separate processes/modules. Route text directly to SmolLM2; route images/video to SmolVLM2; route speech to Qwen3-ASR. Normalize outputs into a tagged event schema such as `<vision>...</vision>`, `<audio transcript="..." language="...">...</audio>`, and timestamps/confidence where available. Feed the compact structured summaries to SmolLM2 for joint reasoning. This gives a useful multimodal product without pretending the three checkpoints share a hidden space. Preserve original media and specialist outputs for auditability and fallback.

### Phase 1: add learned adapters, not merged backbones

Choose one common decoder as the hub. Because the SmolVLM2 card identifies SmolLM2 as its text decoder, the cleanest experiment is to use the SmolLM2-family decoder interface and reuse the already validated SmolVLM2 vision path where feasible. For audio, retain Qwen3-ASR for transcription and train a separate audio encoder or feature tap plus a small Q-Former/resampler/projector into the same decoder width. Train in stages: modality-text contrastive/alignment loss; frozen-decoder next-token instruction loss; then a small mixed-modality LoRA tune. Compare continuous soft tokens against tagged transcript text. Start with a fixed 16–64 learned query tokens per image/audio segment and aggressively pool long audio; avoid feeding 50 raw tokens per second into a 360M CPU decoder.

### Phase 2: selective cross-attention

If soft-token concatenation loses temporal or spatial evidence, add gated cross-attention at a few upper decoder layers. Use separate vision/audio memories, learned modality/type embeddings, causal masks that prevent future leakage, and a gate initialized near zero. Measure whether the decoder actually uses the memory by modality ablations, not only end-task accuracy. For long audio/video, use chunking plus a temporal resampler and cache summaries.

### Phase 3: MoE only after evidence of interference

If one dense hub cannot retain text quality and modality quality simultaneously, introduce a small sparse MoE in FFN sublayers while keeping shared attention. Initialize experts from the same hub, route by token/segment with top-1 or top-2, and train with mixed data plus router load-balancing loss. Begin with text, vision, and audio experts; log routing entropy, expert capacity drops, per-modality utilization, and quality versus activated FLOPs. This follows Uni-MoE’s progressive approach. It is more engineering work than Phase 1 and should not be attempted merely to reduce checkpoint count.

### Do not do initially

Do not average SmolLM2, SmolVLM2, and Qwen3-ASR weights; do not replace one model’s embedding table with another’s; do not infer a universal token space from equal tensor dimensions; and do not merge already quantized GGUF files. If a single artifact is eventually required, distill the late-fusion/adapted system into a common architecture and validate it against specialist baselines rather than merging by arithmetic.

## Risks and acceptance tests

| Risk | Detection | Mitigation / gate |
|---|---|---|
| Vision or audio hallucination is amplified by the language hub | Modality-drop tests, counterfactual image/audio tests, exact ASR set, calibrated confidence | Require the hub to abstain or cite specialist evidence; retain specialist fallback. |
| Audio latency and context explosion | Measure tokens, wall time, peak RSS, and queue latency by duration | Chunk/stream audio; resample to semantic segments; cap adapter tokens. |
| Connector learns shortcuts instead of grounding | Paired-data split by speaker, scene, and wording; shuffled-modality tests | Contrastive/alignment losses plus hard negatives and modality ablations. |
| Catastrophic forgetting of text | Text-only regression suite from `benchmarks/smollm3/prompts.txt` and perplexity checks | Freeze base first; use low-rank adapters and mixed text replay. |
| Router collapse or expert starvation | Per-layer token counts, entropy, dropped tokens, and modality-by-expert matrix | Capacity limits, auxiliary balancing loss, top-1/top-2 ablation, and rollback. |
| Quantization/interface mismatch | Compare fp16 adapter output to GGUF deployment output | Train connectors at higher precision; quantize only after end-to-end calibration. |
| Privacy/licensing and unsafe output | Review training provenance and model licenses; red-team audio credentials and images | Keep Apache-2.0 notices, avoid unauthorized scraped data, and apply existing model-card restrictions. |

Acceptance should require no regression on the existing text, vision, and audio benchmark suites, plus mixed tests containing (a) text-only, (b) one image plus question, (c) one audio clip plus question, and (d) image+audio+text with one modality intentionally contradictory. Track accuracy/grounding, exact transcription, modality attribution, tok/s, peak RSS, and p95 latency. The repository’s current micro benchmarks make these deployment measurements practical and should remain the baseline.

## References

[1] Rohit Girdhar et al., “ImageBind: One Embedding Space To Bind Them All,” CVPR 2023, arXiv:2305.05665. [Paper](https://arxiv.org/abs/2305.05665) · [official code](https://github.com/facebookresearch/ImageBind).

[2] Jun Zhan et al., “AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling,” arXiv:2402.12226 (2024). [Paper](https://arxiv.org/html/2402.12226v2) · [official code](https://github.com/OpenMOSS/AnyGPT).

[3] Haotian Liu et al., “Visual Instruction Tuning,” NeurIPS 2023 (LLaVA), arXiv:2304.08485. [Paper](https://arxiv.org/abs/2304.08485) · [official project/code](https://github.com/haotian-liu/LLaVA).

[4] Junnan Li et al., “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models,” arXiv:2301.12597 (2023). [Paper](https://arxiv.org/html/2301.12597v3) · [official implementation](https://github.com/salesforce/LAVIS/tree/main/projects/blip2).

[5] Jean-Baptiste Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning,” NeurIPS 2022, arXiv:2204.14198. [Paper](https://arxiv.org/html/2204.14198v2).

[6] Yunxin Li et al., “Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts,” arXiv:2405.11273 (2024). [Paper](https://arxiv.org/html/2405.11273v1) · [official code](https://github.com/HITsz-TMG/UMOE-Scaling-Unified-Multimodal-LLMs).

[7] Gabriel Ilharco et al., “Editing Models with Task Arithmetic,” ICLR 2023, arXiv:2212.04089. [Paper](https://arxiv.org/abs/2212.04089).

[8] Mitchell Wortsman et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” ICML 2022, arXiv:2203.05482. [Paper](https://arxiv.org/abs/2203.05482) · [official code](https://github.com/mlfoundations/model-soups).

[9] Yi-Lin Sung et al., “An Empirical Study of Multimodal Model Merging,” Findings of EMNLP 2023, pp. 1563–1575. [ACL Anthology](https://aclanthology.org/2023.findings-emnlp.105/).

[10] Hugging Face, “SmolVLM2-500M-Video-Instruct” official model card. [Model card](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct).

[11] QwenLM, “Qwen3-ASR” official repository. [Repository](https://github.com/QwenLM/Qwen3-ASR).

## Caveats

The research uses primary paper abstracts/HTML and official model cards/repos where available. The Qwen3-ASR repository is current and describes a speech recognition product; it should not be interpreted as a universal audio representation model. Some future architectures may offer a cleaner native audio-token interface, but that would require re-evaluating this recommendation against their actual tokenizer, decoder compatibility, license, and runtime measurements.
