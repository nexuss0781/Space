# Training a Unified Multimodal Model from Pretrained Modality Experts

## Executive recommendation

Build the unified model as a **modular, teacher-preserving system** first, rather than jointly fine-tuning every expert from the start. Keep the pretrained text, vision, and audio experts frozen while learning modality-specific tokenizers/projectors into a common language-model interface. Use a progressive alignment curriculum, with text as the shared semantic bridge, and distill both specialist behavior and intermediate representations into the shared interface. Only after the frozen-expert system passes unimodal retention gates should a small, parameter-efficient subset of the backbone be unfrozen. Every multimodal training phase should include replay of text-only and specialist unimodal examples so that improvements in cross-modal behavior cannot silently erase the capabilities already present in the experts.

This recommendation follows the modular recipe demonstrated by BLIP-2: a lightweight trainable bridge is learned in two stages while the image encoder and LLM remain frozen, explicitly reducing computation and catastrophic forgetting [1]. LLaVA provides a simple, data-efficient baseline in which a trainable linear projector maps visual features into the LLM word-embedding space, followed by autoregressive instruction tuning on assistant answers only [2]. OneLLM is especially relevant to this project because it progressively aligns additional modalities, uses modality tokenizers plus a universal projection module, mixes projection experts with dynamic routing, keeps the LLM frozen during alignment, and then tunes the LLM while freezing the other modules [3].

## What is in `/home/ubuntu/Space`

The repository currently contains separate benchmark runners for text, vision, and audio, plus an evaluation specification in `research/evaluation.md`. The top-level README reports the following reference throughput results on the local x86 host:

| Modality and selected reference | Weights | Average output throughput | Peak RSS |
|---|---:|---:|---:|
| Text: SmolLM2-360M-Instruct Q4_K_M | 0.27 GB | 57.73 tok/s | 474 MB |
| Vision: SmolVLM2-500M-Video-Instruct Q8_0 | 0.44 GB | 67.15 tok/s | 1,026 MB |
| Audio: Qwen3-ASR-0.6B Q8_0 | 0.80 GB | 37.17 tok/s | 2,073 MB |

The higher-quality references are Qwen2.5-3B-Instruct for text (13.43 tok/s), SmolVLM2-2.2B-Instruct for vision (19.15 tok/s), and Qwen3-ASR-1.7B for audio (12.68 tok/s). The local `audio/tasks.tsv` has ten clips with exact reference transcripts, while `vision/tasks.tsv` has ten image prompts, including an OCR/transcription task. The current benchmark scripts record outputs, latency, throughput, and RSS; they do **not** calculate objective audio WER/CER or vision caption/OCR scores themselves. Therefore, a training release must add the scorers and retain the raw request, media hash, output, and scoring inputs. The evaluation specification already calls for WER components, ANLS, CIDEr/SPICE, code `pass@1`, multimodal slot-level scoring, factuality review, repeated deterministic runs, and explicit failure accounting [4].

The existing reports are useful for speed and qualitative regression checks but should not be treated as proof of quality retention. The ten local items are a smoke test, not a statistically representative training or test set. Create a held-out, deduplicated evaluation split before using any of these examples for prompt or training-data decisions.

## Architecture and loss design

Let `E_m` be the frozen expert for modality `m` (text, image, audio, and any future modality), `A_m` its modality-specific adapter/projector, `R` an optional router, and `L` the language backbone. Each expert emits tokens in its own representation space. Do not compare those tokens directly: first map them with an independently parameterized adapter into a common dimension and normalize them. A practical first implementation is a two-layer MLP per expert, followed by a learned query/resampling bottleneck when the native sequence length is large. This directly addresses the incompatible token spaces that MoVE-KD identifies when distilling distinct vision encoders [6].

The interface should carry explicit modality and segment markers, for example `<|image|>`, `<|audio|>`, `<|text|>`, and end markers. Preserve the LLM's original vocabulary and chat format wherever possible. If new markers are required, register them as special tokens and immediately resize the model's input/output embedding matrices to the new tokenizer length, as required by the Hugging Face tokenizer documentation [5]. Do not introduce a second text tokenizer merely to accommodate non-text modalities. If the project requires actual multimodal generation, use discrete modality tokenizers and de-tokenizers for those output streams; AnyGPT demonstrates the alternative in which speech, image, and music are converted to discrete semantic sequences so that the unchanged LLM can model them autoregressively [7]. For a first release focused on perception and text response, continuous projected modality tokens are lower risk; discrete output tokenizers should be added only with a separately validated codec.

A useful combined objective is:

```text
L = L_answer
  + lambda_feat * sum_m MSE(norm(A_m(E_m(x_m))), norm(h_teacher_m))
  + lambda_logit * T^2 * KL(p_teacher(y|x) || p_student(y|x))
  + lambda_align * L_contrastive
  + lambda_replay * L_replay
  + lambda_missing * L_missing.
```

`L_answer` is autoregressive cross-entropy on assistant tokens only. `L_feat` transfers modality-specific intermediate knowledge after adapter alignment. `L_logit` transfers the specialist's answer distribution when a specialist can solve the example; use temperature `T` and mask teacher losses on low-confidence or contradictory examples. `L_contrastive` aligns paired modality/text representations without forcing unrelated examples together. `L_replay` is ordinary answer loss plus optional logit distillation on replayed old examples. `L_missing` trains the model against a full-input teacher or a consistency target for an intentionally absent branch. Start with answer loss and projector learning; ramp feature/logit distillation from zero over the first warm-up interval, because an uncalibrated teacher interface can otherwise dominate useful supervision.

For multiple experts of the same modality, use a teacher-weighted distillation loss rather than an unweighted average. MoVE-KD uses teacher and token weights, encoder adapters, and a mixture-of-LoRA-experts student to reduce conflicts among teachers; its ablation improves the reported average from 66.5 for LLaVA-1.5 to 68.0 after teacher weighting [6]. Treat this as evidence for the mechanism, not as a promised result on Space's models. Keep the original strongest expert as a high-weight teacher during the transition so that fusion cannot discard its capabilities.

## Staged training plan

### Stage 0 — Freeze the contract and establish baselines

Record immutable hashes and versions for every expert, processor, tokenizer, prompt template, dataset manifest, and scorer. Run the local text, vision, and audio suites exactly as prescribed in `research/evaluation.md`, with deterministic decoding (`temperature=0`, fixed seed where supported), three warm-ups, five measured repetitions, and cold/warm latency separated. Add objective scorers before training: normalized WER with substitution/deletion/insertion counts for `audio/tasks.tsv`; ANLS or exact OCR scoring for the text-in-image item; a pinned caption scorer or human atomic-claim review for vision; and code execution for the palindrome item in the text prompts.

The output of this stage is a baseline card containing quality, p50/p90/p95 latency, throughput, peak RSS, failure rate, and per-example outputs for each selected specialist and micro reference. Establish a second, larger held-out suite with paired and interleaved text-image-audio examples; the current ten-item suites cannot measure true cross-modal reasoning.

**Gate:** all required baseline examples complete, all failures are reported, and each metric can be reproduced from a manifest. No training proceeds if tokenizer, media preprocessing, or prompt-template drift is unresolved.

### Stage 1 — Tokenizer, processor, and modality-interface alignment

Retain the text expert's tokenizer as the canonical text tokenizer. Define a versioned processor that maps each message to an ordered sequence of typed content blocks, with deterministic media preprocessing and explicit absence masks. Hugging Face's multimodal chat-template design uses a content list containing typed image/audio/video/text items and lets a processor perform formatting and preprocessing [5]. Follow that separation: data normalization must occur before the model, not through ad hoc string substitutions inside training.

Train only the modality-specific tokenizers/resamplers and projectors `A_m`; keep every expert and the LLM frozen. Use large, clean paired data for each modality-to-text link, balanced by modality and source. For images, the projector should map the vision token sequence to the LLM embedding width, as in LLaVA [2]. For audio, use the ASR expert's semantically meaningful hidden states rather than raw waveform frames unless the target explicitly requires acoustic generation. Include modality ID and begin/end markers, and mask padding and absent branches in attention.

Use a curriculum that first aligns each modality independently to text, then mixes two modalities, then all modalities. OneLLM's progressive alignment and universal projection module provide a useful precedent: first learn a vision-language model, then mix projection experts and dynamically route while aligning other modalities [3].

**Gate:** frozen-expert outputs are unchanged bit-for-bit (or within the declared quantization tolerance); projector-only training reaches at least 95% of the specialist's held-out unimodal quality on each modality; alignment loss is stable across three seeds; and the projector does not increase the median modality-token budget beyond the declared context limit. No cross-modal stage begins if any modality loses more than 5% relative to its specialist baseline.

### Stage 2 — Behavior and representation distillation into the shared interface

Use the aligned frozen experts as teachers and expose the student to the same raw examples. Distill at three levels, with confidence gating. First, distill normalized hidden states after each modality adapter using MSE or cosine loss. Second, distill pooled/query representations and cross-modal contrastive relationships on paired examples. Third, distill answer logits or teacher-generated answers when the specialist produces a reliable response. Keep the LLM frozen initially; train projectors, small resamplers, routers, and optional LoRA modules in the interface.

For heterogeneous experts, never average native features. Learn one adapter per expert, use a common projection space, and optionally route by modality and input quality. MoVE-KD's adapter and attention-guided teacher/token weighting is a directly relevant design [6]. For image and audio examples where a specialist does not produce a calibrated distribution, use confidence thresholds and hard labels or answer text rather than fabricating soft targets. Retain a direct residual path from the best original projector during the first fusion epochs, then anneal it only after the new path matches its teacher.

**Gate:** on a teacher-agreement set, answer KL and representation error decrease without a rise in hallucination or unsupported atomic claims. The fused model must meet the Stage 1 unimodal quality gate and must not exceed 2% relative degradation on the text-only replay suite. A two-modality held-out suite must show positive cross-modal evidence use over the strongest single-modality baseline, measured by joint slot exact match and a media-grounded factuality score.

### Stage 3 — Progressive multimodal pretraining and selective unfreezing

Mix modality-to-text pairs, paired multimodal examples, and text-only data. Begin with the LLM frozen and then unfreeze only the smallest set that is needed: the final few transformer blocks, cross-modal attention blocks, or LoRA adapters. Keep the original modality experts frozen unless an ablation shows a clear, repeatable bottleneck that cannot be fixed in the projectors. If an expert must be adapted, use a small learning rate, LoRA or scale/shift parameters, and retain a frozen copy for replay distillation.

Use source-balanced sampling rather than raw-dataset-proportional sampling; otherwise the largest image or text corpus will erase the smaller modality. Track examples and tokens by source, language, modality, and task. Deduplicate against evaluation media and near-duplicate captions. Preserve difficult cases: OCR, numbers, names, code, temporal order, negation, and low-SNR audio should be oversampled only in training and reported separately in evaluation.

**Gate:** the model improves the held-out cross-modal joint score by a predeclared minimum (recommended: at least 10% relative over the strongest single-modality or naive-concatenation baseline) while retaining every Stage 2 unimodal gate. Any gain that is confined to a single source or prompt template fails this gate.

### Stage 4 — Multimodal instruction tuning

Construct instruction examples in the same typed chat format used at inference. Include single-modality, two-modality, all-modality, interleaved-turn, refusal/uncertainty, OCR, exact transcription, counting, temporal ordering, and grounded reasoning tasks. LLaVA's recipe is a good minimal precedent: a simple projector followed by autoregressive visual instruction tuning, with loss applied to assistant answer tokens rather than user/context tokens [2]. OneLLM further shows the value of a large, diverse multimodal instruction set spanning captioning, question answering, and reasoning [3].

Use a fixed mixture, for example 40% unimodal instruction, 30% paired multimodal, 15% interleaved multi-turn, and 15% text-only replay at the beginning; tune these proportions using held-out results rather than assuming a universal optimum. Keep a clean, human-verified seed set and cap synthetic data per source. Synthetic instruction data should be filtered for media-groundedness, duplicate answers, and teacher hallucinations. Do not use the benchmark prompts as instruction-tuning examples.

**Gate:** assistant-only loss decreases on validation; text-only instruction quality is no worse than 98% of the text baseline on the objective suite; image and audio specialist retention remains at least 95%; and cross-modal joint exact match, grounded claim precision, and calibration improve over Stage 3. Report per-modality and per-combination results, not only a pooled average.

### Stage 5 — Missing-modality robustness and replay consolidation

During training, sample explicit availability masks. Include all non-empty modality subsets that are supported in production, including text-only, image-only, audio-only, text+image, text+audio, image+audio, and all three. Use modality dropout, branch masking, and an explicit `<|missing:modality|>` state. The drop probabilities should match the expected production availability distribution, with a small uniform component so rare subsets are not untrained.

Train consistency against the full-input teacher where the answer is invariant, but do not force a hallucinated answer when the missing signal is essential. Require calibrated abstention or a request for the missing modality for tasks such as exact audio transcription or image OCR. A missing-modality paper reports that ordinary multimodal models can degrade substantially when a modality is absent and that lightweight scale/shift adaptation can improve arbitrary availability combinations with less than 0.7% of model parameters [8]. Therefore, implement availability-conditioned scale/shift or LoRA adapters before considering a separate network for every subset.

Replay old data continuously. Keep a stratified buffer of text-only instruction, each modality's specialist benchmark-like data, rare hard cases, and prior multimodal examples. Mix fresh data with replay and, when feasible, distill the frozen pre-training checkpoint's logits on replay. Experience Replay (CLEAR) demonstrates the stability/plasticity principle: mixing novel experience with replay and behavioral cloning greatly reduced catastrophic forgetting and could approach simultaneous-training performance even with constrained memory [9]. For this application, replay ratio should be a tunable hyperparameter; begin at 15–25% of batches, raise it if old-task loss or quality drifts, and log the ratio in every run.

**Gate:** for each supported subset, report quality and calibration separately. Recommended release thresholds are at least 90% of full-input quality for nonessential missing modalities, no more than a 15% relative drop for any subset on the balanced joint score, and a clear abstention/request for missing evidence on essential-signal cases. Replay must keep text, vision, and audio regression within their previous 2% relative quality bands after the final consolidation epoch.

### Stage 6 — Packaging, ablations, and release decision

Export the experts, adapters, router, tokenizer/processor, and LLM as a versioned manifest. Verify that an inference process can select the correct branch from the availability mask without loading a missing expert. Compare the final system against at least these ablations: naive concatenation with no distillation; no replay; no modality dropout; all experts unfrozen; one shared projector instead of modality-specific projectors; and no tokenizer/marker alignment. A proposed component is justified only if it improves the predeclared metric or reduces cost without violating retention gates.

Run the full local protocol and the expanded held-out suite five times under the pinned environment. Report cold and warm latency, TTFT, p50/p90/p95 end-to-end latency, model-only decode throughput, peak RSS, output lengths, failed examples, and quality confidence intervals. The local micro references imply provisional deployment budgets of at least 80% of their throughput—45 tok/s text, 54 tok/s vision, and 30 tok/s audio—and no more than 1.3 times the corresponding selected-micro peak RSS, unless the product explicitly accepts a different quality/latency point. These are engineering gates for Space, not claims about the research literature.

## Acceptance-criteria matrix

| Area | Measurement | Release criterion |
|---|---|---|
| Text retention | Exact math answer, palindrome `pass@1`, rubric/claim score on remaining fixed prompts, plus replay loss | At least 98% of text baseline on objective items; no more than 2% relative aggregate regression; zero tokenizer/template failures |
| Audio retention | Normalized WER with `S`, `D`, `I` components on all ten referenced clips and an expanded held-out set | At least 95% of specialist word accuracy, or WER no more than 10% relative above specialist; every clip scored and failures counted |
| Vision retention | OCR exact/ANLS on the text image, caption metric or claim precision on expanded set | At least 95% of specialist score; OCR must not regress by more than 5% relative; contradiction rate no worse than baseline |
| Cross-modal gain | Balanced joint exact match, slot-level F1, grounded-claim precision on held-out paired/interleaved examples | At least 10% relative over strongest single-modality or naive-concatenation baseline, with gains on at least two of three modality pairs |
| Missing modalities | Evaluate every supported non-empty subset with availability masks | At least 90% of full-input quality for nonessential omissions; no subset below 85% unless documented; abstain/request evidence when the missing modality is essential |
| Forgetting | Before/after specialist suites and frozen-checkpoint logit agreement on replay | No modality or text suite loses more than 2% relative after final consolidation; teacher KL and old-task loss remain within predeclared bands |
| Data preservation | Manifest counts, deduplication, source/modality balance, truncation and parse-failure audit | 100% of held-out examples remain untouched; less than 1% unexplained parse/truncation loss; every dropped item is logged with a reason |
| Efficiency | Five repeated runs: cold/warm p50/p90/p95, TTFT, throughput, peak RSS | Provisional Space budgets: at least 45/54/30 tok/s for text/vision/audio micro paths and peak RSS at most 1.3x selected micro references, or an approved product exception |
| Reproducibility | Manifest, hashes, seeds, processor version, exact outputs and scorer inputs | A clean rerun reproduces aggregate quality within 1 percentage point and reports every failure rather than dropping it |

## Risks and decisions to revisit

A frozen expert can preserve unimodal quality but may limit genuinely joint reasoning if its representations discard information needed by another modality. Selective unfreezing should therefore be an evidence-based Stage 3 option, not a default. Distillation can copy teacher errors or force incompatible modality semantics together; confidence-gated targets, teacher weighting, and per-modality adapters are safeguards. Discrete tokenizers simplify any-to-any generation but introduce codec reconstruction loss and vocabulary/context pressure; continuous projectors are preferable for the first text-response release. Finally, missing-modality robustness cannot be inferred from unimodal benchmarks: it requires balanced paired data with known availability masks and an abstention policy.

## References

[1] J. Li et al., **BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models**, arXiv:2301.12597 (2023). https://arxiv.org/html/2301.12597

[2] H. Liu et al., **Visual Instruction Tuning** (LLaVA), arXiv:2304.08485 (2023). https://arxiv.org/html/2304.08485

[3] J. Han et al., **OneLLM: One Framework to Align All Modalities with Language**, arXiv:2312.03700 (CVPR 2024 version). https://arxiv.org/html/2312.03700v2

[4] Space repository, **Evaluation specification**, `/home/ubuntu/Space/research/evaluation.md`. The document defines WER, ANLS, CIDEr/SPICE, `pass@1`, mixed joint scoring, factuality review, and reproducible repeated-run requirements.

[5] Hugging Face Transformers documentation, **Tokenizer** and **Multimodal chat templates**. https://huggingface.co/docs/transformers/en/main_classes/tokenizer and https://huggingface.co/docs/transformers/en/chat_templating_multimodal

[6] Y. Wang et al., **MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders**, arXiv:2501.01709 (2025). https://arxiv.org/html/2501.01709v1

[7] J. Zhan et al., **AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling**, arXiv:2402.12226 (2024). https://arxiv.org/html/2402.12226v2

[8] **Robust Multimodal Learning with Missing Modalities via Parameter-Efficient Adaptation**, arXiv:2310.03986 (2024 version). https://arxiv.org/html/2310.03986v3

[9] D. Rolnick et al., **Experience Replay for Continual Learning**, arXiv:1811.11682 / NeurIPS 2019. https://arxiv.org/html/1811.11682

## Caveat

The cited papers establish mechanisms and empirical precedents, not guaranteed thresholds for this repository's models. Thresholds above are proposed engineering gates and should be finalized after Stage 0 baseline distributions and confidence intervals are available.
