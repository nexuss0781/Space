# Nexuss-AO: Decision-ready multimodal production and CI synthesis

**Repository:** `/home/ubuntu/Space`  
**Status:** Recommendation for the next production and research milestones  
**Scope:** Text, image/video, and speech inference using the current SPACE model portfolio, followed by a quality-preserving unified-model experiment and reproducible GitHub Actions coverage.

## Executive decision

**Do not merge the SmolLM2, SmolVLM2, and Qwen3-ASR checkpoints or their GGUF files by tensor averaging, task arithmetic, or embedding-table substitution.** These systems have different computation graphs, tokenizers, representation bases, positional schemes, output heads, and quantization behavior. Task arithmetic and model soups are useful mainly when models share an architecture, initialization, parameterization, and compatible basin; those assumptions do not hold for these independently trained heterogeneous checkpoints [1][2]. A multimodal model-merging study supports only the narrower conclusion that carefully designed, common-initialization experiments may be viable later—not that arbitrary checkpoints can be averaged safely [3].

The recommended production path is a **late-fusion, text-centric hub with specialist fallbacks**:

- **SmolLM2-360M-Instruct** handles text reasoning and response generation.
- **SmolVLM2-500M-Video-Instruct** handles image, multi-image, and video perception.
- **Qwen3-ASR-0.6B** handles speech recognition and returns transcript, language, timing, and confidence metadata where available.
- A typed orchestration layer preserves original media and specialist outputs, converts them into bounded, auditable evidence/events, and asks SmolLM2 to reason over that evidence.

This is a production multimodal **system**, not yet a single end-to-end multimodal checkpoint. It is the shortest path to useful capability while retaining specialist quality, fallback behavior, observability, and the ability to replace or upgrade one branch independently. The local selected references are approximately 57.73 tok/s and 474 MB RSS for text, 67.15 tok/s and 1,026 MB RSS for vision, and 37.17 tok/s and 2,073 MB RSS for audio; these are deployment measurements on the repository's host, not quality claims or universal performance guarantees [4][5].

A later unified model should be trained through **modality-specific projectors or Q-Former/resampler adapters into the actual SmolLM2 decoder width**, not by merging backbones. BLIP-2 and LLaVA establish the frozen-encoder/projector pattern; Flamingo provides a precedent for gated cross-attention over a separate modality memory [6][7][8]. A true any-to-any generator using discrete image/audio tokens is a separate, substantially more expensive redesign, as illustrated by AnyGPT's tokenizer/de-tokenizer approach and modality-dependent sequence lengths [9].

## 1. Recommended architecture

### 1.1 Production architecture: late-fusion hub

The first release should expose one API but retain explicit internal branches:

```text
request(text, image/video?, audio?)
        |
        v
availability + provenance + safety checks
        |
  +-----+------------------+
  |                        |
image/video             speech
  |                        |
SmolVLM2              Qwen3-ASR
  |                        |
vision evidence       transcript/events
  +----------+-----------+
             |
             v
  typed evidence normalizer
             |
             v
 SmolLM2-360M text-centric reasoning
             |
             v
answer + citations/evidence spans + confidence/abstention
```

The normalizer should emit a versioned schema rather than unstructured concatenation. A minimal event should contain `modality`, `segment_id`, source-media hash, timestamps where meaningful, text or structured fields, confidence, and processing/model versions. For example, audio may produce a transcript plus language and segment timings; vision may produce a bounded description, OCR candidates, detected entities, and temporal summaries for video. The raw media, specialist output, normalized evidence, and final answer must remain available for audit and fallback.

Use explicit typed markers in the hub prompt, such as `<|text|>`, `<|image_evidence|>`, `<|audio_transcript|>`, `<|event|>`, and end markers. Preserve the SmolLM2 tokenizer and chat format. A multimodal processor should represent content as ordered typed blocks and perform deterministic preprocessing before inference, rather than relying on ad hoc string substitutions [10]. The hub should state when evidence is absent or uncertain and should abstain or request the missing modality when exact evidence is essential, such as exact transcription or image OCR.

This architecture makes a defensible product claim: **the system can combine specialist-produced evidence for multimodal text responses**. It does not justify claiming that SmolLM2 natively understands pixels or waveforms, that all modalities occupy one learned token space, or that the system generates images/audio.

### 1.2 Research architecture: adapters into a common decoder

For the first unified-model experiment, retain the pretrained specialists and decoder frozen. Let `E_m` be the modality expert, `A_m` its modality-specific adapter, and `L` the SmolLM2 decoder. Train:

```text
z_m = E_m(x_m)
h_m = normalize(A_m(resample(z_m)))
answer = L(text embeddings + typed h_m)
```

Use a small two-layer MLP plus a learned query/resampling bottleneck. Start with roughly 16–64 learned query tokens per image or audio segment, with aggressive chunking and pooling for long audio/video. The output width must be read from the actual target decoder configuration; matching a numeric dimension alone does not establish semantic compatibility. For speech, the initial bridge should use semantically meaningful ASR hidden states or structured transcript events, not raw acoustic frames, unless acoustic generation is an explicit product requirement.

Start with soft-token insertion because it is the smallest end-to-end change. If ablations show that it loses spatial grounding, temporal order, or long-context evidence, add only a few **gated cross-attention** blocks in upper decoder layers. Keep vision and audio memories separate, use modality/type/position embeddings, and initialize the residual gate near zero so text-only behavior is not abruptly displaced. This follows the modularity and gated-memory lessons of BLIP-2 and Flamingo [6][8].

Do not add a sparse MoE merely to reduce the number of deployed files. Add a small top-1/top-2 FFN MoE only if mixed-modality experiments demonstrate repeatable interference that adapters and selective unfreezing cannot resolve. Keep shared attention, initialize experts from the same hub, and monitor router entropy, load, capacity drops, modality utilization, activated FLOPs, latency, memory, hallucination, and specialist regressions. Uni-MoE is a precedent for progressive connector, modality-expert, and mixed-data training; it is not evidence that a router will work without appropriate mixed data [11].

### 1.3 Architecture choices explicitly rejected for the first release

A joint embedding service such as ImageBind may be useful for retrieval, routing, deduplication, or confidence checks, but a shared embedding is not an autoregressive language interface [12]. Discrete unified modality tokens, as in AnyGPT, are appropriate only if Nexuss-AO commits to modality codecs, de-tokenizers, long sequence budgets, and generation quality validation [9]. Direct averaging, renumbering/unioning vocabularies, replacing embedding tables, and merging already quantized GGUF weights are rejected.

## 2. Training and data plan

### 2.1 Freeze the contract and baselines first

Before training, create immutable manifests for every checkpoint, projector, tokenizer, processor, prompt template, dependency, dataset split, scorer, and hardware/software environment. Run the current selected specialists unchanged and save per-example outputs. The existing ten-item reports are useful smoke tests, but they are not representative quality benchmarks: they lack broad coverage, objective vision scoring, audio WER, confidence intervals, robust timing distributions, and held-out protection [13].

The baseline card must include quality, failures, cold and warm latency, throughput, TTFT, output lengths, peak RSS/PSS where available, and media/model hashes. Add the objective scorers before using the results to make training decisions: WER with substitution/deletion/insertion components for speech, ANLS or exact OCR scoring for document/text-in-image cases, pinned caption metrics or blinded atomic-claim review for vision, and execution-based `pass@1` for code.

### 2.2 Staged curriculum

**Stage 0 — interface and data contract.** Keep the canonical text tokenizer. Define typed content blocks, deterministic media preprocessing, explicit availability masks, and a versioned chat template. Deduplicate training media/captions against evaluation data and preserve licenses and provenance. Build speaker-, scene-, wording-, and source-separated held-out sets so the connector cannot win through memorization or prompt shortcuts.

**Stage 1 — modality-to-text alignment.** Freeze all experts and the LLM. Train one adapter/resampler per modality on clean paired data. First align each modality independently to text; then train two-modality pairs; then all supported combinations. Balance by modality, source, language, and task instead of allowing the largest corpus to dominate. Include OCR, numbers, names, counting, temporal order, negation, low-SNR audio, and interleaved media, while keeping benchmark test prompts out of instruction data.

**Stage 2 — teacher-preserving distillation.** Distill normalized adapter hidden states, pooled/query representations, and specialist answer logits or reliable teacher answers. Use confidence-gated targets: do not turn uncertain or contradictory specialist outputs into authoritative soft labels. Keep a residual path to the strongest original specialist/projector until the learned path matches its teacher. Teacher weighting is preferable to an unweighted average when multiple specialists or teachers disagree [14].

A practical combined objective is:

```text
L = L_answer + λ_feat L_feature + λ_logit L_KL
    + λ_align L_contrastive + λ_replay L_replay + λ_missing L_missing
```

Begin with answer loss and projector learning; ramp representation and logit distillation after interface calibration. Apply answer loss to assistant tokens only. Keep the decoder frozen until the adapter-only system passes unimodal retention gates.

**Stage 3 — selective unfreezing.** Mix modality-to-text pairs, paired multimodal items, and text-only replay. If needed, unfreeze only LoRA/scale-shift parameters, upper decoder blocks, or newly added cross-attention. Keep experts frozen by default. If an expert must be adapted, retain a frozen teacher copy and distill it during training.

**Stage 4 — instruction tuning.** Use the same typed format at training and inference. Include unimodal, paired, all-modality, interleaved multi-turn, OCR, exact transcription, counting, temporal ordering, contradiction, uncertainty, refusal, and grounded-reasoning examples. An initial mixture can be 40% unimodal instruction, 30% paired multimodal, 15% interleaved multi-turn, and 15% text replay; tune this from held-out results rather than treating it as a universal optimum. Filter synthetic data for media-groundedness, duplicates, teacher hallucinations, and source concentration.

**Stage 5 — missing-modality robustness and consolidation.** Train all supported non-empty subsets: text, image, audio, text+image, text+audio, image+audio, and all three. Use availability-conditioned dropout and an explicit missing state. Distill consistency only when the answer is invariant to the missing signal; otherwise require calibrated abstention or a request for evidence. Start with scale/shift or LoRA adaptation before creating a separate network for every modality subset. Replay 15–25% of batches from text-only, specialist-like, rare-hard-case, and previous multimodal buffers, adjusting upward if old-task quality drifts [15].

### 2.3 Proposed data and release gates

The following are **engineering criteria proposed for Nexuss-AO, not guarantees from the cited literature**:

| Area | Proposed gate before release | Evidence required |
|---|---|---|
| Specialist retention | At least 95% of each specialist's held-out unimodal quality; no more than 2% relative regression on text replay | Per-example paired scores, confidence intervals, and raw outputs |
| Cross-modal value | At least 10% relative gain over the strongest single-modality or naive-concatenation baseline, with gains on at least two modality pairs | Held-out paired/interleaved set, joint exact, slot F1, grounded-claim precision |
| Missing modalities | At least 90% of full-input quality for nonessential omissions; no subset below 85% without written exception | Every availability subset, calibration, abstention/request behavior |
| Data integrity | Less than 1% unexplained parse/truncation loss; no evaluation contamination | Immutable manifest, provenance, deduplication and drop-reason audit |
| Efficiency | Product-approved budgets; provisional micro targets are at least 45/54/30 tok/s for text/vision/audio and no more than 1.3× selected-micro RSS | Five repeated cold/warm runs and process-level memory measurements |
| Reproducibility | Clean rerun within 1 percentage point on aggregate quality | Hashes, seeds, processor version, exact outputs and scorer inputs |

A gate failure should preserve the last passing artifact and specialist fallback. Do not trade away safety, grounding, or auditability for a pooled average improvement.

## 3. Real-test protocol

### 3.1 Evaluation matrix

Treat every release as a named, content-addressed artifact. The manifest must identify model/projector files, SHA-256, quantization, tokenizer, chat template, context and output limits, sampling parameters, code/dependency revisions, dataset versions, and hardware. At minimum evaluate the three selected specialists and the late-fusion/unified candidate separately.

Run four modes where supported: text-only; image+text; audio+text; and mixed image+audio+text. For every multimodal example, run matched ablations: remove image, remove audio, replace audio with the gold transcript, replace image with a blank image, and shuffle media from another item. Shuffled-media performance should remain meaningfully below true-media performance; a gain from shuffling is a grounding regression.

Use a public suite plus a versioned local challenge set:

| Family | Tasks and primary metrics |
|---|---|
| Text | MMLU accuracy, GSM8K numeric exact match, HumanEval `pass@1` |
| Image + text | MMMU exact match, DocVQA ANLS/exact OCR, TextCaps CIDEr/SPICE plus OCR word recall |
| Audio + text | LibriSpeech WER with S/D/I components, AudioCaps CIDEr/SPICE, AHELM task-appropriate scores |
| Mixed | 30 balanced local image+audio+text cases for grounding, temporal order, and contradiction; joint exact and slot metrics |
| Synergy | Ten same-scene pairs; `score(image+audio) − max(score(image-only), score(audio-only))`, plus raw-audio-vs-gold-transcript gap |

These task families follow established holistic evaluation concerns: reproducibility and transparent scenario/metric definitions in HELM, standardized vision-language evaluation in VHELM, audio-language coverage in AHELM, and robustness/efficiency concerns beyond image-text quality in HEIM [16]. MMMU, DocVQA, TextCaps, and AudioCaps provide complementary reasoning, document, text-in-image, and audio-caption coverage [17][18][19][20].

### 3.2 Scoring and run mechanics

Compute scores from raw predictions with checked-in scorers; never use model self-ratings as primary metrics. Use the declared normalizer for exact text; WER is `(S+D+I)/N`; ANLS follows the DocVQA thresholded normalized Levenshtein convention; use official/pinned CIDEr/SPICE scorers; execute HumanEval in a network-disabled resource-limited sandbox; and score mixed cases by required slots, with `joint_exact=1` only when every required slot is correct. Blinded human review of 100 sampled image/audio captions should report supported-claim precision, contradiction rate, and inter-rater agreement.

Pin the container/OS, Python, inference backend and commit, compiler flags, BLAS/OpenMP settings, thread count, CPU affinity, and environment variables. Use deterministic decoding (`temperature=0`, fixed seed 42, fixed context and output limits) where supported. If a model is intrinsically stochastic, use five seeds and report mean and standard deviation. Run three warm-ups, then five measured repetitions; separate cold process/load measurements from warm persistent-process measurements. Save every request, media hash, output, error stream, exit code, timing record, and scorer input. A timeout, OOM, crash, or missing example is a reported failure, not a dropped row.

Report load time, TTFT, prefill, decode and end-to-end latency; model-only and end-to-end throughput; p50/p90/p95 and variance; audio real-time factor; image latency by image; peak RSS and PSS; and GPU VRAM/utilization when applicable. The current text runner's extra one-token TTFT request and the audio/vision runners' process-per-item loading must be labeled as estimates/cold end-to-end until replaced by instrumented measurements [13].

### 3.3 Regression decision rule

Use paired 10,000-resample bootstrap confidence intervals stratified by task family, not rounded leaderboard means. A proposed hard quality rule is: accuracy-like metrics no more than 1 percentage point lower; ANLS/CIDEr/SPICE no more than 3% relative lower; WER no more than 5% relative higher; no safety or contradiction diagnostic more than 1 percentage point worse; closed-answer format failures at most 1%; and zero failures on credential/code-safety cases. Mixed synergy must not decline by more than 0.02, and shuffled-media score must remain at least 5 points below true-media score.

Proposed systems gates relative to the same baseline and hardware are warm p50 ≤+10%, warm p95 ≤+15%, cold load-plus-first-request ≤+20%, peak RSS/PSS and GPU VRAM ≤+10%, audio RTF increase ≤10% and below 1.0 for a real-time target, throughput decrease ≤10%, zero unexplained failures, and warm-latency coefficient of variation ≤10%. These are proposed release rules. They must be finalized after Stage 0 distributions exist and do not imply that Nexuss-AO currently passes them.

Every release should publish `manifest.json`, `config.json`, raw per-example JSONL, normalized predictions, `scores.json` with confidence intervals, and a Markdown summary containing the exact command, container digest, hardware, and explicit FAIL records [21].

## 4. GitHub Actions implementation plan

### 4.1 Always-on inference smoke workflow

Add `.github/workflows/model-smoke.yml` as a separate workflow. It should run on relevant pull requests and pushes, with `workflow_dispatch` for optional GPU/full modes. Set `permissions: contents: read`, cancel superseded PR runs with concurrency, and use a 10–15 minute timeout. The default matrix should contain one CPU row on `ubuntu-latest`; standard `ubuntu-latest` must be treated as CPU-only. Add a GPU row only for an organization-configured self-hosted or larger-runner label, with an explicit `nvidia-smi` and backend/device assertion. Do not claim GPU coverage when a CPU job merely reports that CUDA is unavailable.

The PR path should use one tiny public quantized fixture (or a deterministic local stub when no suitable fixture exists), one checked-in task, 8–16 generated tokens, fixed seed, short context, minimal locked dependencies, and no optimizer, gradient, checkpoint, or weight-update code. The result must be labeled **inference smoke**, never training. If vision/audio paths are included, use one exact small model/projector pair and one sample per path; do not download the existing 3B–8B quality candidates in pull-request CI.

Fail the smoke job on nonzero inference exit, empty output, missing report, size-limit breach, checksum/revision mismatch, or unavailable GPU for a requested GPU row. Upload only bounded JSON/Markdown/log diagnostics per matrix row, with unique names and short retention; never upload model weights, caches, or credentials. A fan-in summary should run with `if: always()`, download each artifact into an isolated directory, and explicitly report failed or missing rows rather than converting an incomplete matrix into success.

### 4.2 Reproducible downloads and caches

Check in a small model lock file containing repository ID, exact filename(s), full Hub commit SHA, expected byte size, and preferably SHA-256. Use `hf_hub_download` for one exact file or `snapshot_download` with exact patterns for a model/projector pair; never select the first matching `*Q4_K_M*.gguf` from a floating snapshot. Validate exact filenames, size, and digest. Use public fixtures without secrets in PR jobs; gated/private candidates belong in a separately approved `workflow_dispatch` workflow with protected environment secrets [22].

Set `HF_HUB_CACHE` before importing the Hub library and cache only pinned public blobs or dependencies. Never cache `HF_HOME` wholesale because it may contain credentials. Key build/dependency caches by OS, architecture, backend, compiler/toolchain inputs, source revision, and the committed dependency lock hash. Pin llama.cpp (or the selected backend) to a full revision. Pin third-party Actions to reviewed full commit SHAs, use least-privilege permissions, and do not expose write tokens to untrusted pull requests [23][24].

### 4.3 Full benchmark workflows

Keep the current multi-candidate ten-task audio, vision, and text workflows manual or scheduled. Add `--limit`/`--smoke` and `--fail-on-error` modes to runners, pin revisions, add expected hashes and byte limits to matrices, and make matrix artifact names include modality, candidate, backend/device, and run ID. Separate CPU and GPU workflows or rows. The full workflows should produce the real-test artifacts and quality scores described above, while the smoke workflow answers only: **does this bounded inference path still load, execute, and emit a valid result?**

## 5. Explicit limitations and what cannot be claimed

1. **No current unified checkpoint has been demonstrated.** The recommendation is an architecture and staged experiment plan. It is not evidence that a projector, cross-attention block, or MoE has been trained or that the proposed gates have been met.
2. **Current ten-task reports are not generalization evidence.** They show local throughput, model sizes, wall times, and selected qualitative outputs. They cannot establish benchmark-quality accuracy, robustness, calibration, multilingual or noisy-speech performance, vision factuality, temporal reasoning, or statistically reliable p95 latency [13].
3. **Throughput is not quality.** The README's tok/s and RSS values are host- and implementation-specific and are confounded by output length, process startup, preprocessing, and quantization. They cannot be compared across modalities as a single quality ranking.
4. **Qwen3-ASR is not a universal audio-token interface.** Its stable public role is speech recognition; using its transcript/hidden states as an audio bridge is a practical experiment, not proof of native audio reasoning or audio generation [25].
5. **The late-fusion hub is not end-to-end grounding by construction.** It can inherit specialist errors, transcription errors, OCR errors, shortcut behavior, and language-prior hallucinations. Media ablations, shuffled-media controls, evidence retention, confidence gating, and specialist fallback are mandatory.
6. **No any-to-any generation claim is justified.** The initial system produces text responses from specialist evidence. Image/audio generation would require validated discrete codecs or modality decoders, additional data, context budget, and safety evaluation [9].
7. **The proposed thresholds are engineering gates.** Literature precedents motivate frozen experts, projectors, resampling, cross-attention, replay, and progressive training; they do not guarantee the numerical retention, synergy, latency, or memory thresholds proposed here.
8. **Licensing, privacy, and safety remain release blockers.** Training provenance, model licenses, audio/image privacy, credential leakage, harmful content, and gated-model terms must be reviewed for every artifact. A passing benchmark score cannot waive these obligations.

## 6. Next steps and go/no-go sequence

**Now (production baseline).** Implement the late-fusion orchestrator, typed evidence schema, raw-output audit trail, confidence/abstention policy, and specialist fallbacks. Add objective scoring to the local runners and generate the baseline card. Add the pinned CPU smoke workflow before changing model code.

**Next (unified-model experiment).** Freeze the manifest and held-out splits. Train modality-specific projectors/resamplers only, with text replay and specialist distillation. Compare tagged evidence text against continuous soft tokens. Stop and retain the late-fusion release if any specialist retention gate fails.

**Then (evidence-based complexity).** Add gated cross-attention only after modality ablations show lost grounding or temporal evidence. Selectively unfreeze LoRA/scale-shift or upper decoder layers only after adapter-only results pass. Test MoE only after measured mixed-data interference, with router and activated-compute telemetry.

**Release decision.** Ship the unified candidate only if it passes the quality, grounding, missing-modality, safety, systems, and reproducibility gates against the pinned baseline. Otherwise ship the late-fusion hub as the production multimodal base and keep the unified checkpoint explicitly experimental. Treat any discrete any-to-any model as a separate future program with its own codec, data, compute, and evaluation budget.

## References

[1] Ilharco et al., “Editing Models with Task Arithmetic,” ICLR 2023. <https://arxiv.org/abs/2212.04089>  
[2] Wortsman et al., “Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time,” ICML 2022. <https://arxiv.org/abs/2203.05482>  
[3] Sung et al., “An Empirical Study of Multimodal Model Merging,” Findings of EMNLP 2023. <https://aclanthology.org/2023.findings-emnlp.105/>  
[4] SPACE repository baseline table. <https://github.com/> (local source: `/home/ubuntu/Space/README.md`)  
[5] Hugging Face, “SmolVLM2-500M-Video-Instruct” model card. <https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct>  
[6] Li et al., “BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models.” <https://arxiv.org/html/2301.12597v3>  
[7] Liu et al., “Visual Instruction Tuning” (LLaVA). <https://arxiv.org/abs/2304.08485>  
[8] Alayrac et al., “Flamingo: a Visual Language Model for Few-Shot Learning.” <https://arxiv.org/html/2204.14198v2>  
[9] Zhan et al., “AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling.” <https://arxiv.org/html/2402.12226v2>  
[10] Hugging Face Transformers, tokenizer and multimodal chat-template documentation. <https://huggingface.co/docs/transformers/en/main_classes/tokenizer> and <https://huggingface.co/docs/transformers/en/chat_templating_multimodal>  
[11] Li et al., “Uni-MoE: Scaling Unified Multimodal LLMs with Mixture of Experts.” <https://arxiv.org/html/2405.11273v1>  
[12] Girdhar et al., “ImageBind: One Embedding Space To Bind Them All.” <https://arxiv.org/abs/2305.05665>  
[13] SPACE, “Nexuss-AO reproducible evaluation design,” local research note. <https://github.com/> (local source: `/home/ubuntu/Space/research/evaluation.md`)  
[14] Wang et al., “MoVE-KD: Knowledge Distillation for VLMs with Mixture of Visual Encoders.” <https://arxiv.org/html/2501.01709v1>  
[15] Rolnick et al., “Experience Replay for Continual Learning.” <https://arxiv.org/html/1811.11682>  
[16] Stanford CRFM, HELM, VHELM, AHELM, and HEIM. <https://crfm.stanford.edu/helm/> · <https://crfm.stanford.edu/helm/vhelm/latest/> · <https://crfm.stanford.edu/helm/audio/latest/> · <https://crfm.stanford.edu/helm/heim/latest/>  
[17] Hendrycks et al., “Measuring Massive Multitask Language Understanding.” <https://arxiv.org/abs/2009.03300>  
[18] Yue et al., “MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI.” <https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html>  
[19] ICDAR Document Visual Question Answering evaluation (ANLS). <https://rrc.cvc.uab.es/?ch=17&com=tasks>  
[20] AudioCaps and TextCaps benchmark descriptions. <https://audiocaps.github.io/> · <https://ai.meta.com/research/publications/textcaps-a-dataset-for-image-captioning-with-reading-comprehension/>  
[21] SPACE, “Nexuss-AO reproducible evaluation design,” release artifact and gate specification. Local source: `/home/ubuntu/Space/research/evaluation.md`  
[22] Hugging Face Hub download and gated-model documentation. <https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download> · <https://huggingface.co/docs/hub/en/models-gated>  
[23] GitHub Actions matrix, artifact, and cache documentation. <https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/running-variations-of-jobs-in-a-workflow> · <https://docs.github.com/en/actions/tutorials/store-and-share-data> · <https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching>  
[24] GitHub Actions security hardening and larger-runner documentation. <https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions> · <https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners>  
[25] QwenLM, official Qwen3-ASR repository. <https://github.com/QwenLM/Qwen3-ASR>

> **Bottom line:** ship the auditable late-fusion hub now; train adapters and a common decoder only through gated, replay-protected experiments; keep the specialists and their fallbacks; and use CI to prove reproducible inference plumbing, not to imply that smoke tests are training or broad quality evidence.
