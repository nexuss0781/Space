# Nexuss-AO reproducible evaluation design

**Status:** proposed evaluation protocol (2026-08-30)  
**Purpose:** replace the current ten-item smoke reports with a versioned, repeatable quality-and-systems evaluation for text, image, audio, and mixed-modality inference.

## Executive recommendation

Treat the existing ten-task reports as **smoke tests, not quality benchmarks**. Keep them as a fast deployment check, but add a pinned public benchmark suite, a versioned local challenge set, paired modality ablations, and paired bootstrap regression tests. Publish raw generations, normalized predictions, per-example scores, timing samples, hardware/software metadata, and the exact model artifacts. Do not collapse quality into tokens/second: a fast model that hallucinates, omits an image word, or changes a credential is a failed model.

The protocol below uses ideas from HELM, which is explicitly a reproducible and transparent framework and separates scenarios, adaptations, metrics, and models [1]; VHELM, which standardizes prompts, inference parameters, and metrics for vision-language models [2]; AHELM, which makes the same standardization point for audio-language models and covers ten audio-relevant aspects [3]; and HEIM, which demonstrates that image-text alignment and image quality alone miss robustness, fairness, safety, multilinguality, and efficiency [4].

## 1. What is evaluated

### 1.1 Fixed model variants and interfaces

Evaluate each Nexuss-AO release as a named artifact, not merely as a repository branch. The manifest must record model and projector filenames, SHA-256, quantization, tokenizer, chat template, context length, maximum output tokens, sampling parameters, commit IDs, and any downloaded dependencies. At minimum run the selected local specialists (SmolLM2-360M-Instruct text, SmolVLM2-500M-Video-Instruct vision, and Qwen3-ASR-0.6B audio) and the Nexuss-AO fused/late-fusion system under test. If a quality model is included, report it separately rather than mixing its scores with the micro model.

Every example receives a stable `example_id`; data manifests are immutable, content-addressed, and stored with license and provenance. Store the exact bytes or a download URL plus SHA-256. Never silently replace a benchmark's test set with a newly released version.

### 1.2 Required execution modes

Run four modes where the interface supports them:

1. **Text-only:** text prompt to text answer.
2. **Image-plus-text:** one image and a text question to text answer.
3. **Audio-plus-text:** one audio clip and a text question to text answer.
4. **Mixed:** image plus audio plus text, and separately image plus text plus an audio-derived transcript, so the value of raw audio versus transcription is measurable.

For every multimodal item run matched ablations: remove image, remove audio, replace audio with its gold transcript, replace image with a blank image, and shuffle the media from another item. The shuffled-media condition is a leakage/sensitivity test; performance close to it indicates that the model is answering from the prompt or language prior rather than the media.

## 2. Datasets and tasks

The public suite should be run in full when practical. For continuous integration, use a deterministic stratified slice whose IDs are checked into the repository; run the full suite nightly or before release.

| Modality | Benchmark and fixed task | Primary metric | Secondary metric / diagnostic |
|---|---|---|---|
| Text | **MMLU** official test, all 57 subjects, standardized zero-shot and 5-shot settings | Multiple-choice accuracy: `correct / N` | Per-subject accuracy, macro mean, worst-subject accuracy, abstention/format error rate |
| Text | **GSM8K** official test, fixed 8-shot prompt and exact answer extraction | Numeric exact match (EM), `1` only when the final normalized number equals the reference | Parse failure rate; accuracy by problem length |
| Text | **HumanEval** official problems, temperature 0, one sample | `pass@1`: fraction of problems whose generated program passes all hidden unit tests | Sandbox failure, syntax failure, timeout rate; optionally `pass@10` as a separate non-CI diagnostic |
| Image + text | **MMMU** official validation/test protocol, with the released answer parser | Multiple-choice exact-match accuracy | Per-discipline and per-image-type accuracy; shuffled-image delta |
| Image + text | **DocVQA** official test or a pinned public validation split | ANLS (Average Normalized Levenshtein Similarity) | Answer-page accuracy when page prediction is available; exact match; OCR character error rate |
| Image + text | **TextCaps** pinned evaluation split | Official caption scorer, report CIDEr and SPICE | Text-token recall and OCR word accuracy; human factuality sample |
| Audio + text | **LibriSpeech** `test-clean` and `test-other`, with a pinned 16-kHz preprocessing and normalizer | WER | Word accuracy, insertion/deletion/substitution rates, real-time factor (RTF) |
| Audio + text | **AudioCaps** official test split (about 46K audio-caption pairs overall) | CIDEr (official caption scorer) | SPICE, CLIP/audio-text similarity if the scorer is pinned, human faithfulness sample |
| Audio + text | AHELM public tasks, or a versioned subset containing perception and conversational reasoning | Task-appropriate exact match / accuracy, macro-averaged by scenario | Robustness, multilinguality, safety, and audio-ablation deltas |
| Mixed | **MMMU** items augmented with a second, unrelated or supporting audio track from a versioned local set | Answer exact-match accuracy | Evidence/grounding accuracy and modality-shuffle delta |
| Mixed | Versioned local set of 30 image+audio+text cases: 10 object/event grounding, 10 temporal ordering, 10 contradiction resolution | Joint exact match: all required fields correct; score also each field separately | Macro F1 on entities/events, temporal relation accuracy, contradiction detection F1 |
| Mixed | 10 paired “same scene” cases with audio and image referring to the same event | **Synergy**: `score(image+audio) - max(score(image-only), score(audio-only))` | Gold-transcript upper-bound gap and raw-audio-vs-transcript gap |

MMLU's primary paper defines 57 tasks spanning mathematics, history, computer science, law, and other subjects and uses multitask accuracy [5]. MMMU is appropriate for image-plus-text reasoning because it contains 11.5K college-level questions across six disciplines, 30 subjects, 183 subfields, and heterogeneous media such as charts, maps, tables, music sheets, and chemical structures [6]. DocVQA's official evaluation specifies ANLS and states that answers are case-insensitive but space-sensitive; it also defines page accuracy where page prediction is submitted [7]. TextCaps specifically tests recognizing text, relating it to visual context, and deciding what text to copy or paraphrase, with 145K captions for 28K images [8]. AudioCaps contains about 46K human-written audio-caption pairs collected on AudioSet audio [9].

### 2.1 Exact scoring definitions

All scores are computed from raw predictions by a checked-in scorer. No model-generated self-rating is a primary metric.

* **Accuracy:** `A = (1/N) * sum_i 1[pred_i == gold_i]`. For multiple choice, extract exactly one option after stripping a configured answer prefix. A missing, multiple, or unparseable option is wrong and is counted separately as a format failure.
* **Normalized text EM:** lowercase, Unicode NFKC, normalize whitespace, remove a terminal period, and apply only task-declared numeric normalization. Do not remove meaningful punctuation from code or identifiers. GSM8K uses the final parsed numeric answer; if parsing fails, it is wrong.
* **WER:** after applying the benchmark's pinned normalizer to hypothesis and reference, `WER = (S + D + I) / N`, where `S`, `D`, and `I` are word substitutions, deletions, and insertions and `N` is the number of reference words. Report the components, not only WER. MLCommons describes WER as word-level Levenshtein edits and documents lowercasing, alphabetic normalization, and the importance of controlling punctuation, abbreviations, and spelling [10].
* **Word accuracy:** `1 - (S + D + I)/N`; report it only with the normalization and clipping rule stated. WER can exceed 1; word accuracy is clipped to zero for display, while raw edit counts remain available.
* **ANLS:** for prediction `p` and one or more references `g`, calculate `d = Lev(p,g) / max(len(p),len(g))`; score `max(0, 1-d)` when `d <= 0.5`, otherwise `0`, then take the maximum over references and average over questions. This follows the DocVQA convention [7].
* **CIDEr/SPICE:** run the official pinned COCO-caption/AudioCaps scorer and report corpus CIDEr and SPICE, not an improvised token overlap. SPICE measures recovery of objects, attributes, and relations in a scene graph [11]. For captions, also retain per-example scores because corpus averages can conceal catastrophic omissions.
* **HumanEval:** execute generated Python in a network-disabled, resource-limited sandbox with the official hidden tests. `pass@1` is the fraction of prompts with a passing first sample; syntax errors, exceptions, timeouts, and sandbox violations fail.
* **Mixed joint score:** each item declares required slots (for example `event=doorbell`, `count=2`, `order=[bell,door]`). Normalize categorical slots by exact match; score free-text evidence with token F1; score temporal relations by exact relation. `joint_exact` is 1 only when every required slot is correct. The headline mixed score is the macro mean of joint exact over the three balanced strata, with slot-level F1 and accuracy reported alongside it.
* **Factuality review:** for 100 randomly sampled image/audio captions, two blinded raters mark each atomic claim supported, contradicted, or unverifiable by the media. Report supported-claim precision and contradiction rate with Cohen's kappa. This is a release diagnostic, not a substitute for the public benchmark scorer.

## 3. Reproducible run protocol

Pin Ubuntu/container image, Python, llama.cpp or Transformers commit, compiler flags, BLAS/OpenMP settings, CPU governor, thread count, and environment variables. Record CPU model, physical/logical cores, RAM, GPU and VRAM if present, kernel, driver, storage type, and background load. Run on an otherwise idle host with `OMP_NUM_THREADS=4` (or the declared deployment value), affinity pinned to the same physical cores, and no concurrent models.

Use the same prompt templates for all candidates in a comparison. Set `temperature=0`, `top_p=1`, `top_k=0` where supported, fixed seed `42`, fixed max output tokens per task family (text 256, image 128, audio 128, mixed 256), and fixed context size. If a model requires sampling or cannot be deterministic, run five seeds (`42, 43, 44, 45, 46`) and report the mean and standard deviation; never compare one stochastic run against one deterministic run.

For each model and task family:

1. Verify manifest hashes and run a zero-output load check.
2. Do **three warm-up examples** that are excluded from quality and latency statistics.
3. Run every measured example once in a fixed, published order, then repeat the whole order four more times (five measured repetitions). Reset the process between cold-start repetitions; use one persistent process for warm latency, and label the two results separately.
4. Save the exact request, media hash, decoded output, exit code, error stream, and scorer input for every attempt.
5. Fail the run rather than silently dropping an example. A timeout is a failed example and a resource failure is a failed run.

A release result is valid only if at least 95% of required examples complete, every failed example is reported, and the same artifact can be re-run from the manifest. Benchmark test data must never be used for prompt tuning. Any prompt change increments the evaluation protocol version.

## 4. Latency, throughput, and RAM

Report **cold** and **warm** measurements separately. “End-to-end” starts immediately before request submission and ends after output decoding and process cleanup; it includes media decode/preprocessing, model load for cold runs, queueing, generation, and serialization. Also report a **model-only** interval after media preprocessing so that a preprocessing optimization is not misrepresented as a model speedup.

For each example, record:

* load time (cold only), time to first token (TTFT), prompt/prefill time, decode time, end-to-end latency, output token count, and exit code;
* output-token throughput `output_tokens / decode_seconds` and end-to-end throughput `output_tokens / e2e_seconds`;
* p50, p90, and p95 latency across examples and repetitions, plus mean and standard deviation; never report only a mean;
* audio RTF `e2e_seconds / audio_duration_seconds` and audio seconds processed per wall second;
* image latency per image and, if batching is supported, latency and throughput at batch sizes 1, 2, 4, and 8;
* process peak RSS from `/usr/bin/time -v` for the complete child process, and resident/PSS samples from `/proc/<pid>/smaps_rollup` at 10 Hz when available; report model load RSS, steady-state RSS, peak RSS, and peak system used RAM;
* CPU utilization and package power where available, and GPU memory peak plus GPU utilization for GPU runs.

The current scripts already record wall time, llama.cpp-reported eval tokens/second, and peak RSS through `/usr/bin/time -v`, which is useful continuity. However, token throughput is not comparable across modalities or models when output lengths differ, and the present text runner estimates TTFT by performing a second one-token generation. The new harness must measure TTFT in one instrumented request (or clearly label the extra-request estimate). The current audio and vision runners launch one subprocess per item, so their wall time includes process startup and model loading; report this as cold end-to-end, not as steady-state inference.

## 5. Regression gates

Choose a pinned **baseline** artifact and retain its per-example predictions and timing samples. Gates are applied to paired examples, not just rounded leaderboard means. Use 10,000 paired bootstrap resamples stratified by task family and report a 95% confidence interval for candidate-minus-baseline.

### 5.1 Quality gates (release fails if any hard gate fails)

* MMLU, MMMU, GSM8K, HumanEval pass@1, image ANLS, audio caption CIDEr, LibriSpeech WER, and mixed joint exact each must have a lower/upper confidence bound within the limits below: accuracy-like scores **no more than 1.0 percentage point lower**; ANLS/CIDEr/SPICE **no more than 3% relative lower**; WER **no more than 5% relative higher**.
* The macro mean of the four modality headline scores must not fall by more than **0.5 percentage points** after scores are converted to “higher is better” (`1-WER` for ASR). No individual modality may be hidden by the macro average.
* Any safety/toxicity or contradiction-rate diagnostic must not worsen by more than **1 percentage point absolute**. A statistically significant increase in harmful completions or media contradiction rate is a hard fail regardless of aggregate score.
* Format failures must be **<=1%** for closed-answer tasks and **zero** on credentials/code-safety test cases. A crash, timeout, OOM, missing output, or sandbox escape on any mandatory case is a hard fail until triaged.
* Mixed synergy must not decline by more than **0.02 absolute**, and shuffled-media performance must remain at least **5 percentage points below** the true-media score. If shuffled media improves the score, flag a grounding regression.

A gate is considered failed when the bootstrap confidence interval crosses beyond the limit in the bad direction; for small slices, use the full stratified nightly set rather than making a decision from ten items. A quality improvement does not waive a hard safety, grounding, crash, or format failure.

### 5.2 Systems gates

Relative to baseline under identical hardware and run mode:

* warm p50 and p95 end-to-end latency: **<= +10% and <= +15%**, respectively;
* cold load-plus-first-request latency: **<= +20%**;
* peak RSS and peak PSS: **<= +10%**; GPU VRAM, when applicable, **<= +10%**;
* audio RTF must not increase by more than **10%** and must remain **<1.0** for a real-time product target;
* throughput must not fall by more than **10%** at the declared batch size; and
* five repetitions must have zero unexplained process failures and coefficient of variation **<=10%** for warm p50 latency. If variance exceeds this, rerun after investigating thermal throttling, CPU frequency, background load, and cache effects; do not cherry-pick the fastest run.

For a deliberately quality-first release, the owner may approve a systems exception, but the release notes must state the exact tradeoff and include the old and new measurements. There are no exceptions for silent quality regressions, unsafe output, or loss of audit artifacts.

## 6. What the existing ten-task reports can and cannot establish

The repository reports are valuable deployment smoke tests: they identify a repeatable CPU machine, model file sizes, wall time, generated tokens, tokens/second, and peak RSS. They show a useful initial tradeoff, including approximately 57.7 tok/s for the selected 360M text model, 67.2 tok/s for the selected 500M vision model, and 37.2 tok/s for the selected 0.6B ASR model on the stated four-core host. They do **not** establish broad model quality.

**Sample size and representativeness.** Each modality has only ten hand-authored items. The audio set is ten short, clean, apparently synthetic English clips; it does not measure accents, noise, overlap, multilingual speech, diarization, long-form drift, timestamps, or realistic conversational reasoning. The image set is nine open-ended descriptions plus one OCR image; it has no balanced object/count/spatial/diagram/document distribution, no independent reference captions, and no held-out ground truth scorer. The text report is ten prompts, which cannot estimate subject coverage, calibration, robustness, coding correctness, or confidence intervals. Ten observations are especially inadequate for p95 latency and regression decisions.

**Scoring deficiency.** The reports print references for audio but do not compute WER, edit components, exact numeric/identifier accuracy, or a confidence interval. Paraphrases such as “rainfall” versus “rain fall,” punctuation, and number formatting need a declared normalizer. Vision responses are not scored for factuality, OCR exactness, spatial relations, or hallucinated objects; token count is not a quality metric. The reports therefore support qualitative inspection only.

**Prompt and generation confounds.** The text runner uses `do_sample=True`, temperature 0.6, and top-p 0.95, so a single run is not deterministic. Its TTFT is an additional one-token generation rather than TTFT from the measured request. The audio and vision runners force four threads and seed 42, but task-level process startup, media decoding, model loading, and generated output length influence wall time. Different models often produce very different output lengths, making arithmetic mean tokens/second a biased comparison. Output truncation and response extraction can also hide errors.

**Memory and timing scope.** Peak RSS is measured for a child process in the audio/vision runners and for the Python process in the text runner; the scopes are not identical. RSS is not PSS, does not expose peak GPU memory, and does not separate load, prefill, decode, projector, and media preprocessing. The reports provide an average throughput and total wall time but no warm/cold distinction, p50/p95, TTFT, RTF, repeated-run variance, CPU utilization, or batch scaling.

**Leakage and reproducibility risks.** Reports do not version a dataset manifest, hash media, pin every dependency/commit, or publish a machine-readable per-example scorer output. The current vision task file has no references; the audio file has references but no scoring implementation. The ten examples can be memorized or prompt-tuned, and public image filenames alone do not prove provenance or stable bytes. These reports must not be used as evidence of generalization or as release gates.

## 7. Minimum artifact set for every release

Publish `manifest.json` (model/data/software hashes), `config.json` (prompts, seeds, sampling, context and limits), raw per-example JSONL including request/output/timing/RAM/error, normalized predictions, `scores.json` with per-task and confidence intervals, and a Markdown summary. Include the exact command, container digest, hardware snapshot, and a `FAIL` record for every missing or timed-out example. A small CI slice may gate pull requests, but a full benchmark run must gate model, quantization, projector, tokenizer, or prompt-template changes.

## References

[1] Stanford CRFM, **Holistic Evaluation of Language Models (HELM)**, reproducible framework and multimodal leaderboards: <https://crfm.stanford.edu/helm/>.  
[2] Stanford CRFM, **VHELM: Holistic Evaluation of Vision-Language Models**, standardized prompts/inference/metrics and nine aspects: <https://crfm.stanford.edu/helm/vhelm/latest/>.  
[3] Stanford CRFM, **AHELM: Holistic Evaluation of Audio-Language Models**, ten audio-language evaluation aspects and standardized inference: <https://crfm.stanford.edu/helm/audio/latest/>.  
[4] Stanford CRFM, **HEIM: Holistic Evaluation of Text-to-Image Models**, twelve deployment aspects and 33 metrics: <https://crfm.stanford.edu/helm/heim/latest/>.  
[5] Hendrycks et al., **Measuring Massive Multitask Language Understanding**, arXiv:2009.03300: <https://arxiv.org/abs/2009.03300>.  
[6] Yue et al., **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**, CVPR 2024: <https://openaccess.thecvf.com/content/CVPR2024/html/Yue_MMMU_A_Massive_Multi-discipline_Multimodal_Understanding_and_Reasoning_Benchmark_for_CVPR_2024_paper.html>.  
[7] ICDAR Document Visual Question Answering challenge, **ANLS and page-accuracy evaluation**: <https://rrc.cvc.uab.es/?ch=17&com=tasks>.  
[8] Sidorov et al., **TextCaps: A Dataset for Image Captioning with Reading Comprehension**, Meta AI/ECCV: <https://ai.meta.com/research/publications/textcaps-a-dataset-for-image-captioning-with-reading-comprehension/>.  
[9] Kim et al., **AudioCaps: Generating Captions for Audios in the Wild**, NAACL 2019: <https://audiocaps.github.io/>.  
[10] MLCommons, **Whisper: An MLPerf Inference Benchmark for Automatic Speech Recognition**, normalization, WER, accuracy, and tokens/s methodology: <https://mlcommons.org/2025/09/whisper-inferencev5-1/>.  
[11] Anderson et al., **SPICE: Semantic Propositional Image Caption Evaluation**, arXiv:1607.08822: <https://arxiv.org/html/1607.08822v1>.

**Scope caveat:** this document designs the evaluation; it does not claim that Nexuss-AO has passed any proposed gate. Scores from the existing reports are not directly comparable to the public benchmark metrics until the new harness, manifests, and scorers are run.
