# GitHub Actions research for CPU/GPU model experiments

**Repository inspected:** `/home/ubuntu/Space`  
**Scope:** Existing Actions workflows and benchmark scripts, plus practical patterns for CPU/GPU matrices, caches, artifacts, and safe model downloads.  
**Recommendation status:** The repository should add a small, always-on smoke workflow and keep the existing full benchmark workflows explicitly manual or scheduled. A smoke run is an inference/integrity check; it must not be presented as training or as evidence that multi-billion-parameter weights were trained.

## 1. Current repository inventory

| Area | Current behavior | CI implication |
|---|---|---|
| `.github/workflows/audio-benchmark.yml` | A five-entry `include` matrix runs on `ubuntu-latest`, with `fail-fast: false`, a 45-minute timeout, and one report artifact per candidate. A second job downloads `audio-*` artifacts and merges them. | The matrix/fan-in shape is useful, but every candidate downloads sizeable GGUF and mm-project files and runs ten audio tasks. A single failed matrix job currently causes the normal `merge` job (which uses `needs: benchmark`) to be skipped. |
| `.github/workflows/vision-benchmark.yml` | Same llama.cpp build/cache pattern, currently one 2.2B candidate, ten image tasks, and a merge job. | Suitable as a quality/benchmark workflow, not as a pull-request smoke test. |
| `.github/workflows/smollm3-benchmark.yml` | One matrix entry named `qwen2.5-3b`; caches a GGUF directory under a static `v1` key, downloads a broad `*Q4_K_M*.gguf` snapshot, forces llama.cpp, and runs ten prompts. | The workflow title and defaults are inconsistent with the matrix entry. The model download and ten-prompt run are too heavy for an always-on check. |
| `upload-models.yml`, `update-card.yml` | Manual workflows use a Hugging Face write token from a repository secret. | Keep publishing in separate, manually approved workflows; never expose write credentials to a pull-request smoke job. |
| `benchmarks/audio/benchmark_audio.py` and `benchmarks/vision/benchmark_vision.py` | Each calls `snapshot_download` twice, allows exact filenames, but does not pin `revision`, pass a token explicitly, or expose a task limit. Each invokes the binary for every task and records a nonzero return code without failing the overall process. | Add a smoke/task-limit mode and explicit download revision. Treat a failed inference process as a failed smoke test, rather than only recording `rc` in a report. |
| `benchmarks/smollm3/benchmark_smollm3.py` | Has a GGUF path and a Transformers path. It downloads a pattern, selects a Q4_K_M file, and limits prompts to ten, but has no one-prompt smoke mode. | Make smoke mode force a tiny, quantized public fixture and a short generation limit. Do not let memory heuristics accidentally choose a full Transformers load in CI. |
| `README.md` and reports | Baselines already show that 3B–8B candidates take substantial time/RAM; the selected “micro” models are approximately 360M text, 500M vision, and 0.6B audio. | Use the selected micro model (or a still smaller test fixture) for CI smoke. Keep quality candidates in a separately invoked benchmark. |

Additional current concerns are that llama.cpp is cloned from its moving default branch, the llama.cpp build cache key is static (`llamacpp-mtmd-v2`), and Python dependencies are installed without a lock/requirements file. A cache can therefore contain a build made from a different source revision or toolchain than the current job.

## 2. Patterns supported by the documentation

### Matrix experiments

GitHub's matrix strategy creates one job for each combination of matrix values. `include` is the right way to describe irregular experiments (for example, a CPU row using a portable build and a GPU row using CUDA) without creating invalid Cartesian products [1]. Set `fail-fast: false` for benchmark comparisons so one candidate does not cancel the others. Use `max-parallel` to control concurrency and resource pressure. `continue-on-error` can be tied to a matrix field for an explicitly experimental GPU row, but the CPU smoke row should remain required [1].

A practical matrix should carry behavior, not just a model name:

```yaml
strategy:
  fail-fast: false
  max-parallel: 2
  matrix:
    include:
      - name: text-micro
        modality: text
        runner: ubuntu-latest
        backend: cpu
        model_repo: <public-repo>
        model_file: <small-q4-file>
        revision: <40-char-commit>
        experimental: false
      - name: text-micro-cuda
        modality: text
        runner: <org-gpu-runner-label>
        backend: cuda
        model_repo: <public-repo>
        model_file: <small-q4-file>
        revision: <40-char-commit>
        experimental: true
```

`<org-gpu-runner-label>` is intentionally a configured organization/self-hosted or larger-runner label, not a guessed universal label. Runner availability and labels are organization-specific; GitHub documents larger runners as an organization/enterprise capability [5]. Standard `ubuntu-latest` should be treated as CPU-only for this project. A GPU row should therefore be in a manually enabled workflow or be skipped unless the required label is configured. Do not claim GPU coverage when the job merely reports `cuda_available: false` on a CPU runner.

For llama.cpp, build CPU and CUDA variants as distinct cache entries. The CPU build can retain `-DGGML_NATIVE=OFF` for portability. A CUDA build should run only where the CUDA compiler/toolkit and a visible GPU are guaranteed by the runner image, and should verify `nvidia-smi`, the built backend, and a one-inference smoke before measuring throughput.

### Artifacts and fan-in

Artifacts are the appropriate handoff between matrix jobs and a merge/report job; GitHub explicitly supports uploading test output, downloading it in a dependent job, and setting a retention period [2]. Give every matrix job a unique artifact name containing modality, model, backend/device, and (if needed) run ID. Upload only Markdown/JSON summaries and bounded raw logs—not model weights, HF caches, credentials, or the build directory.

Recommended shape:

```yaml
- name: Upload smoke result
  if: always()
  uses: actions/upload-artifact@v4 # pin this action to a reviewed full SHA in production
  with:
    name: smoke-${{ matrix.modality }}-${{ matrix.name }}-${{ matrix.backend }}
    path: results/${{ matrix.modality }}/${{ matrix.name }}/
    if-no-files-found: error
    retention-days: 14
```

Use `if: always()` when preserving diagnostics after a failed inference, but make the test step itself fail on a bad return code or missing expected output. A merge job should use `needs: smoke`, `if: ${{ always() }}`, download with a pattern, and summarize failed/missing rows rather than silently converting an incomplete matrix into a successful benchmark. For safer fan-in, download each artifact into its own directory (`merge-multiple: false`) or normalize names before merging; do not rely on same-name files from multiple jobs. GitHub's artifact v4 behavior is immutable, so a downstream job must upload a new artifact name rather than trying to overwrite an earlier one [2]. The download action also validates the artifact digest and warns if it differs [2].

### Dependency and build caching

Use caching for reproducible dependencies/build products, not as a model registry. GitHub's cache action searches the exact key, then prefix/restore keys, creates a new cache only after a successful miss, and does not mutate an existing cache [3]. Keys should include all inputs that affect usability:

```yaml
key: >-
  ${{ runner.os }}-${{ runner.arch }}-llamacpp-${{ matrix.backend }}-
  ${{ env.LLAMACPP_REV }}-${{ hashFiles('ci/requirements-ci.txt') }}
restore-keys: |
  ${{ runner.os }}-${{ runner.arch }}-llamacpp-${{ matrix.backend }}-
  ${{ runner.os }}-${{ runner.arch }}-llamacpp-
```

For Python packages, add a committed requirements/lock file and use `actions/setup-python`'s pip cache or `actions/cache` keyed by that file. For llama.cpp, pin a commit or release and key by that revision plus backend, OS, architecture, compiler/toolchain inputs. A static key such as `llamacpp-mtmd-v2` is unsafe for reproducibility and can restore an executable built from an unrelated source state; the existing executable probe is useful as a fallback, but it is not a substitute for an input-sensitive key.

A Hugging Face cache can reduce repeated public-model downloads, but scope it narrowly to the Hub blob cache and a pinned revision. Do not cache `$HF_HOME` wholesale: Hugging Face stores its token under `HF_HOME` by default, while `HF_HUB_CACHE` is the model/dataset repository cache [6]. GitHub warns that cache contents can be read by users who can create pull requests, so never put a token or other secret in a cache path [3]. Public, immutable model blobs may be cached; restricted-model downloads should either use a non-persistent temporary directory or a carefully reviewed cache that demonstrably contains no credentials.

### Safe and bounded model downloads

The current `snapshot_download` use of exact `allow_patterns` for audio/vision is a good start, but the revision is not pinned. Hugging Face documents both `revision` (including a commit hash), `token`, exact `hf_hub_download`, and `snapshot_download` `allow_patterns`/`ignore_patterns` [8]. For CI:

1. Prefer a tiny, public, permissively licensed fixture for the always-on smoke. Put the repo ID, exact file name, commit SHA, expected byte size, and optionally SHA-256 in a versioned matrix/lock file.
2. Call `snapshot_download(..., revision=PINNED_SHA, allow_patterns=[EXACT_FILE], token=os.environ.get("HF_TOKEN") or None, cache_dir=...)`; never use a floating `main`/`master` revision for a test that is expected to be reproducible.
3. Validate that exactly the expected file exists, its size is within the declared bound, and (where feasible) its SHA-256 matches. Do not select “the first Q4 file” from a broad wildcard; a repository can add a larger or differently named file later.
4. Use `HF_TOKEN` only in the download step's environment. Do not call `hf auth login`, write a token into the workspace, echo the token, or include it in an artifact. Hugging Face says gated files require an authenticated user/token and access may be granted or revoked by the model author [7].
5. Run gated/private candidates only in a separate `workflow_dispatch` workflow with an approved environment secret. Do not pass `HF_TOKEN` to workflows triggered by untrusted fork pull requests. For public smoke fixtures, omit the secret entirely.
6. Set `HF_HOME`/`HF_HUB_CACHE` before Python imports `huggingface_hub`; the library reads these variables at import time [6]. Use a temporary download directory and bounded cache on PR jobs, and clean it before artifact upload.
7. Add network/download timeouts and fail clearly on authentication, revision, checksum, or size errors. A cache hit is an optimization, not proof that the expected model was downloaded.

A safe helper should resemble:

```python
from hashlib import sha256
from pathlib import Path
import os
from huggingface_hub import hf_hub_download

repo = os.environ["MODEL_REPO"]
revision = os.environ["MODEL_REVISION"]       # full commit SHA from lock data
filename = os.environ["MODEL_FILE"]
path = Path(hf_hub_download(
    repo_id=repo,
    filename=filename,
    revision=revision,
    token=os.environ.get("HF_TOKEN") or None,
    cache_dir=os.environ["HF_HUB_CACHE"],
))
assert path.name == filename
assert path.stat().st_size <= int(os.environ["MAX_MODEL_BYTES"])
# Compare sha256(path) with a checked-in expected digest when available.
```

`hf_hub_download` is preferable when one exact file is needed; `snapshot_download` remains appropriate for the audio/vision model-plus-projector pair when both exact patterns and a pinned revision are supplied [8].

## 3. Proposed implementation: a truthful smoke workflow

Add `.github/workflows/model-smoke.yml` (or equivalent) with these properties:

- Triggers on `pull_request` and relevant pushes, plus `workflow_dispatch` with an explicit `run_gpu`/`full` option. The PR path uses only public tiny fixtures and no secrets.
- Declares `permissions: contents: read`, a short `concurrency` group that cancels superseded PR runs, and a 10–15 minute job timeout.
- Uses a small matrix of **one CPU micro row by default** and an optional GPU row only when a configured GPU runner is selected. Keep `fail-fast: false`, `max-parallel: 1` or `2`, and make GPU `experimental: true` only if its absence should not block merges.
- Installs locked, minimal dependencies. Builds or installs the CPU backend once per cache key; for a GPU row, builds the CUDA backend and verifies the device before running.
- Downloads one exact quantized fixture at a pinned Hub commit. If audio/vision code paths are smoke-tested too, use one tiny sample/task for each and exact model/projector files. Do not download the existing 3B–8B quality candidates in the PR job.
- Runs `--smoke`/`--limit 1` with `max_new_tokens` around 8–16, deterministic seed, short context, and no optimizer, gradient, checkpoint, or weight-update code. The output should be labeled “inference smoke,” and the report should include model revision, file digest, backend, device, duration, peak RSS, and return code.
- Fails when the process exits nonzero, the expected report is missing, output is empty, the model is above the declared size bound, or a CUDA row cannot see a GPU. It may record a throughput number, but must not call it a training result.
- Uploads bounded JSON/Markdown/log artifacts per matrix row with 14-day retention, excluding model files and caches. A final summary downloads all row artifacts with `if: always()` and reports pass/fail/missing rows. Full ten-task comparisons remain manual or scheduled and can retain the existing merge-report approach.

Illustrative workflow skeleton (action references should be pinned to reviewed full commit SHAs before merging):

```yaml
name: Model inference smoke
on:
  pull_request:
    paths: ['benchmarks/**', '.github/workflows/**', 'ci/**']
  push:
    branches: [main]
    paths: ['benchmarks/**', '.github/workflows/**', 'ci/**']
  workflow_dispatch:
    inputs:
      run_gpu:
        description: 'Run the optional configured GPU smoke row'
        type: boolean
        default: false
permissions:
  contents: read
concurrency:
  group: model-smoke-${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true
jobs:
  smoke:
    if: ${{ matrix.backend == 'cpu' || inputs.run_gpu == true }}
    runs-on: ${{ matrix.runner }}
    timeout-minutes: 15
    strategy:
      fail-fast: false
      max-parallel: 1
      matrix:
        include:
          - name: text-micro
            backend: cpu
            runner: ubuntu-latest
            model_repo: <public-repo>
            model_file: <tiny-q4-file>
            revision: <full-commit-sha>
          - name: text-micro-cuda
            backend: cuda
            runner: <configured-gpu-label>
            model_repo: <public-repo>
            model_file: <tiny-q4-file>
            revision: <full-commit-sha>
    steps:
      - uses: actions/checkout@v4 # pin SHA
      - uses: actions/setup-python@v5 # pin SHA
        with:
          python-version: '3.11'
          cache: pip
          cache-dependency-path: ci/requirements-ci.txt
      - name: Install smoke dependencies
        run: python -m pip install -r ci/requirements-ci.txt
      - name: Download exact fixture
        env:
          MODEL_REPO: ${{ matrix.model_repo }}
          MODEL_FILE: ${{ matrix.model_file }}
          MODEL_REVISION: ${{ matrix.revision }}
          HF_HUB_CACHE: ${{ runner.temp }}/hf-hub
        run: python ci/download_fixture.py
      - name: Verify device (CUDA row only)
        if: ${{ matrix.backend == 'cuda' }}
        run: nvidia-smi
      - name: Run one inference smoke
        env:
          MODEL_FILE: ${{ matrix.model_file }}
          BACKEND: ${{ matrix.backend }}
        run: python benchmarks/smoke.py --backend "$BACKEND" --model "$MODEL_FILE" --limit 1 --max-new-tokens 16 --seed 42 --report-dir "results/${{ matrix.name }}"
      - name: Upload result
        if: always()
        uses: actions/upload-artifact@v4 # pin SHA
        with:
          name: smoke-${{ matrix.name }}
          path: results/${{ matrix.name }}/
          if-no-files-found: error
          retention-days: 14
```

The exact model identifiers and revisions should be selected by the project maintainer and checked into a small lock file rather than copied from this illustrative skeleton. If no suitable tiny multimodal fixture is available, a deterministic local fake/stub backend can validate argument parsing, task selection, report schema, and artifact plumbing; that is preferable to silently downloading a huge model. It must be labeled as a harness smoke, not a model-quality result.

## 4. Changes to the existing full benchmark workflows

Keep the full workflows useful, but make them reproducible and opt-in:

- Add a pinned llama.cpp revision, and include it plus OS/architecture/backend/compiler in the build-cache key. Prefer a committed dependency file for Python installs.
- Add matrix fields for `revision`, `expected_sha256`, and `max_model_bytes`; pass them to a shared download helper. Keep `allow_patterns` exact and avoid broad `*Q4_K_M*.gguf` selection.
- Add `--limit`/`--smoke` to all benchmark scripts. Full workflows use ten tasks; smoke uses one. Add `--fail-on-error` (or make nonzero subprocess return codes fatal by default) so reports cannot imply successful evaluation after a failed process.
- Use a unique artifact name per modality/candidate/backend and `retention-days`; do not upload `models/`, `gguf/`, `HF_HOME`, or caches. Let the merge job run with `if: always()` and explicitly mark incomplete candidates.
- Separate CPU and GPU workflow files or matrix rows. The GPU workflow should be manual/scheduled, use an organization-configured runner label, assert GPU visibility, and report driver/runtime details. An ordinary `ubuntu-latest` run is not a GPU benchmark.
- Keep `upload-models.yml` and `update-card.yml` manual, add least-privilege job permissions, and consider a protected environment for `HUGGINGFACE_TOKEN`. Pin third-party actions to full commit SHAs; GitHub's security guidance identifies this as the immutable action reference and recommends least-privilege tokens [4].

## 5. Decision summary

The shortest honest path is **CPU inference smoke on a tiny pinned quantized fixture, one prompt/task, no secret, and result-only artifacts**, with an optional separately triggered GPU smoke on a configured runner. Caches should accelerate dependencies and pinned public blobs but never hold credentials; artifacts should preserve bounded diagnostics but never package weights. The existing ten-task, multi-candidate reports remain valuable benchmark experiments, but they belong behind manual/scheduled controls and should be labeled as inference performance measurements—not training runs.

## References

[1] GitHub Docs, “Running variations of jobs in a workflow,” matrix `include`, failure handling, and `max-parallel`: <https://docs.github.com/actions/writing-workflows/choosing-what-your-workflow-does/running-variations-of-jobs-in-a-workflow>

[2] GitHub Docs, “Store and share data with workflow artifacts,” upload/download, retention, job handoff, immutability, and digest validation: <https://docs.github.com/en/actions/tutorials/store-and-share-data>

[3] GitHub Docs, “Dependency caching reference,” key matching, restore keys, immutable caches, and the warning not to cache secrets: <https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching>

[4] GitHub Docs, “Secure use reference,” least-privilege permissions, secret handling, untrusted pull requests, and pinning actions to full-length commit SHAs: <https://docs.github.com/en/actions/security-for-github-actions/security-guides/security-hardening-for-github-actions>

[5] GitHub Docs, “Using larger runners,” organization/enterprise larger-runner configuration and labels: <https://docs.github.com/en/actions/using-github-hosted-runners/about-larger-runners>

[6] Hugging Face Hub Docs, “Environment variables,” `HF_HOME`, `HF_HUB_CACHE`, `HF_TOKEN`, and import-time configuration: <https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables>

[7] Hugging Face Hub Docs, “Gated models,” access requests and token authentication for scripted downloads: <https://huggingface.co/docs/hub/en/models-gated>

[8] Hugging Face Hub Docs, “Downloading files,” `hf_hub_download`/`snapshot_download`, `revision`, `token`, exact patterns, and dry-run support: <https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download>
