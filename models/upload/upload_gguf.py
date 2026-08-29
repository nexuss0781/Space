#!/usr/bin/env python3
"""Upload benchmarked GGUF models into a single HuggingFace repo.

Supports text (single GGUF) and vision (LLM GGUF + mmproj GGUF) entries.

Environment:
  HF_TOKEN  - write token (GitHub Actions secret)
  HF_USERNAME - hub username/org, e.g. Nexuss0781
  TARGET_REPO - one repo that will hold all models, e.g. SPACE
  DELETE_REPOS - optional comma-separated repo_ids to delete first
"""

import fnmatch
import os
import pathlib

from huggingface_hub import HfApi, snapshot_download

MODELS = [
    {
        "kind": "text",
        "source": "QuantFactory/SmolLM2-360M-Instruct-GGUF",
        "pattern": "*Q4_K_M*.gguf",
        "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "license": "apache-2.0",
    },
    {
        "kind": "text",
        "source": "QuantFactory/Qwen2.5-3B-Instruct-GGUF",
        "pattern": "*Q4_K_M*.gguf",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "license": "qwen-research",
    },
    {
        "kind": "vision",
        "source": "ggml-org/SmolVLM2-2.2B-Instruct-GGUF",
        "pattern": "SmolVLM2-2.2B-Instruct-Q4_K_M.gguf",
        "mmproj": "mmproj-SmolVLM2-2.2B-Instruct-f16.gguf",
        "base_model": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "license": "apache-2.0",
    },
    {
        "kind": "vision",
        "source": "ggml-org/SmolVLM2-500M-Video-Instruct-GGUF",
        "pattern": "SmolVLM2-500M-Video-Instruct-Q8_0.gguf",
        "mmproj": "mmproj-SmolVLM2-500M-Video-Instruct-f16.gguf",
        "base_model": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "license": "apache-2.0",
    },
    {
        "kind": "audio",
        "source": "ggml-org/Qwen3-ASR-0.6B-GGUF",
        "pattern": "Qwen3-ASR-0.6B-Q8_0.gguf",
        "mmproj": "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf",
        "base_model": "Qwen/Qwen3-ASR-0.6B",
        "license": "apache-2.0",
    },
    {
        "kind": "audio",
        "source": "ggml-org/Qwen3-ASR-1.7B-GGUF",
        "pattern": "Qwen3-ASR-1.7B-Q8_0.gguf",
        "mmproj": "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
        "base_model": "Qwen/Qwen3-ASR-1.7B",
        "license": "apache-2.0",
    },
]


def main():
    token = os.environ["HF_TOKEN"]
    username = os.environ["HF_USERNAME"]
    target_repo = os.environ["TARGET_REPO"]

    api = HfApi(token=token)
    who = api.whoami()
    print(f"[upload] authenticated as {who['name']}")

    for repo_id in [
        r.strip() for r in os.environ.get("DELETE_REPOS", "").split(",") if r.strip()
    ]:
        print(f"[upload] deleting {repo_id} ...")
        api.delete_repo(repo_id=repo_id, repo_type="model", missing_ok=True)

    target_id = f"{username}/{target_repo}"
    api.create_repo(repo_id=target_id, repo_type="model", exist_ok=True)
    print(f"[upload] target repo ok: https://huggingface.co/{target_id}")

    uploaded = []
    for m in MODELS:
        patterns = [m["pattern"]]
        if m.get("mmproj"):
            patterns.append(m["mmproj"])
        local = snapshot_download(
            repo_id=m["source"],
            allow_patterns=patterns,
            local_dir=f"models/{m['source'].split('/')[-1]}",
        )
        files = list(pathlib.Path(local).rglob("*.gguf"))
        model_gguf = next(f for f in files if fnmatch.fnmatch(f.name, m["pattern"]))
        mmproj_path = next(
            (f for f in files if m.get("mmproj") and f.name == m["mmproj"]), None
        )
        size_gb = round(model_gguf.stat().st_size / 1e9, 2)
        print(f"[upload] {m['kind']}: {model_gguf.name} ({size_gb} GB) -> {target_id}")

        api.upload_file(
            path_or_fileobj=str(model_gguf),
            path_in_repo=model_gguf.name,
            repo_id=target_id,
            commit_message=f"Upload {model_gguf.name}",
        )
        extra = ""
        if mmproj_path is not None:
            api.upload_file(
                path_or_fileobj=str(mmproj_path),
                path_in_repo=mmproj_path.name,
                repo_id=target_id,
                commit_message=f"Upload {mmproj_path.name}",
            )
            extra = f" + `{mmproj_path.name}`"
        uploaded.append(
            f"- **{m['base_model']}** - `{model_gguf.name}`{extra} - {size_gb} GB"
            f" - license: {m['license']} - {m['kind']}"
        )

    readme = f"""---
license: other
pipeline_tag: image-text-to-text
tags:
  - text-generation
  - image-text-to-text
quantized_by: Nexuss0781
---

# SPACE

Single repository holding the GGUF weights (llama.cpp) benchmarked in the
Space GitHub Actions pipeline. Vision and audio models include their `mmproj`
encoder and run with `llama-mtmd-cli`.

## Models

Text (`text-generation`):

""" + "\n".join(u for u in uploaded if u.rsplit(" ", 1)[1] == "text") + """

Vision (`image-text-to-text`, use alongside the `mmproj`):

""" + "\n".join(u for u in uploaded if u.rsplit(" ", 1)[1] == "vision") + """

Audio (`automatic-speech-recognition`, use alongside the `mmproj`):

""" + "\n".join(u for u in uploaded if u.rsplit(" ", 1)[1] == "audio") + """

Benchmarks: https://github.com/nexuss0781/Space
"""
    readme_path = pathlib.Path("README.md")
    readme_path.write_text(readme)
    commit = api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=target_id,
        commit_message="Add model card",
    )
    print(f"[upload] README commit: {commit.commit_url}")
    print(f"[upload] DONE: https://huggingface.co/{target_id}")


if __name__ == "__main__":
    main()