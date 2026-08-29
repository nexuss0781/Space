#!/usr/bin/env python3
"""Upload benchmarked Q4_K_M GGUF models into a single HuggingFace repo.

Environment:
  HF_TOKEN  - write token (GitHub Actions secret)
  HF_USERNAME - hub username/org, e.g. Nexuss0781
  TARGET_REPO - one repo that will hold all models, e.g. SPACE
  DELETE_REPOS - optional comma-separated repo_ids to delete first
"""

import os
import pathlib

from huggingface_hub import HfApi, snapshot_download

MODELS = [
    {
        "source": "QuantFactory/SmolLM2-360M-Instruct-GGUF",
        "pattern": "*Q4_K_M*.gguf",
        "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
        "license": "apache-2.0",
    },
    {
        "source": "QuantFactory/Qwen2.5-3B-Instruct-GGUF",
        "pattern": "*Q4_K_M*.gguf",
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "license": "qwen-research",
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
        local = snapshot_download(
            repo_id=m["source"],
            allow_patterns=m["pattern"],
            local_dir=f"models/{m['source'].split('/')[-1]}",
        )
        gguf = next(pathlib.Path(local).rglob("*.gguf"))
        size_gb = round(gguf.stat().st_size / 1e9, 2)
        print(f"[upload] gguf: {gguf.name} ({size_gb} GB) -> {target_id}")

        api.upload_file(
            path_or_fileobj=str(gguf),
            path_in_repo=gguf.name,
            repo_id=target_id,
            commit_message=f"Upload {gguf.name} (Q4_K_M)",
        )
        uploaded.append(
            f"- **{m['base_model']}** - `{gguf.name}` - {size_gb} GB - license: {m['license']}"
        )

    readme = f"""---
license: other
pipeline_tag: text-generation
quantized_by: Nexuss0781
---

# SPACE

Single repository holding the Q4_K_M (llama.cpp) GGUF weights benchmarked in the
Space GitHub Actions pipeline.

## Models

""" + "\n".join(uploaded) + """

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